from __future__ import annotations

from typing import Iterable, Sequence, TYPE_CHECKING

import json
import re
import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, LazyTorchTensor, gguf, logger
from .qwen import DSparkModel


@ModelBase.register("DSparkMiniMaxDraftModel")
class MiniMaxM3DSparkModel(DSparkModel):
    model_arch = gguf.MODEL_ARCH.DFLASH

    def __init__(self, dir_model, *args, **kwargs):
        config = kwargs.pop("hparams", None)
        if config is None:
            with open(dir_model / "config.json", encoding="utf-8") as f:
                config = json.load(f)
        self._source_config = json.dumps(config, sort_keys=True)
        hparams = {**config, **config.get("text_config", {})}
        hparams.pop("text_config", None)
        if not hparams.get("dspark_use_dense_ffn") or hparams.get("dspark_hybrid_context_fusion"):
            raise ValueError("MiniMax DSpark export requires dense FFN and plain context fusion")
        if not hparams.get("use_gemma_norm") or hparams.get("hidden_act") != "swigluoai":
            raise ValueError("MiniMax DSpark export requires Gemma norms and SwiGLU-OAI")
        if kwargs.get("fp8_as_q8") or kwargs.get("fuse_qkv") or kwargs.get("fuse_gate_up_exps"):
            raise ValueError("MiniMax DSpark lossless export requires unfused BF16/F32 tensors")

        # The draft config inherits target MoE fields, but all draft layers are dense.
        for key in ("num_local_experts", "num_experts_per_tok", "n_shared_experts", "scoring_func"):
            hparams.pop(key, None)
        hparams["intermediate_size"] = hparams["dspark_ffn_hidden_size"]
        hparams["dense_intermediate_size"] = hparams["dspark_ffn_hidden_size"]
        hparams["use_sliding_window"] = True
        hparams["rope_is_neox_style"] = True
        quant_config = dict(hparams.get("quantization_config", {}))
        for key in ("moe_quant_algo", "moe_quant_format", "group_size"):
            quant_config.pop(key, None)
        hparams["quantization_config"] = quant_config
        self._lossless_qtypes: dict[str, gguf.GGMLQuantizationType] = {}
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        if self.ftype not in (gguf.LlamaFileType.MOSTLY_BF16, gguf.LlamaFileType.ALL_F32):
            raise ValueError("MiniMax DSpark lossless export supports --outtype bf16 or f32")

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        name = name.removeprefix("language_model.")
        if not name.startswith("model.dspark."):
            raise ValueError(f"Unexpected MiniMax DSpark tensor: {name}")
        name = name.replace("model.dspark.", "model.", 1).replace(".decoder_layer.", ".")
        name = name.replace("model.final_norm.", "model.norm.", 1)
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_dimension_count(self.hparams["rotary_dim"])
        self.gguf_writer.add_string("dflash.backbone", "minimax-m3")
        self.gguf_writer.add_string("dflash.feed_forward_activation", self.hparams["hidden_act"])
        self.gguf_writer.add_float32("dflash.swiglu_alpha", self.hparams["swiglu_alpha"])
        self.gguf_writer.add_float32("dflash.swiglu_limit", self.hparams["swiglu_limit"])
        self.gguf_writer.add_uint32("dflash.markov_rank", self.hparams["dspark_markov_rank"])
        self.gguf_writer.add_bool("dflash.confidence_head_with_markov", self.hparams["confidence_head_with_markov"])
        self.gguf_writer.add_string("dflash.minimax.source_config", self._source_config)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        data_torch = LazyTorchTensor.to_eager(data_torch).float()
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1.0
        if not torch.isfinite(data_torch).all():
            raise ValueError(f"Non-finite MiniMax DSpark tensor: {name}")

        # MXFP8 power-of-two scales normally decode exactly to BF16. Check every value.
        qtype = gguf.GGMLQuantizationType.F32
        if self.ftype == gguf.LlamaFileType.MOSTLY_BF16 and data_torch.ndim > 1:
            if torch.equal(data_torch, data_torch.bfloat16().float()):
                qtype = gguf.GGMLQuantizationType.BF16
            else:
                logger.info(f"Keeping {name} in F32 to avoid BF16 rounding")
        self._lossless_qtypes[name] = qtype
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        return self._lossless_qtypes[name]


@ModelBase.register("MiniMaxText01ForCausalLM")
@ModelBase.register("MiniMaxM1ForCausalLM")
@ModelBase.example("MiniMaxAI/MiniMax-Text-01", "MiniMaxAI/MiniMax-M1-40k")
class MiniMaxText01Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINIMAX01

    def _get_suppress_tokens(self) -> Sequence[int] | None:
        import json
        from transformers import AutoTokenizer
        from .base import LazyTorchTensor

        # check added tokens embeddings in embeddings tensor for zero-valued embeddings
        # they get in the way of the token sampling process and must be suppressed

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        tokenizer_vocab_size = tokenizer.vocab_size

        with open(self.dir_model / "model.safetensors.index.json", "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        embeddings_tensor_name = "model.embed_tokens.weight"
        embeddings_shard_name = weight_map[embeddings_tensor_name]
        with gguf.utility.SafetensorsLocal(self.dir_model / embeddings_shard_name) as model_shard:
            embeddings_data = model_shard[embeddings_tensor_name]

        embeddings_weights_dtype = LazyTorchTensor._dtype_str_map[embeddings_data.dtype]
        embeddings_weights = torch.from_numpy(embeddings_data.mmap_bytes()).view(embeddings_weights_dtype).reshape(embeddings_data.shape)
        embeddings_vocab_size = embeddings_weights.shape[0]

        embeddings_added_tokens = embeddings_weights[tokenizer_vocab_size:embeddings_vocab_size]
        embeddings_zero_rows = torch.all(embeddings_added_tokens == 0, dim=1)
        tokens_zero_embeddings_ids = (torch.nonzero(embeddings_zero_rows, as_tuple=False).flatten() + tokenizer_vocab_size).tolist()

        return tokens_zero_embeddings_ids

    def set_vocab(self) -> None:
        from pathlib import Path

        self._set_vocab_gpt2()

        for tmpl_file in [
            self.dir_model / "chat_template.jinja",
            Path(__file__).parent.parent / "models" / "templates" / "MiniMax-M1.jinja"
        ]:
            if tmpl_file.is_file():
                self.gguf_writer.add_chat_template(tmpl_file.read_text(encoding="utf-8"))
                logger.info(f"Chat template overridden with {tmpl_file}.")
                break

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        suppress_tokens = self._get_suppress_tokens()
        if suppress_tokens:
            logger.info(f"Suppressing tokens with zero embeddings {suppress_tokens}")
            self.gguf_writer.add_suppress_tokens(suppress_tokens)

        layernorm_full_attention_alpha = self.hparams["layernorm_full_attention_alpha"]
        layernorm_full_attention_beta = self.hparams["layernorm_full_attention_beta"]
        layernorm_linear_attention_alpha = self.hparams["layernorm_linear_attention_alpha"]
        layernorm_linear_attention_beta = self.hparams["layernorm_linear_attention_beta"]
        layernorm_mlp_alpha = self.hparams["layernorm_mlp_alpha"]
        layernorm_mlp_beta = self.hparams["layernorm_mlp_beta"]
        assert layernorm_full_attention_alpha == layernorm_linear_attention_alpha == layernorm_mlp_alpha
        assert layernorm_full_attention_beta == layernorm_linear_attention_beta == layernorm_mlp_beta == 1.0
        # we do not store the layernorm betas as they are all 1.0
        # layernorm alphas are stored as single residual_scale hparam
        self.gguf_writer.add_residual_scale(layernorm_full_attention_alpha)

        self.gguf_writer.add_rope_dimension_count(self.hparams["rotary_dim"])

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    yield from super().modify_tensors(data_torch, new_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM2ForCausalLM")
@ModelBase.example("MiniMaxAI/MiniMax-M2")
class MiniMaxM2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINIMAXM2
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_expert_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_rope_dimension_count(self.find_hparam(["rotary_dim"]))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # merge expert weights
        if "block_sparse_moe.experts." in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]

            # not enough expert weights to merge
            if len(expert_cache) < n_experts * len(expert_weights):
                return

            for w_name in expert_weights:
                datas: list[Tensor] = []

                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]

                data_torch = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"
                new_name = self.map_tensor_name(merged_name)
                yield from super().modify_tensors(data_torch, new_name, bid)

            del self._experts_cache[bid]
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")
@ModelBase.example("MiniMaxAI/MiniMax-M3")
class MiniMaxM3Model(MiniMaxM2Model):
    model_arch = gguf.MODEL_ARCH.MINIMAXM3

    def __init__(self, dir_model, *args, **kwargs):
        if kwargs.get("hparams") is None:
            with open(dir_model / "config.json", encoding="utf-8") as f:
                config = json.load(f)
            if config.get("quantization_config", {}).get("quant_method") == "mxfp8":
                kwargs["hparams"] = config
        super().__init__(dir_model, *args, **kwargs)

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        match = re.search(r"block_sparse_moe\.experts\.(\d+)\.(w[123])\.", name)
        if match and name.endswith((".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale")):
            projection = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[match[2]]
            name = name[:match.start()] + f"mlp.experts.{match[1]}.{projection}." + name[match.end():]
        return super().filter_tensors((name, gen))

    def _generate_nvfp4_tensors(self):
        quant_config = self.hparams.get("quantization_config", {})
        if quant_config.get("quant_method") != "mxfp8" or quant_config.get("moe_quant_algo") != "NVFP4":
            return super()._generate_nvfp4_tensors()
        if quant_config.get("group_size") != 16:
            raise ValueError("MiniMax NVFP4 requires group_size 16")
        if self.fuse_gate_up_exps or self.fuse_qkv or self.is_big_endian:
            raise ValueError("MiniMax mixed NVFP4 export requires unfused little-endian tensors")

        n_experts = self.hparams["num_local_experts"]
        groups = {}
        consumed = []
        for name, weight_gen in self.model_tensors.items():
            if ".experts." not in name or not name.endswith(".weight"):
                continue
            match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name)
            if match is None:
                raise ValueError(f"Unexpected MiniMax NVFP4 expert {name}")
            bid, expert_id = int(match[1]), int(match[2])
            projection = match[3]
            scale_name = name.replace(".weight", ".weight_scale")
            scale2_name = name.replace(".weight", ".weight_scale_2")
            input_name = name.replace(".weight", ".input_scale")
            scale_gen = self.model_tensors[scale_name]
            weight, scale = weight_gen(), scale_gen()
            if weight.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
                raise ValueError(f"Invalid NVFP4 dtypes for {name}")
            rows, packed_cols = weight.shape
            if packed_cols % 32 or tuple(scale.shape) != (rows, packed_cols // 8):
                raise ValueError(f"Invalid NVFP4 weight/scale shapes for {name}")
            scale2 = float(LazyTorchTensor.to_eager(self.model_tensors[scale2_name]()).float().item())
            input_scale = float(LazyTorchTensor.to_eager(self.model_tensors[input_name]()).float().item())
            if not np.isfinite(scale2) or scale2 <= 0 or not np.isfinite(input_scale) or input_scale <= 0:
                raise ValueError(f"Invalid NVFP4 global scales for {name}")

            def load_chunk(w=weight_gen, s=scale_gen):
                weight = LazyTorchTensor.to_eager(w())
                scale = LazyTorchTensor.to_eager(s())
                bits = scale.view(torch.uint8)
                if torch.any((bits & 128) != 0) or torch.any(bits == 127):
                    raise ValueError("NVFP4 local scales must be finite and nonnegative")
                return self._nvfp4_pack(weight, scale)[0]

            group = groups.setdefault((bid, projection), {})
            if expert_id in group:
                raise ValueError(f"Duplicate expert {expert_id} in layer {bid}")
            group[expert_id] = (load_chunk, rows, packed_cols * 2, scale2, input_scale)
            consumed.extend((name, scale_name, scale2_name, input_name))

        expected = {(bid, projection) for bid, is_moe in enumerate(self.hparams["moe_layer_freq"]) if is_moe for projection in ("gate_proj", "up_proj", "down_proj")}
        if set(groups) != expected:
            raise ValueError("MiniMax NVFP4 expert groups do not match moe_layer_freq")
        for (bid, projection), group in sorted(groups.items()):
            if set(group) != set(range(n_experts)):
                raise ValueError(f"Incomplete MiniMax expert group {bid}/{projection}")
            experts = [group[i] for i in range(n_experts)]
            rows, cols = experts[0][1:3]
            if any(expert[1:3] != (rows, cols) for expert in experts):
                raise ValueError(f"Inconsistent expert shapes in layer {bid}")
            raw = gguf.LazyChunkedTensor([expert[0] for expert in experts], (n_experts, rows, cols // 64 * 36), np.uint8)
            new_name = self.map_tensor_name(f"model.layers.{bid}.mlp.experts.{projection}.weight")
            self.gguf_writer.add_tensor(new_name, raw, raw_dtype=gguf.GGMLQuantizationType.NVFP4)
            self._write_scales_tensor(new_name.replace(".weight", ".scale"), [expert[3] for expert in experts])
            self._write_scales_tensor(new_name.replace(".weight", ".input_scale"), [expert[4] for expert in experts])
            logger.info("Scheduled %s as NVFP4, %d experts, streamed one expert at a time", new_name, n_experts)
        for name in consumed:
            del self.model_tensors[name]

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".indexer." in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_expert_shared_count(self.find_hparam(["n_shared_experts"]))
        self.gguf_writer.add_expert_weights_scale(self.find_hparam(["routed_scaling_factor"]))
        self.gguf_writer.add_expert_weights_norm(True)

        sac = self.find_hparam(["sparse_attention_config"])
        pattern = sac.get("sparse_attention_freq")
        if pattern is not None:
            if len(pattern) != self.block_count or any(value not in (0, 1) for value in pattern):
                raise ValueError("Invalid MiniMax sparse attention pattern")
            self.gguf_writer.add_attention_sparse_pattern([bool(value) for value in pattern])
        self.gguf_writer.add_indexer_head_count(sac["sparse_num_index_heads"])
        self.gguf_writer.add_indexer_key_length(sac["sparse_index_dim"])
        self.gguf_writer.add_indexer_top_k(sac["sparse_topk_blocks"])
        self.gguf_writer.add_indexer_block_size(sac["sparse_block_size"])
        self.gguf_writer.add_indexer_local_blocks(sac["sparse_local_block"])

        moe_layer_freq = self.find_hparam(["moe_layer_freq"])
        n_dense = 0
        for v in moe_layer_freq:
            if v == 0:
                n_dense += 1
            else:
                break
        self.gguf_writer.add_leading_dense_block_count(n_dense)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # Gemma-style (1 + w) RMSNorm: bake the +1 in so llama.cpp can use plain RMSNorm
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1.0

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3VLForConditionalGeneration")
@ModelBase.example("MiniMaxAI/MiniMax-M3")
class MiniMaxM3VisionModel(MmprojModel):
    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        # keep only the vision-side tensors; text / mtp / sparse-index are dropped
        if not name.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")):
            return None
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MINIMAXM3)
        self.gguf_writer.add_vision_use_gelu(True)

        # the ViT carries its own LayerNorm eps (text tower uses a different one)
        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.hparams_vision.get("layer_norm_eps", 1e-5)
        )

        comp = self.hparams_vision.get("img_token_compression_config", {})
        merge_size = comp.get("spatial_merge_size", 2)
        self.gguf_writer.add_vision_spatial_merge_size(int(merge_size))

        for filename in ("config.json", "preprocessor_config.json"):
            self.gguf_writer.add_string(f"clip.vision.minimax.{filename.removesuffix('.json')}", (self.dir_model / filename).read_text(encoding="utf-8"))

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # The original Conv3D kernel is F32; keep both temporal slices exact.
        if name == "vision_tower.vision_model.embeddings.patch_embedding.weight" and self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch, name, bid):
        assert self.hparams_vision is not None

        # Conv3d patch embed -> Conv2d slices
        if name == "vision_tower.vision_model.embeddings.patch_embedding.weight":
            if data_torch.ndim != 5:
                raise ValueError(f"unexpected patch_embedding rank {data_torch.ndim} for {name}")
            kt = data_torch.shape[2]
            base = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH]
            for t in range(kt):
                suffix = ".weight" if t == 0 else f".weight.{t}"
                yield (base + suffix, data_torch[:, :, t, ...])
            return

        # Permute ViT q/k. HF [Ta Ha Wa | Tb Hb Wb | pad] reorder to [Ta Tb | Ha Hb | Wa Wb | pad].
        for new_name, tensor in super().modify_tensors(data_torch, name, bid):
            if ".attn_q." in new_name or ".attn_k." in new_name:
                tensor = self._permute_vit_qk(tensor, new_name)
            yield new_name, tensor

    def _permute_vit_qk(self, t: "Tensor", new_name: str) -> "Tensor":
        assert self.hparams_vision is not None
        n_head = self.hparams_vision["num_attention_heads"]
        d_head = t.shape[0] // n_head
        axis_dim = 2 * ((2 * (d_head // 2) // 3) // 2)
        ah = axis_dim // 2
        half = 3 * ah
        perm = []
        perm += list(range(0, ah))
        perm += list(range(half, half + ah))
        perm += list(range(ah, 2 * ah))
        perm += list(range(half + ah, half + 2 * ah))
        perm += list(range(2 * ah, 3 * ah))
        perm += list(range(half + 2 * ah, half + 3 * ah))
        perm += list(range(2 * half, d_head))

        assert axis_dim % 2 == 0
        assert 3 * axis_dim <= d_head
        assert len(perm) == d_head
        assert sorted(perm) == list(range(d_head)), "perm is not a bijection of d_head"
        assert t.shape[0] == n_head * d_head, f"{new_name}: {t.shape[0]} != {n_head}*{d_head}"
        assert d_head == 80

        idx = torch.tensor(perm, dtype=torch.long)
        if t.ndim == 2:
            return t.reshape(n_head, d_head, t.shape[1])[:, idx, :].reshape(t.shape)
        return t.reshape(n_head, d_head)[:, idx].reshape(t.shape)
