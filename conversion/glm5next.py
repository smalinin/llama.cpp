from __future__ import annotations

import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf
from .qwen3vl import Glm4VVisionModel


@ModelBase.register("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
@ModelBase.example("zai-org/GLM-5.3-Flash")
class Glm5NextModel(TextModel):

    model_arch = gguf.MODEL_ARCH.GLM5NEXT
    supports_mtp_export = True

    _experts: list[dict[str, Tensor]] | None = None
    _n_main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_nextn_layers = self.hparams.get("num_nextn_predict_layers", 0)
        self.skip_mtp = self.no_mtp or self.n_nextn_layers == 0

        if not self.skip_mtp:
            self.block_count += self.n_nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self.hparams.pop("head_dim", None)

    def set_vocab(self):
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        except ValueError:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.dir_model / "tokenizer.json"))
        self._set_vocab_glm(tokenizer, tokpre="glm5")

    def index_tensors(self, remote_hf_model_id: str | None = None):
        hp = self.hparams.get("text_config", self.hparams)
        type(self)._n_main_layers = hp["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        assert cls._n_main_layers is not None
        m = re.match(r"model\.layers\.(\d+)\.", name)
        is_mtp = m is not None and int(m.group(1)) >= cls._n_main_layers

        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        layer_types = hp["layer_types"]
        n_kv_heads = [0 if t == "linear_attention" else 1 for t in layer_types]
        assert len(n_kv_heads) == hp["num_hidden_layers"]
        # Pad to block_count
        n_kv_heads += [1] * (self.block_count - len(n_kv_heads))
        self.gguf_writer.add_head_count_kv(n_kv_heads)
        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        self.gguf_writer.add_layer_norm_eps(1e-6)

        if not self.skip_mtp:
            self.gguf_writer.add_nextn_predict_layers(self.n_nextn_layers)

        # KDA
        lin = hp["linear_attn_config"]
        assert lin["num_heads"] == hp["num_attention_heads"]
        self.gguf_writer.add_ssm_conv_kernel(lin["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(lin["head_dim"])
        if (lb := lin.get("gate_lower_bound")) is not None:
            self.gguf_writer.add_kda_gate_lower_bound(lb)

        # MLA (nope only)
        assert hp.get("mla_use_nope") and hp["qk_rope_head_dim"] == 0, "expected nope-only MLA"
        kv_lora_rank = hp["kv_lora_rank"]
        qk_rope = hp["qk_rope_head_dim"]
        self.gguf_writer.add_q_lora_rank(hp["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_rope_dimension_count(qk_rope)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope)
        self.gguf_writer.add_value_length(kv_lora_rank)
        self.gguf_writer.add_key_length_mla(hp["qk_nope_head_dim"] + qk_rope)
        self.gguf_writer.add_value_length_mla(hp["v_head_dim"])

        # DSA indexer with k-pool compression
        self.gguf_writer.add_indexer_head_count(hp["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["index_topk"])
        self.gguf_writer.add_indexer_kpool(hp["index_kpool"])
        self.gguf_writer.add_indexer_kpool_select_tail(hp.get("index_kpool_always_select_tail", True))
        if (indexer_types := hp.get("indexer_types")) is not None:
            self.gguf_writer.add_indexer_types([t == "full" for t in indexer_types])

        # mHC
        assert hp.get("mhc", True)
        self.gguf_writer.add_hyper_connection_count(hp["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hp["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hp["hc_eps"])

        # MoE
        self.gguf_writer.add_leading_dense_block_count(hp["first_k_dense_replace"])
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hp["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hp["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hp["norm_topk_prob"])
        if (limit := hp.get("swiglu_limit")) is not None:
            self.gguf_writer.add_swiglu_clamp_exp([limit] * self.block_count)
            self.gguf_writer.add_swiglu_clamp_shexp([limit] * self.block_count)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight" and self.hparams.get("tie_word_embeddings", False):
            return

        # routed experts
        if ".mlp.experts." in name:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) < n_experts * 3:
                return
            for w_name in ("down_proj", "gate_proj", "up_proj"):
                datas: list[Tensor] = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                    datas.append(self._experts[bid].pop(ename))
                merged = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                yield from super().modify_tensors(torch.stack(datas, dim=0), merged, bid)
            return

        # MLA absorption
        if name.endswith("kv_b_proj.weight"):
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            yield from super().modify_tensors(k_b.transpose(1, 2), name.replace("kv_b_proj", "k_b_proj"), bid)
            yield from super().modify_tensors(v_b, name.replace("kv_b_proj", "v_b_proj"), bid)
            return

        # KDA conv1d
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 3:
                d_inner, _, d_conv = data_torch.shape
            elif data_torch.ndim == 2:
                d_inner, d_conv = data_torch.shape
            else:
                raise ValueError(f"unexpected conv1d rank {data_torch.ndim} for {name}")
            data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        if name.endswith(".A_log"):
            n_head = self.hparams["num_attention_heads"]
            data_torch = -torch.exp(data_torch.float().flatten()[:n_head])

        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        if re.search(r"\.(hc_(?:attn|ffn)_(?:fn|base|scale)|index_kpool_compress_(?:ape|gate))$", name):
            yield self.map_tensor_name(name) + ".weight", data_torch
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # keep the small mHC / gating parameters exact
        exact_keys = ("hc_attn_", "hc_ffn_", "indexer_compressor_", "ssm_a", "ssm_dt", "exp_probs_b")
        if new_name.startswith(("blk.", "output_hc")) and any(k in new_name for k in exact_keys):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)
        if not self.mtp_only or not from_dir:
            return
        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            leftover = [k for d in self._experts for k in d.keys()]
            if leftover:
                raise ValueError(f"Unprocessed experts: {leftover}")


@ModelBase.register("Glm5NextForConditionalGeneration")
# [TAG_HF_EXAMPLE_MISSING]
class Glm5NextVisionModel(Glm4VVisionModel):
    """The vision tower is the GLM-OCR ViT under a `model.visual.` prefix.

    Every tensor already maps through the GLM-4V entries. The one structural
    difference is a clamp on the SwiGLU gate and up projections, applied in the
    per-block MLP and again in the merger, so it gets its own projector type
    rather than a flag on glm4v.
    """

    clip_projector_type = gguf.VisionProjectorType.GLM5NEXT

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        # Glm4VVisionModel bypasses Qwen3VLVisionModel entirely, which is also where
        # the merge size is written, so no GLM4V-family mmproj carries this key and
        # clip.cpp falls back to a hardcoded 2. Write it rather than rely on that
        self.gguf_writer.add_vision_spatial_merge_size(int(self.hparams_vision["spatial_merge_size"]))

        self.gguf_writer.add_vision_swiglu_clamp(float(self.hparams_vision["swiglu_limit"]))
