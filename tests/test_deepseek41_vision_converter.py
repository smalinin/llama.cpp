import unittest
from types import SimpleNamespace

from conversion import get_model_class
from conversion.base import MmprojModel
from conversion.deepseek import DeepseekV41FlashVisionModel
from gguf import GGMLQuantizationType, LlamaFileType


class TestDeepseekV41VisionConverter(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        vision = {
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": 32,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 2816,
            "patch_size": 14,
            "rope_theta": 10000,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        }
        vision.update(overrides)
        return vision

    def test_mmproj_registry(self):
        self.assertIs(
            get_model_class("DeepseekV41ForCausalLM", mmproj=True),
            DeepseekV41FlashVisionModel,
        )

    def test_nested_vision_config(self):
        model = SimpleNamespace(global_config={"vision_config": self.config()})
        cfg = DeepseekV41FlashVisionModel.get_vision_config(model)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["image_size"], 672)
        self.assertEqual(cfg["max_image_tokens"], 1024)
        self.assertIsNone(cfg["max_wh_ratio"])

    def test_invalid_config_rejected(self):
        for vision, message in (
            (None, "requires a vision_config"),
            (self.config(max_image_tokens=2), "max_image_tokens"),
            (self.config(max_wh_ratio=0), "max_wh_ratio"),
            (self.config(rope_theta=500000), "rope_theta"),
        ):
            with self.subTest(vision=vision):
                model = SimpleNamespace(global_config={"vision_config": vision})
                with self.assertRaisesRegex(ValueError, message):
                    DeepseekV41FlashVisionModel.get_vision_config(model)

    def test_source_manifest(self):
        required = DeepseekV41FlashVisionModel._required_source_tensors(32)
        self.assertEqual(len(required), 266)
        self.assertIn("image_newline", required)
        self.assertNotIn("image_pad", required)
        self.assertIn("vision.blocks.31.mlp.w2.weight", required)

    def test_missing_vision_sentinel_is_rejected(self):
        present = DeepseekV41FlashVisionModel._required_source_tensors(32)
        present.remove("image_newline")
        with self.assertRaisesRegex(ValueError, "missing=\\['image_newline'\\]"):
            DeepseekV41FlashVisionModel._validate_source_manifest(present, 32)

    def test_old_image_pad_tensor_is_rejected(self):
        present = DeepseekV41FlashVisionModel._required_source_tensors(32)
        present.add("image_pad")
        with self.assertRaisesRegex(ValueError, "unexpected=\\['image_pad'\\]"):
            DeepseekV41FlashVisionModel._validate_source_manifest(present, 32)

    def test_bf16_patch_embedding_is_preserved(self):
        model = SimpleNamespace(ftype=LlamaFileType.MOSTLY_BF16)
        qtype = MmprojModel.tensor_force_quant(
            model,
            "vision.patch_embed.proj.weight",
            "v.patch_embd.weight",
            None,
            4,
        )
        self.assertEqual(qtype, GGMLQuantizationType.BF16)


if __name__ == "__main__":
    unittest.main()
