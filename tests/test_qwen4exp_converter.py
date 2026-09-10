import unittest
from types import SimpleNamespace

from conversion.qwen4exp import Qwen4ExpTextModel


class TestQwen4ExpConverter(unittest.TestCase):
    def test_output_gate_type(self):
        for hparams in (
            {"output_gate_type": "sigmoid", "hidden_act": "silu"},
            {"output_gate_type": None, "hidden_act": "sigmoid"},
        ):
            with self.subTest(hparams=hparams):
                model = SimpleNamespace(hparams=hparams)
                self.assertEqual(Qwen4ExpTextModel._output_gate_type(model), "sigmoid")

        for hparams in (
            {"output_gate_type": "silu", "hidden_act": "silu"},
            {"hidden_act": "silu"},
        ):
            with self.subTest(hparams=hparams):
                model = SimpleNamespace(hparams=hparams)
                with self.assertRaisesRegex(ValueError, "requires a sigmoid output gate"):
                    Qwen4ExpTextModel._output_gate_type(model)

    def test_eos_token_id(self):
        for eos, expected in ((248044, 248044), ([248044, 248046], 248044)):
            with self.subTest(eos=eos):
                model = SimpleNamespace(hparams={"eos_token_id": eos})
                self.assertEqual(Qwen4ExpTextModel._eos_token_id(model), expected)


if __name__ == "__main__":
    unittest.main()
