import unittest
from types import SimpleNamespace

from conversion.qwen4exp import Qwen4ExpTextModel


class TestQwen4ExpConverter(unittest.TestCase):
    def test_eos_token_id(self):
        for eos, expected in ((248044, 248044), ([248044, 248046], 248044)):
            with self.subTest(eos=eos):
                model = SimpleNamespace(hparams={"eos_token_id": eos})
                self.assertEqual(Qwen4ExpTextModel._eos_token_id(model), expected)


if __name__ == "__main__":
    unittest.main()
