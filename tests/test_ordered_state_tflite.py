import unittest

import numpy as np

from microwakeword.ordered_state_tflite import tflite_output_logits


class OrderedStateTFLiteAdapterTest(unittest.TestCase):
    def test_dequantizes_uint8_logits(self):
        raw = np.arange(23, dtype=np.uint8).reshape(1, 1, 23)
        logits = tflite_output_logits(raw, {"quantization": (0.25, 4)})
        np.testing.assert_allclose(logits, (np.arange(23) - 4) * 0.25)
        self.assertEqual(logits.shape, (23,))

    def test_rejects_missing_quantization_and_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "positive scale"):
            tflite_output_logits(
                np.zeros((1, 1, 23), dtype=np.uint8),
                {"quantization": (0.0, 0)},
            )
        with self.assertRaisesRegex(ValueError, "expected one"):
            tflite_output_logits(
                np.zeros((1, 1, 1), dtype=np.float32),
                {"quantization": (0.0, 0)},
            )


if __name__ == "__main__":
    unittest.main()
