import argparse
import unittest

from microwakeword import mixednet
from microwakeword.model_train_eval import apply_model_parameters


class MixedNetDefaultsTest(unittest.TestCase):
    def test_layer_parameter_defaults_have_matching_lengths(self):
        parser = argparse.ArgumentParser()
        mixednet.model_parameters(parser)
        flags = parser.parse_args([])

        layer_parameters = (
            mixednet.parse(flags.pointwise_filters),
            mixednet.parse(flags.repeat_in_block),
            mixednet.parse(flags.mixconv_kernel_sizes),
            mixednet.parse(flags.residual_connection),
        )

        self.assertEqual({len(values) for values in layer_parameters}, {4})

    def test_training_config_can_pin_model_architecture(self):
        parser = argparse.ArgumentParser()
        mixednet.model_parameters(parser)
        flags = parser.parse_args([])

        apply_model_parameters(
            {
                "model_parameters": {
                    "first_conv_filters": 48,
                    "first_conv_kernel_size": 5,
                    "stride": 3,
                    "pointwise_filters": "96,96,96,96",
                }
            },
            flags,
        )

        self.assertEqual(flags.first_conv_filters, 48)
        self.assertEqual(flags.first_conv_kernel_size, 5)
        self.assertEqual(flags.stride, 3)
        self.assertEqual(flags.pointwise_filters, "96,96,96,96")

    def test_training_config_rejects_unknown_or_non_model_flags(self):
        parser = argparse.ArgumentParser()
        mixednet.model_parameters(parser)
        flags = parser.parse_args([])

        with self.assertRaisesRegex(ValueError, "unknown model parameters"):
            apply_model_parameters({"model_parameters": {"missing": 1}}, flags)

        flags.train = 0
        with self.assertRaisesRegex(ValueError, "non-model flags"):
            apply_model_parameters({"model_parameters": {"train": 1}}, flags)


if __name__ == "__main__":
    unittest.main()
