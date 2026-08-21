import argparse
import unittest

from microwakeword import mixednet


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


if __name__ == "__main__":
    unittest.main()
