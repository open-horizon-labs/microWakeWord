import unittest

import numpy as np

from tools.report_ordered_state_resources import (
    latency_summary,
    persistent_stream_state_bytes,
    tensor_bytes,
)


class OrderedStateResourceReportTest(unittest.TestCase):
    def test_tensor_and_persistent_state_bytes(self):
        details = [
            {
                "name": "stream/Cast/ReadVariableOp",
                "shape": np.asarray([1, 4, 1, 96]),
                "dtype": np.int8,
            },
            {
                "name": "temporary",
                "shape": np.asarray([1, 8]),
                "dtype": np.float32,
            },
        ]
        self.assertEqual(tensor_bytes(details[0]), 384)
        self.assertEqual(persistent_stream_state_bytes(details), 384)

    def test_latency_summary(self):
        report = latency_summary([1.0, 4.0, 2.0, 3.0])
        self.assertEqual(report["iterations"], 4)
        self.assertEqual(report["median_ms"], 2.5)
        self.assertEqual(report["p95_ms"], 4.0)


if __name__ == "__main__":
    unittest.main()
