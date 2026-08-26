import unittest

import numpy as np

from tools.analyze_kizz_ctc_alignment import delay_cross_entropies


class AlignmentDiagnosticTest(unittest.TestCase):
    def test_detects_causal_student_delay(self):
        teacher = np.log(
            np.asarray(
                [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
                dtype=np.float64,
            )
        )
        student = np.vstack((teacher[:1], teacher[:-1]))
        losses = delay_cross_entropies(student, teacher, max_delay=2)
        self.assertEqual(int(np.argmin(losses)), 1)
        self.assertLess(losses[1], losses[0])

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            delay_cross_entropies(np.zeros((2, 2)), np.zeros((3, 2)), 1)


if __name__ == "__main__":
    unittest.main()
