import unittest

import numpy as np

from microwakeword.ordered_state_data import (
    CANONICAL_KIZZ_PHONES,
    OrderedStateExample,
    PhoneSpan,
    TimeSpan,
    example_from_mapping,
    frame_state_targets,
    validate_examples,
)


def aligned_positive():
    phone_spans = []
    for index, phone in enumerate(CANONICAL_KIZZ_PHONES):
        start = 0.3 + index * 0.3
        phone_spans.append(PhoneSpan(start, start + 0.3, phone))
    return OrderedStateExample(
        source_id="speaker/session/clip",
        truth=True,
        duration_s=3.0,
        text="Hi-Fi Kizz",
        phrase_span=TimeSpan(0.3, 2.4),
        phone_spans=tuple(phone_spans),
    )


class OrderedStateDataTest(unittest.TestCase):
    def test_parser_accepts_an_explicit_replacement_phrase_phone_contract(self):
        record = {
            "source_id": "replacement",
            "truth": True,
            "duration_s": 1.0,
            "phrase_span": {"start_s": 0.1, "end_s": 0.8},
            "phone_spans": [
                {"phone": "k", "start_s": 0.1, "end_s": 0.4},
                {"phone": "z", "start_s": 0.4, "end_s": 0.8},
            ],
        }
        example = example_from_mapping(record, expected_phones=("k", "z"))
        self.assertEqual(example.expected_phones, ("k", "z"))

    def test_positive_requires_exact_canonical_phone_sequence(self):
        record = {
            "source_id": "high-five-kids",
            "truth": True,
            "duration_s": 3.0,
            "phrase_span": {"start_s": 0.2, "end_s": 2.5},
            "phone_spans": [
                {"phone": phone, "start_s": 0.2 + i * 0.3, "end_s": 0.5 + i * 0.3}
                for i, phone in enumerate(("h", "aɪ", "f", "aɪ", "v", "k", "ɪ"))
            ],
        }
        with self.assertRaisesRegex(ValueError, "exactly match canonical"):
            example_from_mapping(record)

    def test_phrase_only_span_is_valid_for_sequence_loss_but_not_frame_loss(self):
        example = OrderedStateExample(
            source_id="weak-positive",
            truth=True,
            duration_s=2.0,
            phrase_span=TimeSpan(0.4, 1.6),
        )
        self.assertIsNone(frame_state_targets(example, [0.3, 0.9, 1.7]))

    def test_aligned_phones_map_to_three_ordered_states_each(self):
        example = aligned_positive()
        frame_times = [0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 2.45]
        targets = frame_state_targets(example, frame_times)
        np.testing.assert_array_equal(targets, [2, 3, 4, 5, 6, 7, 1])

    def test_negative_frames_are_all_background(self):
        example = OrderedStateExample("tv/session", False, 4.0)
        np.testing.assert_array_equal(
            frame_state_targets(example, [0.0, 1.0, 3.99]), [0, 0, 0]
        )

    def test_collection_requires_both_classes_and_optional_alignment(self):
        positive = {
            "source_id": "positive",
            "truth": True,
            "duration_s": 2.0,
            "phrase_span": {"start_s": 0.2, "end_s": 1.8},
        }
        negative = {"source_id": "negative", "truth": False, "duration_s": 5.0}
        self.assertEqual(len(validate_examples([positive, negative])), 2)
        with self.assertRaisesRegex(ValueError, "requires aligned"):
            validate_examples([positive, negative], require_phone_alignment=True)

    def test_rejects_non_finite_alignment_metadata(self):
        with self.assertRaisesRegex(ValueError, "span must"):
            TimeSpan(float("nan"), 1.0)
        with self.assertRaisesRegex(ValueError, "frame times"):
            frame_state_targets(
                OrderedStateExample("negative", False, 1.0), [float("nan")]
            )


if __name__ == "__main__":
    unittest.main()
