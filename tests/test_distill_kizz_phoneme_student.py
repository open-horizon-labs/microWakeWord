import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from microwakeword.phoneme_student import compact_phone_contract
from microwakeword.phoneme_student import student_output_times_seconds
from microwakeword.kizz_viterbi_decoder import exhaustive_suffix_score
from microwakeword.ctc_forward import exhaustive_sliding_forward_score
from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.distill_kizz_phoneme_student import (
    WINDOW_LENGTHS_SECONDS,
    WINDOW_LENGTHS_FRAMES,
    _student_scores,
    checkpoint_binding,
    checkpoint_selection_key,
    channel_consistency_loss,
    paired_path_consistency_loss,
    collision_path_supervision,
    deployment_path_scores,
    delayed_occupation_loss,
    deployable_causal_mask,
    distillation_loss,
    expected_schedule_counts,
    map_ordered_targets,
    multichannel_checkpoint_selection_key,
    positive_indices_by_provider,
    provenance_ref,
    require_cache_binding,
    require_overlay_parent_binding,
    student_decoder_contract,
    student_decoder_contract_hash,
    student_flags,
    require_teacher_gates,
    sha256_file,
    student_architecture_contract,
    student_flags_for_architecture,
    teacher_sequence_score_targets,
    strict_collision_negative_loss,
    teacher_sequence_ranking_loss,
    teacher_sequence_listwise_loss,
    temporal_representation_loss,
    load_temporal_representation_cache,
    utterance_representation_loss,
    validate_causal_loss_contract,
    validate_reference_causal_contract,
)
from tools.package_kizz_phoneme_student_firmware import EXPECTED_WINDOWS
from tools.qualify_kizz_phoneme_student import WINDOW_LENGTHS as QUALIFICATION_WINDOWS
from tools.qualify_kizz_phoneme_student import score_features


class DistillKizzPhonemeStudentTests(unittest.TestCase):
    def test_paired_path_consistency_anchors_deployed_fit_and_margin(self):
        import tensorflow as tf

        contract = compact_phone_contract()
        clean = tf.random.stateless_normal((2, 66, 20), seed=(7, 11))
        device = tf.tensor_scatter_nd_add(
            clean, indices=[[0, 30, 1], [1, 40, 2]], updates=[0.25, -0.25]
        )
        endpoints = tf.constant([54, 54], dtype=tf.int32)
        equal = paired_path_consistency_loss(
            clean,
            clean,
            endpoints,
            tf.constant([1.0, 1.0]),
            contract,
            algorithm="forward_sum_ctc",
        )
        different = paired_path_consistency_loss(
            device,
            clean,
            endpoints,
            tf.constant([1.0, 1.0]),
            contract,
            algorithm="forward_sum_ctc",
        )
        self.assertAlmostEqual(float(equal), 0.0, places=6)
        self.assertGreater(float(different), 0.0)

    def test_causal_teacher_sampling_excludes_undeployable_prefixes(self):
        scores = np.zeros((2, 66), dtype=np.float32)
        valid = deployable_causal_mask(scores)
        self.assertFalse(valid[:, : min(WINDOW_LENGTHS_FRAMES) - 1].any())
        self.assertTrue(valid[:, min(WINDOW_LENGTHS_FRAMES) - 1 :].all())

    def test_causal_transfer_rejects_clip_label_ranking_losses(self):
        with self.assertRaisesRegex(ValueError, "clip-label ranking"):
            validate_causal_loss_contract(
                teacher_causal_window_cache=Path("teacher-cache"),
                ranking_weight=0.0,
                tail_ranking_weight=1.0,
            )
        validate_causal_loss_contract(
            teacher_causal_window_cache=Path("teacher-cache"),
            ranking_weight=0.0,
            tail_ranking_weight=0.0,
        )

    def test_reference_causal_cache_binds_architecture_and_features(self):
        metadata = {
            "source_student": {"architecture": {"architecture_id": "memory"}},
            "corpus": {"features_sha256": "features"},
        }
        validate_reference_causal_contract(
            metadata,
            architecture={"architecture_id": "memory"},
            features_sha256="features",
        )
        with self.assertRaisesRegex(ValueError, "different inputs"):
            validate_reference_causal_contract(
                metadata,
                architecture={"architecture_id": "control"},
                features_sha256="features",
            )

    def test_listwise_teacher_loss_prefers_teacher_order(self):
        import tensorflow as tf

        teacher = tf.constant([2.0, 0.5, -1.0, 99.0])
        mask = tf.constant([1.0, 1.0, 1.0, 0.0])
        aligned = teacher_sequence_listwise_loss(
            tf.constant([2.0, 0.5, -1.0, -99.0]),
            teacher,
            mask,
            temperature=0.5,
        )
        reversed_order = teacher_sequence_listwise_loss(
            tf.constant([-1.0, 0.5, 2.0, 99.0]),
            teacher,
            mask,
            temperature=0.5,
        )
        self.assertLess(float(aligned), float(reversed_order))

    def test_checkpoint_selection_prefers_fewer_false_accepts_at_recall_floor(self):
        common = {"qualified": False, "recall": 0.90}
        fewer_false_accepts = checkpoint_selection_key(
            {**common, "false_accepts_at_recall_floor": 2}, 0.75, -0.5
        )
        more_false_accepts = checkpoint_selection_key(
            {**common, "false_accepts_at_recall_floor": 8}, 0.75, 0.5
        )
        self.assertGreater(fewer_false_accepts, more_false_accepts)

    def test_checkpoint_selection_handles_unreached_recall_floor(self):
        key = checkpoint_selection_key(
            {
                "qualified": False,
                "recall": 0.0,
                "false_accepts_at_recall_floor": None,
            },
            0.0,
            None,
        )
        self.assertTrue(np.isfinite(key).all())

    def test_multichannel_selection_cannot_hide_device_failure_behind_clean_score(self):
        point = {
            "qualified": True,
            "recall": 0.95,
            "false_accepts_at_recall_floor": 0,
        }
        device_pass = multichannel_checkpoint_selection_key(
            point, 0.92, 0.91, 10, 10, -0.2
        )
        tempting_clean_only_checkpoint = multichannel_checkpoint_selection_key(
            point, 1.0, 0.55, 6, 10, 0.5
        )
        self.assertGreater(device_pass, tempting_clean_only_checkpoint)

    def test_channel_consistency_ignores_unpaired_rows_and_penalizes_drift(self):
        import tensorflow as tf

        clean = tf.zeros((2, 4, 3), dtype=tf.float32)
        identical = channel_consistency_loss(clean, clean, tf.constant([1.0, 0.0]))
        device = tf.tensor_scatter_nd_update(clean, [[0, 0, 0]], [4.0])
        drift = channel_consistency_loss(device, clean, tf.constant([1.0, 0.0]))
        ignored = channel_consistency_loss(device, clean, tf.constant([0.0, 0.0]))
        self.assertEqual(float(identical), 0.0)
        self.assertGreater(float(drift), 0.0)
        self.assertEqual(float(ignored), 0.0)

    def test_checkpoint_forward_sum_scores_match_deployment_normalization(self):
        contract = compact_phone_contract()
        rng = np.random.default_rng(238)
        logits = rng.normal(size=(1, 66, len(contract["tokens"]))).astype(np.float32)
        offsets = rng.normal(size=(1, 66, 1)).astype(np.float32) * 10

        class FixedModel:
            def __call__(self, values, training=False):
                return np.repeat(logits + offsets, len(values), axis=0)

        features = np.zeros((1, 260, 40), dtype=np.float32)
        selected = _student_scores(
            FixedModel(),
            features,
            contract,
            1,
            decoder_algorithm="forward_sum_ctc",
        )[0]
        deployed = score_features(
            FixedModel(),
            features[0],
            contract,
            decoder_algorithm="forward_sum_ctc",
        )
        self.assertAlmostEqual(float(selected), float(deployed), places=5)

    def test_teacher_sequence_ranking_transfers_cross_label_order(self):
        import tensorflow as tf

        teacher = tf.constant([1.0, -1.0])
        mask = tf.ones((2,))
        ordered = teacher_sequence_ranking_loss(tf.constant([1.0, -1.0]), teacher, mask)
        reversed_order = teacher_sequence_ranking_loss(
            tf.constant([-1.0, 1.0]), teacher, mask
        )
        self.assertLess(float(ordered), float(reversed_order))

    def test_projected_representation_loss_accepts_signed_teacher_space(self):
        import tensorflow as tf

        student = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
        teacher = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
        loss = utterance_representation_loss(student, teacher, tf.constant([1.0, 1.0]))
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_delayed_occupation_selects_one_causal_sequence_shift(self):
        import tensorflow as tf

        teacher = np.full((1, 4, 3), -20.0, dtype=np.float32)
        student = np.full((1, 4, 3), -20.0, dtype=np.float32)
        teacher[0, np.arange(4), [0, 1, 2, 0]] = 0.0
        student[0, np.arange(4), [2, 0, 1, 2]] = 0.0
        no_delay = delayed_occupation_loss(
            tf.constant(student),
            tf.constant(teacher),
            tf.ones((1,)),
            max_delay_frames=0,
        )
        one_frame = delayed_occupation_loss(
            tf.constant(student),
            tf.constant(teacher),
            tf.ones((1,)),
            max_delay_frames=1,
        )
        self.assertGreater(float(no_delay), 10.0)
        self.assertLess(float(one_frame), 1e-4)

    def test_explicit_collision_loss_cannot_escape_through_low_canonical_fit(self):
        import tensorflow as tf

        loss = strict_collision_negative_loss(
            tf.constant([-0.40, 0.25, -0.50], dtype=tf.float32),
            tf.constant([1.0, 1.0, 0.0], dtype=tf.float32),
            required_margin=0.10,
        )
        # The first explicit collision is safely below -0.10.  The second is
        # still canonical-favoring and contributes 0.35, independent of its
        # absolute canonical score.  The masked third row contributes nothing.
        self.assertAlmostEqual(float(loss), 0.175, places=6)

    def test_collision_supervision_binds_named_paths_and_leaves_quiz_generic(self):
        contract = compact_phone_contract()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "source_id": "patrol",
                                "semantic_label": "phonetic_collision",
                                "render_text": "Kizz patrol",
                            },
                            {
                                "source_id": "quiz",
                                "semantic_label": "phonetic_collision",
                                "render_text": "Quiz Control",
                            },
                        ]
                    }
                )
            )
            indexes, report = collision_path_supervision(
                [
                    {
                        "source_group": "kizz_control_phonetic_collision",
                        "parent_source_id": "patrol",
                    },
                    {
                        "source_group": "kizz_control_phonetic_collision",
                        "parent_source_id": "quiz",
                    },
                    {"source_group": "public_speech"},
                ],
                source,
                contract,
            )
        path_order = list(contract["collision_paths"])
        self.assertEqual(indexes.tolist(), [path_order.index("kizpatrol"), -1, -1])
        self.assertEqual(report["generic_text_counts"], {"Quiz Control": 1})

    def test_decoder_windows_match_distillation_qualification_and_packaging(self):
        self.assertEqual(list(WINDOW_LENGTHS_FRAMES), list(QUALIFICATION_WINDOWS))
        self.assertEqual(list(WINDOW_LENGTHS_FRAMES), EXPECTED_WINDOWS)

    def test_distillation_loss_supports_variable_collision_path_lengths(self):
        import tensorflow as tf

        contract = compact_phone_contract()
        batch = 8
        frames = 66
        count = len(contract["tokens"])
        logits = tf.zeros((batch, frames, count), dtype=tf.float32)
        hard = tf.zeros((batch, frames), dtype=tf.int32)
        teacher = tf.nn.log_softmax(logits, axis=-1)
        teacher_mask = tf.ones((batch,), dtype=tf.float32)
        labels = tf.cast(tf.range(batch) % 2, tf.float32)

        @tf.function
        def compute():
            return distillation_loss(
                logits,
                hard,
                teacher,
                teacher_mask,
                labels,
                contract,
                hard_weight=0.5,
                teacher_weight=1.0,
                ctc_weight=0.2,
                collision_weight=0.2,
                negative_weight=0.2,
                negative_score_target=-1.0,
            )[0]

        self.assertTrue(np.isfinite(float(compute())))

    def test_vectorized_training_score_matches_portable_deployment_decoder(self):
        import tensorflow as tf

        contract = compact_phone_contract()
        logits = (
            np.random.default_rng(238)
            .normal(size=(2, 66, len(contract["tokens"])))
            .astype(np.float32)
        )
        canonical, margin = deployment_path_scores(tf.constant(logits), contract)
        normalized = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = normalized - np.log(np.exp(normalized).sum(axis=-1, keepdims=True))
        expected = [
            exhaustive_suffix_score(
                sequence,
                contract,
                window_lengths=WINDOW_LENGTHS_FRAMES,
                beta=-1.0e9,
            )
            for sequence in log_probs
        ]
        np.testing.assert_allclose(
            canonical.numpy(), [item.canonical_fit for item in expected], atol=1e-5
        )
        np.testing.assert_allclose(
            margin.numpy(), [item.collision_margin for item in expected], atol=1e-5
        )

    def test_teacher_sequence_targets_record_canonical_fit_and_margin(self):
        contract = compact_phone_contract()
        logits = (
            np.random.default_rng(239)
            .normal(size=(1, 66, len(contract["tokens"])))
            .astype(np.float32)
        )
        normalized = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = normalized - np.log(np.exp(normalized).sum(axis=-1, keepdims=True))
        targets = teacher_sequence_score_targets(log_probs, contract)
        expected = exhaustive_suffix_score(
            log_probs[0],
            contract,
            window_lengths=WINDOW_LENGTHS_FRAMES,
            beta=-1.0e9,
        )
        np.testing.assert_allclose(
            targets[0], [expected.canonical_fit, expected.collision_margin], atol=1e-6
        )

    def test_vectorized_forward_sum_training_score_matches_reference_suffixes(self):
        import tensorflow as tf

        contract = compact_phone_contract()
        logits = (
            np.random.default_rng(240)
            .normal(size=(2, 66, len(contract["tokens"])))
            .astype(np.float32)
        )
        canonical, margin = deployment_path_scores(
            tf.constant(logits), contract, algorithm="forward_sum_ctc"
        )
        normalized = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = normalized - np.log(np.exp(normalized).sum(axis=-1, keepdims=True))
        expected = []
        for sequence in log_probs:
            candidates = [
                exhaustive_sliding_forward_score(
                    sequence[-length:],
                    contract,
                    window_lengths=(length,),
                    hop=1,
                    beta=-1.0e9,
                )
                for length in WINDOW_LENGTHS_FRAMES
            ]
            expected.append(
                max(
                    candidates,
                    key=lambda item: (
                        item.canonical_fit,
                        item.collision_margin,
                    ),
                )
            )
        np.testing.assert_allclose(
            canonical.numpy(), [item.canonical_fit for item in expected], atol=2e-5
        )
        np.testing.assert_allclose(
            margin.numpy(), [item.collision_margin for item in expected], atol=2e-5
        )

    def test_training_score_honors_per_example_streaming_endpoints(self):
        import tensorflow as tf

        contract = compact_phone_contract()
        logits = (
            np.random.default_rng(241)
            .normal(size=(2, 66, len(contract["tokens"])))
            .astype(np.float32)
        )
        endpoints = np.asarray([38, 57], dtype=np.int32)
        canonical, margin = deployment_path_scores(
            tf.constant(logits),
            contract,
            algorithm="max_add_ctc_viterbi",
            endpoints=tf.constant(endpoints),
        )
        expected = []
        for sequence, endpoint in zip(logits, endpoints):
            expected.append(
                exhaustive_suffix_score(
                    sequence[:endpoint],
                    contract,
                    window_lengths=WINDOW_LENGTHS_FRAMES,
                    beta=-1.0e9,
                )
            )
        np.testing.assert_allclose(
            canonical.numpy(), [item.canonical_fit for item in expected], atol=1e-5
        )
        np.testing.assert_allclose(
            margin.numpy(), [item.collision_margin for item in expected], atol=1e-5
        )

    def test_hard_gate_requires_bound_continuous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip.json"
            clip.write_text(
                json.dumps(
                    {
                        "gate_scope": "teacher_clip_and_anchor_prequalification",
                        "qualified": True,
                        "phones": {"phrase_id": "kizz-control"},
                        "counts": {"natural_positive": 24, "false_wake_accepted": 0},
                        "validation_operating_point": {"recall": 0.95},
                        "model": {
                            "weights_sha256": "a" * 64,
                            "config_sha256": "c" * 64,
                            "tokenizer_vocab_sha256": "v" * 64,
                        },
                    }
                )
            )
            continuous = root / "continuous.json"
            continuous.write_text(
                json.dumps(
                    {
                        "gate_scope": "untouched_continuous_qualification",
                        "qualified": True,
                        "teacher_qualification": {"report_sha256": sha256_file(clip)},
                        "counts": {"exposure_hours": 100.0, "faph_upper_95": 0.03},
                        "model": {
                            "weights_sha256": "a" * 64,
                            "config_sha256": "c" * 64,
                            "tokenizer_vocab_sha256": "v" * 64,
                        },
                    }
                )
            )
            require_teacher_gates(clip, continuous)
            payload = json.loads(continuous.read_text())
            payload["teacher_qualification"]["report_sha256"] = "b" * 64
            continuous.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not bound"):
                require_teacher_gates(clip, continuous)

    def test_ordered_overlay_targets_map_each_real_topology_state_to_compact_ids(self):
        contract = compact_phone_contract()
        topology = OrderedStateTopology(KIZZ_CONTROL.phones, states_per_phone=2)
        provenance = {
            "states_per_phone": 2,
            "state_count": topology.state_count,
            "wake_phrase": {
                "phrase_id": "kizz-control",
                "phones": list(KIZZ_CONTROL.phones),
            },
            "target_frame_times_seconds": (0.015 + 0.030 * np.arange(87)).tolist(),
        }
        values = np.full((1, 87), topology.background_index, dtype=np.int32)
        target_times = np.asarray(provenance["target_frame_times_seconds"])
        student_times = student_output_times_seconds(
            student_flags(len(contract["tokens"])), 66
        )
        output_positions = list(range(20, 20 + topology.state_count))
        for state_id, output_position in enumerate(output_positions):
            source_position = int(
                np.abs(target_times - student_times[output_position]).argmin()
            )
            values[0, source_position] = state_id
        mapped = map_ordered_targets(values, provenance, contract)
        self.assertEqual(mapped.shape, (1, 66))
        expected = [contract["blank_id"], contract["blank_id"]]
        expected.extend(
            token_id
            for token_id in contract["canonical_path"]
            for _ in range(topology.states_per_phone)
        )
        self.assertEqual(mapped[0, 20:42].tolist(), expected)
        self.assertEqual(contract["canonical_path"][0], contract["canonical_path"][3])

    def test_ordered_overlay_target_mapping_rejects_contract_and_topology_drift(self):
        contract = compact_phone_contract()
        provenance = {
            "states_per_phone": 2,
            "state_count": 22,
            "wake_phrase": {
                "phrase_id": "kizz-control",
                "phones": list(KIZZ_CONTROL.phones),
            },
            "target_frame_times_seconds": (0.015 + 0.030 * np.arange(87)).tolist(),
        }
        values = np.zeros((1, 87), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "compact phone contract"):
            map_ordered_targets(values, provenance, {**contract, "blank_id": 99})
        with self.assertRaisesRegex(ValueError, "state_count"):
            map_ordered_targets(values, {**provenance, "state_count": 23}, contract)
        with self.assertRaisesRegex(ValueError, "phone topology"):
            map_ordered_targets(
                values,
                {
                    **provenance,
                    "wake_phrase": {
                        **provenance["wake_phrase"],
                        "phones": list(KIZZ_CONTROL.phones[:-1]),
                    },
                },
                contract,
            )

    def test_ordered_overlay_target_mapping_rejects_timeline_shape_and_unexpected_state(
        self,
    ):
        contract = compact_phone_contract()
        provenance = {
            "states_per_phone": 2,
            "state_count": 22,
            "wake_phrase": {
                "phrase_id": "kizz-control",
                "phones": list(KIZZ_CONTROL.phones),
            },
            "target_frame_times_seconds": (0.015 + 0.030 * np.arange(87)).tolist(),
        }
        values = np.zeros((1, 87), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "timeline"):
            map_ordered_targets(
                values, {**provenance, "target_frame_times_seconds": [0.0]}, contract
            )
        values[0, 0] = 22
        with self.assertRaisesRegex(ValueError, "unexpected"):
            map_ordered_targets(values, provenance, contract)

    def test_checkpoint_binding_records_exact_best_and_last_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best.weights.h5").write_bytes(b"best")
            (root / "last.weights.h5").write_bytes(b"last")
            binding = checkpoint_binding(root, 25, (0.9, 1.2))
            self.assertEqual(binding["selected_checkpoint"], "best")
            self.assertEqual(binding["best_step"], 25)
            self.assertEqual(
                binding["weights_sha256"], sha256_file(root / "best.weights.h5")
            )
            self.assertEqual(
                binding["best_weights"]["sha256"], sha256_file(root / "best.weights.h5")
            )
            self.assertEqual(
                binding["last_weights"]["sha256"], sha256_file(root / "last.weights.h5")
            )
            (root / "best.weights.h5").write_bytes(b"changed")
            self.assertNotEqual(
                binding["best_weights"]["sha256"], sha256_file(root / "best.weights.h5")
            )

    def test_checkpoint_binding_requires_both_emitted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "both best and last"):
                checkpoint_binding(Path(directory), 1, (0.9, 1.0))

    def test_realized_schedule_is_exact_and_deterministic(self):
        first = expected_schedule_counts(7, 16)
        second = expected_schedule_counts(7, 16)
        self.assertEqual(first, second)
        self.assertEqual(sum(first[0].values()), 7 * 8)
        self.assertEqual(sum(first[1].values()), 7 * 8)
        self.assertEqual(
            set(first[0]), {"assemblyai", "deepgram", "elevenlabs", "kokoro"}
        )
        self.assertEqual(
            set(first[1]),
            {
                "public_speech",
                "kizz_control_phonetic_collision",
                "device_collision",
                "no_speech",
            },
        )
        variant_counts = first[2]
        self.assertEqual(len(variant_counts), 12)
        self.assertLessEqual(
            max(variant_counts.values()) - min(variant_counts.values()), 1
        )
        self.assertEqual(sum(variant_counts.values()), 7 * 8)
        provider_counts = first[0]
        self.assertLessEqual(
            max(provider_counts.values()) - min(provider_counts.values()), 1
        )

    def test_directory_provenance_hash_changes_when_member_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b").mkdir()
            (root / "a.txt").write_text("a")
            (root / "b" / "c.txt").write_text("c")
            before = provenance_ref(root)
            self.assertEqual(before["path"], str(root.resolve()))
            (root / "b" / "c.txt").write_text("changed")
            self.assertNotEqual(before["sha256"], provenance_ref(root)["sha256"])

    def test_cache_must_bind_active_qualification_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = root / "qualification.json"
            qualification.write_text('{"qualified": true}\n')
            cache = {
                "manifest_sha256": "m" * 64,
                "provenance": {
                    "teacher_qualification": {"sha256": sha256_file(qualification)}
                },
            }
            require_cache_binding(cache, qualification, "m" * 64)
            cache["provenance"]["teacher_qualification"]["sha256"] = "x" * 64
            with self.assertRaisesRegex(ValueError, "active qualification"):
                require_cache_binding(cache, qualification, "m" * 64)

    def test_cache_accepts_monotonic_threshold_rebinding_for_same_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("{}")
            source_hash = sha256_file(source)
            qualification = root / "qualified.json"
            qualification.write_text(
                json.dumps(
                    {
                        "model": {"revision": "r", "weights_sha256": "w"},
                        "operating_point_rebinding": {
                            "source_teacher_qualification": {"sha256": source_hash}
                        },
                    }
                )
            )
            cache = {
                "manifest_sha256": "m" * 64,
                "model": {"revision": "r", "weights_sha256": "w"},
                "provenance": {"teacher_qualification": {"sha256": source_hash}},
            }
            require_cache_binding(cache, qualification, "m" * 64)
            cache["model"]["weights_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "different weights_sha256"):
                require_cache_binding(cache, qualification, "m" * 64)

    def test_student_checkpoint_scoring_uses_portable_viterbi_reference(self):
        contract = compact_phone_contract()
        rng = np.random.default_rng(231)
        logits = rng.normal(size=(1, 66, len(contract["tokens"]))).astype(np.float32)
        read_only_logits = logits.copy()
        read_only_logits.setflags(write=False)

        class FakeModel:
            def __call__(self, features, training=False):
                self.seen_shape = np.asarray(features).shape
                return read_only_logits

        score = _student_scores(
            FakeModel(), np.zeros((1, 260, 40), dtype=np.float32), contract, 1
        )[0]
        decoder = student_decoder_contract(contract)
        expected = exhaustive_suffix_score(
            logits[0],
            contract,
            window_lengths=decoder["window_lengths_frames"],
            beta=decoder["beta"],
        ).canonical_fit
        self.assertAlmostEqual(score, expected, places=6)
        self.assertEqual(decoder["window_lengths_frames"], list(WINDOW_LENGTHS_FRAMES))
        self.assertEqual(decoder["window_lengths_frames"], [19, 23, 27, 32, 39, 47, 54])
        self.assertEqual(WINDOW_LENGTHS_SECONDS[-1], 1.60)

    def test_student_decoder_contract_hash_is_stable_and_explicit(self):
        contract = compact_phone_contract()
        decoder = student_decoder_contract(contract)
        self.assertEqual(decoder["algorithm"], "max_add_ctc_viterbi")
        self.assertEqual(
            decoder["implementation"],
            "microwakeword.kizz_viterbi_decoder.exhaustive_suffix_score",
        )
        self.assertEqual(
            student_decoder_contract_hash(contract),
            student_decoder_contract_hash(dict(contract)),
        )

    def test_device_channel_positives_are_partitioned_from_clean(self):
        rows = [
            {
                "split": "train",
                "label": 1,
                "provider": provider,
                "source_group": source_group,
            }
            for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro")
            for source_group in ("synthetic_clean", "device_channel_positive")
        ]
        clean = positive_indices_by_provider(rows, "clean")
        device = positive_indices_by_provider(rows, "device")
        for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro"):
            self.assertEqual(len(clean[provider]), 1)
            self.assertEqual(len(device[provider]), 1)
            self.assertNotEqual(clean[provider][0], device[provider][0])

    def test_overlay_parents_must_exactly_match_active_clean_train_inventory(self):
        corpus = [
            {
                "split": "train",
                "label": 1,
                "source_group": "clean",
                "provider": "assemblyai",
                "source_audio_sha256": "a" * 64,
            },
            {
                "split": "train",
                "label": 1,
                "source_group": "device_channel_positive",
                "provider": "assemblyai",
                "source_audio_sha256": "d" * 64,
            },
        ]
        provenance = {
            "examples": [
                {
                    "split": "train",
                    "variant": variant,
                    "provider": "assemblyai",
                    "source_audio_sha256": "a" * 64,
                }
                for variant in ("overlay-0", "overlay-1", "overlay-2", "overlay-3")
            ]
        }
        binding = require_overlay_parent_binding(corpus, provenance)
        self.assertEqual(binding["parents"], 1)
        bad = json.loads(json.dumps(provenance))
        bad["examples"][0]["source_audio_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "differ"):
            require_overlay_parent_binding(corpus, bad)

    def test_metadata_architecture_contract_is_explicit_and_complete(self):
        contract = compact_phone_contract()
        architecture = student_architecture_contract(contract)
        self.assertEqual(architecture["input_shape"], [260, 40])
        self.assertEqual(architecture["output_frames"], 66)
        self.assertEqual(architecture["output_count"], len(contract["tokens"]))
        self.assertEqual(architecture["stride"], 3)

    def test_temporal_residual_architecture_is_distinct_and_shape_compatible(self):
        from microwakeword.ordered_state_model import model as build_student

        contract = compact_phone_contract()
        control = student_architecture_contract(contract)
        residual = student_architecture_contract(contract, "temporal_residual")
        self.assertNotEqual(residual, control)
        self.assertEqual(residual["architecture_id"], "temporal_residual")
        self.assertEqual(residual["residual_connection"], [1, 1, 1, 1, 1])
        model = build_student(
            student_flags_for_architecture(
                "temporal_residual", len(contract["tokens"])
            ),
            (260, 40),
            None,
        )
        self.assertEqual(model.output_shape, (None, 66, len(contract["tokens"])))
        self.assertIsNotNone(model.get_layer("encoder_hidden"))

    def test_dilated_temporal_memory_preserves_output_and_extends_context(self):
        from microwakeword.ordered_state_model import model as build_student
        from microwakeword.ordered_state_model import receptive_field_ms

        contract = compact_phone_contract()
        flags = student_flags_for_architecture(
            "dilated_temporal_memory", len(contract["tokens"])
        )
        architecture = student_architecture_contract(
            contract, "dilated_temporal_memory"
        )
        model = build_student(flags, (260, 40), None)
        self.assertEqual(model.output_shape, (None, 66, len(contract["tokens"])))
        self.assertEqual(architecture["temporal_dilations"], [1, 2, 4, 8, 16])
        self.assertEqual(architecture["warmup_output_drop"], 20)
        self.assertGreaterEqual(receptive_field_ms(flags), 1900)
        self.assertAlmostEqual(student_output_times_seconds(flags, 66)[0], 0.635)

    def test_wide_temporal_memory_is_shape_compatible_and_larger(self):
        from microwakeword.ordered_state_model import model as build_student

        contract = compact_phone_contract()
        narrow = build_student(
            student_flags_for_architecture(
                "dilated_temporal_memory", len(contract["tokens"])
            ),
            (260, 40),
            None,
        )
        wide = build_student(
            student_flags_for_architecture(
                "dilated_temporal_memory_wide", len(contract["tokens"])
            ),
            (260, 40),
            None,
        )
        self.assertEqual(wide.output_shape, narrow.output_shape)
        self.assertGreater(wide.count_params(), narrow.count_params())

    def test_temporal_representation_loss_and_cache_are_bounded(self):
        student = np.ones((2, 3, 4), dtype=np.float32)
        teacher = np.ones((2, 3, 4), dtype=np.float32)
        mask = np.asarray([1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(
            float(temporal_representation_loss(student, teacher, mask)), 0.0
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "cache"
            matrix = np.zeros((2, 3, 4), dtype=np.float16)
            metadata = {
                "schema_version": 1,
                "representation": "qualified_teacher_last_hidden_frame_aligned_train_pca",
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
            }
            digest = hashlib.sha256()
            digest.update(matrix.tobytes(order="C"))
            digest.update(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            )
            metadata["cache_sha256"] = digest.hexdigest()
            np.save(prefix.with_suffix(".npy"), matrix)
            prefix.with_suffix(".json").write_text(json.dumps(metadata))
            loaded_metadata, loaded = load_temporal_representation_cache(prefix)
            self.assertEqual(loaded_metadata["cache_sha256"], digest.hexdigest())
            self.assertEqual(loaded.shape, matrix.shape)


if __name__ == "__main__":
    unittest.main()
