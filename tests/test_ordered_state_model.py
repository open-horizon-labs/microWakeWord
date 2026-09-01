import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from microwakeword import ordered_state_model
from microwakeword import utils
from microwakeword.layers import modes
from microwakeword.model_train_eval import load_config
from microwakeword.ordered_state import OrderedStateDecoder
from microwakeword.ordered_state_tflite import tflite_output_logits
from microwakeword.phoneme_student import student_stream_phase_offset_frames
from microwakeword.train import configured_training_loss


class OrderedStateModelTest(unittest.TestCase):
    def flags(self):
        parser = argparse.ArgumentParser()
        ordered_state_model.model_parameters(parser)
        return parser.parse_args([])

    def test_default_model_emits_state_logits_at_each_timestep(self):
        flags = self.flags()
        model = ordered_state_model.model(flags, (128, 40), batch_size=2)

        self.assertEqual(model.output_shape[0], 2)
        self.assertGreater(model.output_shape[1], 1)
        self.assertEqual(model.output_shape[2], 23)
        self.assertLess(model.count_params(), 60000)
        self.assertEqual(ordered_state_model.receptive_field_ms(flags), 670)

    def test_training_length_preserves_the_declared_output_timeline(self):
        flags = self.flags()
        input_frames = ordered_state_model.training_spectrogram_length(flags, 66)
        model = ordered_state_model.model(flags, (input_frames, 40), batch_size=1)

        self.assertEqual(input_frames, 260)
        self.assertEqual(model.output_shape, (1, 66, 23))

    def test_config_loader_uses_ordered_state_output_geometry(self):
        flags = self.flags()
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "training.yaml"
            config_path.write_text(
                "train_dir: /tmp/ordered-state-test\n"
                "clip_duration_ms: 2000\n"
                "window_step_ms: 10\n",
                encoding="utf-8",
            )
            flags.training_config = str(config_path)
            config = load_config(flags, ordered_state_model)

        self.assertEqual(config["spectrogram_length_final_layer"], 66)
        self.assertEqual(config["spectrogram_length"], 260)
        self.assertEqual(config["training_input_shape"], (260, 40))

    def test_streaming_conversion_emits_one_state_vector(self):
        flags = self.flags()
        model = ordered_state_model.model(flags, (128, 40), batch_size=1)
        streaming = utils.to_streaming_inference(
            model,
            {"spectrogram_length": 128, "stride": flags.stride},
            modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )

        self.assertEqual(tuple(streaming.input.shape), (1, 3, 40))
        self.assertEqual(tuple(streaming.output.shape), (1, 1, 23))
        output = streaming(np.zeros((1, 3, 40), dtype=np.float32))
        self.assertEqual(tuple(output.shape), (1, 1, 23))

    def test_streaming_phase_reproduces_non_streaming_logits(self):
        flags = self.flags()
        features = (
            np.random.default_rng(231).normal(size=(1, 260, 40)).astype(np.float32)
        )
        non_streaming = ordered_state_model.model(flags, (260, 40), batch_size=1)
        expected = np.asarray(non_streaming(features, training=False))[0]
        streaming = utils.to_streaming_inference(
            non_streaming,
            {"spectrogram_length": 260, "stride": flags.stride},
            modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )
        phase = student_stream_phase_offset_frames(flags)
        primer = np.zeros((1, flags.stride, 40), dtype=np.float32)
        primer[:, -phase:] = features[:, :phase]
        emitted = [np.asarray(streaming(primer, training=False))[0, 0]]
        for offset in range(phase, 260 - flags.stride + 1, flags.stride):
            emitted.append(
                np.asarray(
                    streaming(
                        features[:, offset : offset + flags.stride], training=False
                    )
                )[0, 0]
            )
        actual = np.asarray(emitted)[-len(expected) :]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_dilated_temporal_memory_stream_matches_non_streaming_tail(self):
        from microwakeword.phoneme_student import compact_phone_contract
        from tools.distill_kizz_phoneme_student import student_flags_for_architecture

        flags = student_flags_for_architecture(
            "dilated_temporal_memory",
            len(compact_phone_contract()["tokens"]),
        )
        features = (
            np.random.default_rng(238).normal(size=(1, 260, 40)).astype(np.float32)
        )
        non_streaming = ordered_state_model.model(flags, (260, 40), batch_size=1)
        expected = np.asarray(non_streaming(features, training=False))[0]
        streaming = utils.to_streaming_inference(
            non_streaming,
            {"spectrogram_length": 260, "stride": flags.stride},
            modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )
        emitted = []
        for offset in range(0, 258, flags.stride):
            emitted.extend(
                np.asarray(
                    streaming(
                        features[:, offset : offset + flags.stride], training=False
                    )
                )[0]
            )
        actual = np.asarray(emitted)[-len(expected) :]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_training_wrapper_runs_the_sequence_endpoint_loss(self):
        flags = self.flags()
        acoustic = ordered_state_model.model(flags, (128, 40), batch_size=2)
        wrapped = ordered_state_model.training_model(
            acoustic, {"name": "ordered_state_sequence"}
        )
        wrapped.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=configured_training_loss(
                {"training_loss": {"name": "ordered_state_sequence"}}
            ),
        )
        features = (
            np.random.default_rng(231).normal(size=(2, 128, 40)).astype(np.float32)
        )
        labels = np.asarray([[1.0], [0.0]], dtype=np.float32)
        endpoint_loss = configured_training_loss(
            {"training_loss": {"name": "ordered_state_sequence"}}
        )
        with tf.GradientTape() as tape:
            predictions = wrapped(features, training=True)
            untrained_loss = endpoint_loss(labels, predictions)
        gradients = tape.gradient(untrained_loss, wrapped.trainable_variables)
        gradient_l1 = sum(
            float(tf.reduce_sum(tf.abs(gradient)).numpy())
            for gradient in gradients
            if gradient is not None
        )
        self.assertTrue(np.isfinite(float(untrained_loss.numpy())))
        self.assertGreater(gradient_l1, 0.0)
        loss = wrapped.train_on_batch(features, labels)
        self.assertTrue(np.isfinite(float(loss)))
        extracted = ordered_state_model.acoustic_model(wrapped)
        self.assertEqual(extracted.output_shape[-1], 23)
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "ordered.weights.h5"
            wrapped.save_weights(weights)
            reloaded = ordered_state_model.training_model(
                ordered_state_model.model(flags, (128, 40), batch_size=1),
                {"name": "ordered_state_sequence"},
            )
            reloaded.load_weights(weights)
            self.assertEqual(
                ordered_state_model.acoustic_model(reloaded).output_shape,
                (1, 22, 23),
            )

    def test_rejects_too_few_acoustic_states(self):
        flags = self.flags()
        flags.num_states = 2
        with self.assertRaisesRegex(
            ValueError, "needs background, silence, and a path"
        ):
            ordered_state_model.model(flags, (96, 40), batch_size=1)

    def test_decoder_contract_binds_training_score_settings(self):
        contract = ordered_state_model.decoder_contract(
            {
                "state_evidence_floor": -0.5,
                "self_loop_probability": 0.7,
                "next_state_probability": 0.3,
            },
            0.03,
        )
        self.assertEqual(contract["state_count"], 23)
        self.assertEqual(contract["frame_step_seconds"], 0.03)
        self.assertEqual(contract["decoder_args"]["state_evidence_floor"], -0.5)

    def test_quantized_streaming_output_keeps_all_states(self):
        flags = self.flags()
        model = ordered_state_model.model(flags, (96, 40), batch_size=1)
        config = {
            "spectrogram_length": 96,
            "stride": flags.stride,
            "train_dir": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            config["train_dir"] = temporary
            saved = Path(temporary) / "stream"
            converted = utils.convert_model_saved(
                model,
                config,
                folder="stream",
                mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
            )
            self.assertEqual(tuple(converted.output.shape), (1, 1, 23))

            class CalibrationData:
                def get_data(self, *_args, **_kwargs):
                    samples = np.zeros((4, 96, 40), dtype=np.float32)
                    samples[0, 0, 1] = 26.0
                    return samples, np.zeros(4), np.ones(4)

            output_dir = Path(temporary) / "tflite"
            utils.convert_saved_model_to_tflite(
                config,
                CalibrationData(),
                str(saved),
                str(output_dir),
                "ordered_state.tflite",
                quantize=True,
            )
            interpreter = tf.lite.Interpreter(
                model_path=str(output_dir / "ordered_state.tflite")
            )
            interpreter.allocate_tensors()
            self.assertEqual(
                tuple(interpreter.get_output_details()[0]["shape"]), (1, 1, 23)
            )
            self.assertEqual(
                interpreter.get_input_details()[0]["dtype"], np.dtype(np.int8)
            )
            self.assertEqual(
                interpreter.get_output_details()[0]["dtype"], np.dtype(np.uint8)
            )
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
            interpreter.set_tensor(
                input_detail["index"],
                np.zeros(input_detail["shape"], dtype=input_detail["dtype"]),
            )
            interpreter.invoke()
            logits = tflite_output_logits(
                interpreter.get_tensor(output_detail["index"]), output_detail
            )
            self.assertEqual(logits.shape, (23,))
            self.assertIsNone(
                OrderedStateDecoder(completion_margin=np.inf).step(logits)
            )


if __name__ == "__main__":
    unittest.main()
