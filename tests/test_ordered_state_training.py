import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from microwakeword import ordered_state_model
from microwakeword import train


class FakeFeatureHandler:
    def __init__(self, features):
        self.features = features

    def get_mode_label_counts(self, mode):
        return {0: 1, 1: 1}

    def set_training_class_weights(self, positive, negative):
        pass

    def get_data(self, mode, **_kwargs):
        return self.features, np.asarray([1.0, 0.0]), np.ones(2)

    def get_mode_size(self, mode):
        return 0 if mode.endswith("_ambient") else 2

    def sampling_ledger(self):
        return {"test": True}


class OrderedStateTrainingEntrypointTest(unittest.TestCase):
    def test_standard_trainer_runs_one_ordered_state_step(self):
        parser = argparse.ArgumentParser()
        ordered_state_model.model_parameters(parser)
        flags = parser.parse_args([])
        acoustic = ordered_state_model.model(flags, (128, 40), batch_size=2)
        model = ordered_state_model.training_model(
            acoustic, {"name": "ordered_state_sequence"}
        )
        features = (
            np.random.default_rng(240).normal(size=(2, 128, 40)).astype(np.float32)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aligned = root / "aligned"
            aligned.mkdir()
            targets = np.zeros((2, acoustic.output_shape[1]), dtype=np.int32)
            targets[0, :21] = np.arange(2, 23)
            np.save(aligned / "features.npy", features)
            np.save(aligned / "targets.npy", targets)
            config = {
                "train_dir": str(root),
                "summaries_dir": str(root / "logs"),
                "training_loss": {
                    "name": "ordered_state_sequence",
                    "frame_weight": 0.25,
                    "frame_supervision": {
                        "directory": str(aligned),
                        "batch_size": 2,
                        "seed": 240,
                    },
                },
                "training_steps": [1],
                "learning_rates": [1e-4],
                "batch_size": 2,
                "spectrogram_length": 128,
                "eval_step_interval": 1,
                "minimization_metric": "validation_false_positives",
                "maximization_metric": "recall_at_no_faph",
                "target_minimization": 0,
            }
            train.train(model, config, FakeFeatureHandler(features))
            self.assertTrue((root / "last_weights.weights.h5").is_file())
            self.assertTrue((root / "best_weights.weights.h5").is_file())

    def test_frame_supervision_rejects_invalid_state_targets(self):
        parser = argparse.ArgumentParser()
        ordered_state_model.model_parameters(parser)
        flags = parser.parse_args([])
        model = ordered_state_model.training_model(
            ordered_state_model.model(flags, (128, 40), batch_size=1)
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "features.npy", np.zeros((1, 128, 40), np.float32))
            np.save(root / "targets.npy", np.full((1, 22), 23, np.int32))
            from microwakeword.ordered_state_training import (
                OrderedStateFrameSupervisor,
            )

            with self.assertRaisesRegex(ValueError, "invalid state"):
                OrderedStateFrameSupervisor(
                    model,
                    model.optimizer,
                    {"directory": str(root), "frame_weight": 0.25},
                )


if __name__ == "__main__":
    unittest.main()
