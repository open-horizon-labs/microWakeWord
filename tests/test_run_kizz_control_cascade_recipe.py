import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.run_kizz_control_cascade_recipe import (
    RecipeError,
    execute,
    load_recipe,
    load_state,
    parse_overrides,
    preflight,
    resolve_variables,
    selected_stage_ids,
    stage_is_current,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/kizz/control-cascade-v9.yaml"


class KizzCascadeRecipeTests(unittest.TestCase):
    def test_checked_in_recipe_is_portable_and_preserves_one_way_gates(self):
        recipe = load_recipe(RECIPE)
        serialized = RECIPE.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("/private/tmp", serialized)
        self.assertEqual(recipe.raw["phrase"], "Kizz Control")
        self.assertEqual(
            recipe.raw["policy"]["positive_providers"],
            ["assemblyai", "deepgram", "elevenlabs", "kokoro"],
        )
        self.assertEqual(
            recipe.raw["policy"]["excluded_positive_provider"], "macos-say"
        )
        self.assertEqual(
            recipe.raw["public_inputs"]["musan"]["md5"],
            "0c472d4fc0c5141eca47ad1ffeb2a7df",
        )
        self.assertEqual(
            recipe.raw["public_inputs"]["librispeech_train_clean_360"]["md5"],
            "c0e676e450a7ff2f54aeade5171606fa",
        )
        freeze_musan = recipe.by_id["freeze_negative_assets"]
        self.assertIn("--musan-archive-md5", freeze_musan.command)
        freeze_librispeech = recipe.by_id["freeze_final_continuous_lock"]
        self.assertIn("--source-archive-md5", freeze_librispeech.command)
        threshold = recipe.by_id["freeze_device_validation_threshold"]
        self.assertEqual(threshold.selection_role, "threshold")
        self.assertEqual(threshold.reads_splits, {"validation"})
        test = recipe.by_id["score_fresh_device_test"]
        self.assertIn(threshold.id, test.depends_on)
        self.assertEqual(test.reads_splits, {"test"})
        locked = [
            stage
            for stage in recipe.stages
            if stage.id.startswith("evaluate_continuous_shard_")
        ]
        self.assertEqual(len(locked), 8)
        self.assertTrue(all("locked_test" in stage.reads_splits for stage in locked))
        package = recipe.by_id["package_firmware_handoff"]
        self.assertIn("--accept-observed-operating-point", package.command)
        for stage in recipe.stages:
            self.assertTrue(
                set(stage.outputs) <= set(stage.evidence),
                f"{stage.id} has unhashed outputs",
            )

    def test_reference_models_match_declared_hashes(self):
        payload = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
        reference = ROOT / "recipes/kizz/reference-cascade-v9"
        bindings = {
            "detector": reference / "kizz_control_detector.tflite",
            "compact_verifier": reference
            / "kizz_control_compact_verifier_int8_v9.tflite",
            "ordered_verifier": reference
            / "kizz_control_ordered_verifier_int8.tflite",
        }
        for role, path in bindings.items():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, payload["reference_result"][role]["sha256"])

    def test_schema_rejects_test_access_during_threshold_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "variables": {"workspace": {"default": "work"}},
                        "stages": [
                            {
                                "id": "bad",
                                "command": ["python3", "-V"],
                                "selection_role": "threshold",
                                "reads_splits": ["validation", "test"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RecipeError, "selection cannot read test"):
                load_recipe(path)

    def test_preflight_requires_explicit_paid_authority(self):
        recipe = load_recipe(RECIPE)
        variables = resolve_variables(
            recipe,
            {"workspace": "/tmp/kizz-recipe-test-does-not-run"},
            environment={},
        )
        selected = selected_stage_ids(recipe, ["synthesize_assemblyai"])
        errors = preflight(
            recipe,
            variables,
            selected,
            allow_paid=False,
            allow_hardware=False,
        )
        self.assertTrue(any("--allow-paid" in error for error in errors))

    def test_runner_resumes_only_while_evidence_hashes_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "work"
            recipe_path = root / "recipes" / "kizz" / "fixture.yaml"
            recipe_path.parent.mkdir(parents=True)
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "variables": {
                            "workspace": {"default": str(workspace)},
                            "python": {"default": "python3"},
                        },
                        "stages": [
                            {
                                "id": "write",
                                "command": [
                                    "{python}",
                                    "-c",
                                    "from pathlib import Path; "
                                    "p=Path(r'{workspace}/result.txt'); "
                                    "p.parent.mkdir(parents=True, exist_ok=True); "
                                    "p.write_text('bound')",
                                ],
                                "outputs": ["{workspace}/result.txt"],
                                "evidence": ["{workspace}/result.txt"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            recipe = load_recipe(recipe_path)
            variables = resolve_variables(recipe, {}, environment={})
            selected = selected_stage_ids(recipe, [])
            execute(
                recipe,
                variables,
                selected,
                jobs=1,
                dry_run=False,
                force=False,
            )
            stage = recipe.by_id["write"]
            state = load_state(workspace / "recipe-state.json")
            self.assertTrue(stage_is_current(stage, variables, recipe, state))
            (workspace / "result.txt").write_text("drift", encoding="utf-8")
            self.assertFalse(stage_is_current(stage, variables, recipe, state))

    def test_override_parser_rejects_ambiguous_values(self):
        with self.assertRaisesRegex(RecipeError, "NAME=VALUE"):
            parse_overrides(["workspace"])


if __name__ == "__main__":
    unittest.main()
