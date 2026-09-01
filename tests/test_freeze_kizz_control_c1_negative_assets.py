import unittest

from tools.freeze_kizz_control_c1_negative_assets import partition_rows


class FreezeKizzControlC1NegativeAssetsTests(unittest.TestCase):
    def rows(self):
        rows = []
        # 109.5 hours total. Music has two files per artist group so the test
        # can prove grouping, not merely hash disjointness.
        for family, count, hours in (
            ("speech", 61, 1.0),
            ("music", 86, 0.5),
            ("noise", 22, 0.25),
        ):
            for index in range(count):
                group_index = index // 2 if family == "music" else index
                rows.append(
                    {
                        "family": family,
                        "source_group_id": f"{family}-group-{group_index}",
                        "source_id": f"{family}-{index}",
                        "audio_sha256": f"{family}-{index:04d}",
                        "duration_seconds": hours * 3600,
                    }
                )
        return rows

    def test_freezes_over_100_hours_and_keeps_groups_disjoint(self):
        train, holdout = partition_rows(self.rows(), seed=7)
        self.assertGreaterEqual(
            sum(row["duration_seconds"] for row in holdout) / 3600, 100.0
        )
        self.assertEqual(
            {row["family"] for row in train}, {"speech", "music", "noise"}
        )
        self.assertEqual(
            {row["family"] for row in holdout}, {"speech", "music", "noise"}
        )
        self.assertFalse(
            {row["source_group_id"] for row in train}
            & {row["source_group_id"] for row in holdout}
        )
        self.assertFalse(
            {row["audio_sha256"] for row in train}
            & {row["audio_sha256"] for row in holdout}
        )

    def test_partition_preserves_required_identity_metadata(self):
        rows = self.rows()
        for row in rows:
            row.update(
                {
                    "speaker_id": row["source_group_id"],
                    "session_id": row["source_group_id"],
                    "semantic_label": f"{row['family']}_negative",
                }
            )
        train, holdout = partition_rows(rows, seed=9)
        self.assertTrue(
            all(
                row["speaker_id"]
                and row["session_id"]
                and row["semantic_label"].endswith("_negative")
                for row in train + holdout
            )
        )

    def test_rejects_a_sub_100_hour_lock(self):
        with self.assertRaisesRegex(ValueError, "below 100"):
            partition_rows(self.rows(), minimum_holdout_hours=99.0)


if __name__ == "__main__":
    unittest.main()
