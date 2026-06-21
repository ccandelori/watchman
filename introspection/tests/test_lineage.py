import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.lineage import (
    LineageError,
    load_lineage_manifest,
    sha256_file,
    validate_lineage_manifest,
)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


class LineageTest(unittest.TestCase):
    def test_sha256_file_hashes_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("aegis\n", encoding="utf-8")

            digest = sha256_file(path)

        self.assertEqual("c49ffda863700e6155e771ac6ba1100f58aefc25684fec5b4fe0d986630fa485", digest)

    def test_load_lineage_manifest_accepts_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_hash = _write_text(root / "data" / "prompts.jsonl", "{}\n")
            artifact_hash = _write_text(root / "data" / "activations" / "features.pt", "artifact\n")
            report_hash = _write_text(root / "data" / "reports" / "binary.json", "{}\n")
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [
                            {
                                "id": "baseline_prompts",
                                "path": "data/prompts.jsonl",
                                "sha256": dataset_hash,
                                "purpose": "Baseline prompt dataset.",
                                "label_counts": {
                                    "benign": 1,
                                    "secret_present_safe": 1,
                                    "exfiltration_intent": 1,
                                },
                                "family_count": 3,
                            }
                        ],
                        "artifacts": [
                            {
                                "id": "baseline_features",
                                "path": "data/activations/features.pt",
                                "sha256": artifact_hash,
                                "dataset_id": "baseline_prompts",
                                "model_id": "Qwen/Qwen3-0.6B",
                                "feature_count": 10,
                            }
                        ],
                        "reports": [
                            {
                                "id": "baseline_binary",
                                "path": "data/reports/binary.json",
                                "sha256": report_hash,
                                "dataset_id": "baseline_prompts",
                                "artifact_id": "baseline_features",
                                "evaluation_strategy": "stratified_kfold",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            validate_lineage_manifest(manifest, root)

        self.assertEqual("baseline_prompts", manifest.datasets[0].id)
        self.assertEqual("baseline_features", manifest.artifacts[0].id)
        self.assertEqual("baseline_binary", manifest.reports[0].id)

    def test_validate_lineage_manifest_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [
                            {
                                "id": "baseline_prompts",
                                "path": "data/missing.jsonl",
                                "sha256": "0" * 64,
                                "purpose": "Baseline prompt dataset.",
                                "label_counts": {
                                    "benign": 1,
                                    "secret_present_safe": 1,
                                    "exfiltration_intent": 1,
                                },
                                "family_count": 3,
                            }
                        ],
                        "artifacts": [],
                        "reports": [],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            with self.assertRaises(LineageError):
                validate_lineage_manifest(manifest, root)

    def test_validate_lineage_manifest_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_text(root / "data" / "prompts.jsonl", "{}\n")
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [
                            {
                                "id": "baseline_prompts",
                                "path": "data/prompts.jsonl",
                                "sha256": "0" * 64,
                                "purpose": "Baseline prompt dataset.",
                                "label_counts": {
                                    "benign": 1,
                                    "secret_present_safe": 1,
                                    "exfiltration_intent": 1,
                                },
                                "family_count": 3,
                            }
                        ],
                        "artifacts": [],
                        "reports": [],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            with self.assertRaises(LineageError):
                validate_lineage_manifest(manifest, root)

    def test_validate_lineage_manifest_rejects_unknown_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_hash = _write_text(root / "data" / "activations" / "features.pt", "artifact\n")
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [],
                        "artifacts": [
                            {
                                "id": "baseline_features",
                                "path": "data/activations/features.pt",
                                "sha256": artifact_hash,
                                "dataset_id": "missing_dataset",
                                "model_id": "Qwen/Qwen3-0.6B",
                                "feature_count": 10,
                            }
                        ],
                        "reports": [],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            with self.assertRaises(LineageError):
                validate_lineage_manifest(manifest, root)

    def test_validate_lineage_manifest_counts_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_hash = _write_text(root / "data" / "prompts.jsonl", "{}\n")
            artifact_hash = _write_text(root / "data" / "activations" / "features.pt", "artifact\n")
            report_hash = _write_text(root / "data" / "reports" / "binary.json", "{}\n")
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [
                            {
                                "id": "baseline_prompts",
                                "path": "data/prompts.jsonl",
                                "sha256": dataset_hash,
                                "purpose": "Baseline prompt dataset.",
                                "label_counts": {
                                    "benign": 1,
                                    "secret_present_safe": 1,
                                    "exfiltration_intent": 1,
                                },
                                "family_count": 3,
                            }
                        ],
                        "artifacts": [
                            {
                                "id": "baseline_features",
                                "path": "data/activations/features.pt",
                                "sha256": artifact_hash,
                                "dataset_id": "baseline_prompts",
                                "model_id": "Qwen/Qwen3-0.6B",
                                "feature_count": 10,
                            }
                        ],
                        "reports": [
                            {
                                "id": "baseline_binary",
                                "path": "data/reports/binary.json",
                                "sha256": report_hash,
                                "dataset_id": "baseline_prompts",
                                "artifact_id": "baseline_features",
                                "evaluation_strategy": "stratified_kfold",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            validate_lineage_manifest(manifest, root)

        self.assertEqual(1, len(manifest.datasets))
        self.assertEqual(1, len(manifest.artifacts))
        self.assertEqual(1, len(manifest.reports))


if __name__ == "__main__":
    unittest.main()
