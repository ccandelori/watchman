from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from aegis_introspection.cift_model_training import CiftTrainingArtifact, CiftTrainingArtifactMetadata
from aegis_introspection.cift_probe_competition import promotion_paper_method_from_probe_competition
from aegis_introspection.cift_probe_head_to_head import (
    CiftProbeHeadToHeadConfig,
    CiftProbeHeadToHeadError,
    cift_probe_head_to_head_report_to_json,
    evaluate_cift_probe_head_to_head,
    write_cift_probe_head_to_head_json,
)
from introspection.scripts.compare_cift_probe_head_to_head import _head_to_head_config, _parse_args


class CiftProbeHeadToHeadTest(unittest.TestCase):
    def test_head_to_head_report_uses_grouped_repeated_seeds_and_promotes_linear_when_it_matches_paper(self) -> None:
        report = evaluate_cift_probe_head_to_head(_config(random_seeds=(11, 17, 23)))
        method = promotion_paper_method_from_probe_competition(report.competition_report)

        self.assertEqual("cift_probe_competition/v1", report.competition_report.schema_version)
        self.assertEqual("synthetic-linear-vs-paper-mlp", report.competition_report.report_id)
        self.assertEqual((11, 17, 23), report.competition_report.random_seeds)
        self.assertEqual("linear_logistic_regression", method.probe_architecture)
        self.assertEqual("regularized_logistic_loss", method.training_loss)
        self.assertGreaterEqual(
            report.competition_report.candidate_probe_metric_value,
            report.competition_report.paper_probe_metric_value,
        )
        self.assertGreaterEqual(
            report.competition_report.candidate_probe.metric_confidence_interval_low,
            report.competition_report.paper_probe.metric_value,
        )
        self.assertEqual(3, len(report.seed_results))
        self.assertTrue(all(seed_result.fold_count == 3 for seed_result in report.seed_results))

    def test_write_head_to_head_json_includes_seed_results_without_breaking_competition_schema(self) -> None:
        report = evaluate_cift_probe_head_to_head(_config(random_seeds=(11, 17, 23)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head_to_head.json"
            write_cift_probe_head_to_head_json(path=path, report=report)
            decoded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual("cift_probe_competition/v1", decoded["schema_version"])
        self.assertEqual("synthetic-linear-vs-paper-mlp", decoded["report_id"])
        self.assertEqual(3, len(decoded["seed_results"]))
        self.assertEqual("mlp_128_64_1", decoded["seed_results"][0]["paper_probe_architecture"])
        self.assertEqual("linear_logistic_regression", decoded["seed_results"][0]["candidate_probe_architecture"])

    def test_head_to_head_supports_fold_local_diagonal_mahalanobis_cci_features(self) -> None:
        report = evaluate_cift_probe_head_to_head(
            replace(
                _config(random_seeds=(11, 17, 23)),
                feature_representation="diagonal_mahalanobis_cci",
                activation_feature_key="",
                source_feature_keys=("selected_choice_window_layer_01", "selected_choice_window_layer_02"),
                calibration_source_labels=("secret_present_safe",),
                ridge=0.001,
            )
        )
        decoded = cift_probe_head_to_head_report_to_json(report)

        self.assertEqual("diagonal_mahalanobis_cci", decoded["feature_representation"])
        self.assertEqual(
            ["selected_choice_window_layer_01", "selected_choice_window_layer_02"],
            decoded["source_feature_keys"],
        )
        self.assertEqual(["secret_present_safe"], decoded["calibration_source_labels"])
        self.assertGreaterEqual(report.competition_report.paper_probe_metric_value, 0.99)
        self.assertGreaterEqual(report.competition_report.candidate_probe_metric_value, 0.99)

    def test_head_to_head_requires_at_least_three_repeated_seeds(self) -> None:
        with self.assertRaisesRegex(CiftProbeHeadToHeadError, "random_seeds"):
            evaluate_cift_probe_head_to_head(_config(random_seeds=(11, 17)))

    def test_head_to_head_cli_parses_repeated_seeds_and_model_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "artifact.pt"
            output_path = Path(directory) / "report.json"
            cli_config = _parse_args(
                (
                    "--artifact",
                    str(artifact_path),
                    "--output-json",
                    str(output_path),
                    "--report-id",
                    "watchman-linear-vs-paper-mlp",
                    "--training-dataset-id",
                    "watchman_semantic_v3_480_secret_present_binary",
                    "--task",
                    "safe_secret_vs_exfiltration",
                    "--positive-label",
                    "exfiltration_intent",
                    "--feature-representation",
                    "diagonal_mahalanobis_cci",
                    "--activation-feature",
                    "selected_choice_window_layer_19",
                    "--source-feature",
                    "selected_choice_window_layer_19",
                    "--source-feature",
                    "selected_choice_window_layer_20",
                    "--calibration-source-label",
                    "secret_present_safe",
                    "--ridge",
                    "0.001",
                    "--fold-count",
                    "4",
                    "--random-seeds",
                    "11,17,23",
                    "--decision-threshold",
                    "0.5",
                    "--linear-max-epochs",
                    "1000",
                    "--linear-regularization-c",
                    "1.0",
                    "--paper-mlp-max-epochs",
                    "220",
                    "--paper-mlp-learning-rate",
                    "0.05",
                    "--paper-mlp-l1-softplus-weight",
                    "0.0001",
                    "--paper-mlp-batch-size",
                    "32",
                    "--paper-hyperparameter-search-trials",
                    "1",
                    "--candidate-hyperparameter-search-trials",
                    "1",
                    "--evaluation-split-id",
                    "watchman/grouped-cv",
                    "--evaluation-split-manifest-id",
                    "watchman/grouped-cv/manifest",
                    "--metric-name",
                    "grouped_cv_macro_f1",
                    "--created-at",
                    "2026-06-24T00:00:00Z",
                )
            )
            config = _head_to_head_config(cli_config=cli_config, artifact=_artifact())

        self.assertEqual(artifact_path, cli_config.artifact_path)
        self.assertEqual(output_path, cli_config.output_json_path)
        self.assertEqual((11, 17, 23), cli_config.random_seeds)
        self.assertEqual("diagonal_mahalanobis_cci", cli_config.feature_representation)
        self.assertEqual(
            ("selected_choice_window_layer_19", "selected_choice_window_layer_20"),
            cli_config.source_feature_keys,
        )
        self.assertEqual(("secret_present_safe",), cli_config.calibration_source_labels)
        self.assertEqual("watchman-linear-vs-paper-mlp", config.report_id)
        self.assertEqual("selected_choice_window_layer_19", config.activation_feature_key)
        self.assertEqual("diagonal_mahalanobis_cci", config.feature_representation)
        self.assertEqual(220, config.paper_mlp_max_epochs)


def _config(random_seeds: tuple[int, ...]) -> CiftProbeHeadToHeadConfig:
    return CiftProbeHeadToHeadConfig(
        report_id="synthetic-linear-vs-paper-mlp",
        artifact=_artifact(),
        training_dataset_id="synthetic-cift-lab",
        task_name="safe_secret_vs_exfiltration",
        positive_label="exfiltration_intent",
        feature_representation="raw_activation",
        activation_feature_key="selected_choice_window_layer_01",
        source_feature_keys=("selected_choice_window_layer_01",),
        calibration_source_labels=("secret_present_safe",),
        ridge=0.001,
        fold_count=3,
        random_seeds=random_seeds,
        decision_threshold=0.5,
        linear_max_epochs=250,
        linear_regularization_c=1.0,
        paper_mlp_max_epochs=400,
        paper_mlp_learning_rate=0.05,
        paper_mlp_l1_softplus_weight=0.0,
        paper_mlp_batch_size=4,
        paper_hyperparameter_search_trials=1,
        candidate_hyperparameter_search_trials=1,
        evaluation_split_id="synthetic-cift-lab/grouped-cv",
        evaluation_split_manifest_id="synthetic-cift-lab/grouped-cv/manifest",
        metric_name="grouped_cv_macro_f1",
        created_at="2026-06-24T00:00:00Z",
    )


def _artifact() -> CiftTrainingArtifact:
    return CiftTrainingArtifact(
        metadata=CiftTrainingArtifactMetadata(
            model_id="Qwen/Qwen3-test",
            revision="main",
            selected_device="cpu",
            hidden_size=4096,
            layer_count=36,
            tokenizer_fingerprint_sha256="b" * 64,
            special_tokens_map_sha256="c" * 64,
            chat_template_sha256="d" * 64,
            layer_indices=(1, 2),
            pooling_methods=("selected_choice_window",),
        ),
        example_ids=tuple(f"safe-{index}" for index in range(6)) + tuple(f"exfil-{index}" for index in range(6)),
        labels=("secret_present_safe",) * 6 + ("exfiltration_intent",) * 6,
        families=(
            "safe-a",
            "safe-a",
            "safe-b",
            "safe-b",
            "safe-c",
            "safe-c",
            "exfil-a",
            "exfil-a",
            "exfil-b",
            "exfil-b",
            "exfil-c",
            "exfil-c",
        ),
        texts=tuple(f"text {index}" for index in range(12)),
        tags=(("test",),) * 12,
        features={
            "selected_choice_window_layer_01": np.asarray(
                [
                    [-2.0, -2.0],
                    [-2.1, -1.8],
                    [-1.8, -2.2],
                    [-1.9, -2.0],
                    [-2.3, -1.7],
                    [-2.0, -2.4],
                    [2.0, 2.0],
                    [2.1, 1.8],
                    [1.8, 2.2],
                    [1.9, 2.0],
                    [2.3, 1.7],
                    [2.0, 2.4],
                ],
                dtype=np.float32,
            ),
            "selected_choice_window_layer_02": np.asarray(
                [
                    [1.0, 1.0],
                    [1.1, 0.9],
                    [0.9, 1.2],
                    [1.2, 0.8],
                    [0.8, 1.1],
                    [1.0, 0.7],
                    [4.0, 4.0],
                    [4.1, 3.9],
                    [3.9, 4.2],
                    [4.2, 3.8],
                    [3.8, 4.1],
                    [4.0, 3.7],
                ],
                dtype=np.float32,
            ),
        },
    )


if __name__ == "__main__":
    unittest.main()
