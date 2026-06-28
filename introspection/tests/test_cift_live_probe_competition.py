from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aegis_introspection.cift_live_probe_competition import (
    CiftLiveProbeCompetitionConfig,
    CiftLiveProbeRun,
    cift_live_probe_competition_report_from_mapping,
    cift_live_probe_competition_report_to_json,
    compare_cift_live_probe_candidates,
    live_promotion_paper_method_from_probe_competition,
    materialize_cift_live_probe_competition,
)


class CiftLiveProbeCompetitionTest(unittest.TestCase):
    def test_live_sealed_competition_records_strict_candidate_win(self) -> None:
        report = compare_cift_live_probe_candidates(_config())

        self.assertEqual("aegis_introspection.cift_live_probe_competition/v1", report.schema_version)
        self.assertEqual(0.9979166576243821, report.paper_probe_metric_value)
        self.assertEqual(1.0, report.candidate_probe_metric_value)
        self.assertGreater(report.candidate_delta, 0.0)
        self.assertTrue(report.candidate_strictly_outperforms_paper)
        self.assertEqual("linear_logistic_regression", report.winner_probe_architecture)

        decoded = cift_live_probe_competition_report_from_mapping(cift_live_probe_competition_report_to_json(report))
        method = live_promotion_paper_method_from_probe_competition(decoded)

        self.assertEqual("raw_activation", method.feature_representation)
        self.assertEqual("linear_logistic_regression", method.probe_architecture)
        self.assertEqual(report.report_id, method.head_to_head_report_id)
        self.assertEqual(report.paper_probe_metric_value, method.paper_probe_metric_value)
        self.assertEqual(report.candidate_probe_metric_value, method.candidate_probe_metric_value)
        self.assertIn("live sealed", method.paper_faithfulness_exception or "")

    def test_live_sealed_competition_promotes_paper_mlp_when_candidate_does_not_win(self) -> None:
        report = compare_cift_live_probe_candidates(
            CiftLiveProbeCompetitionConfig(
                report_id="synthetic-live-head-to-head",
                training_dataset_id="synthetic-cift-lab",
                task_name="safe_secret_vs_exfiltration",
                evaluation_split_id="synthetic-cift-lab/sealed",
                evaluation_split_manifest_id="synthetic-manifest",
                evaluation_split_sha256="c" * 64,
                feature_representation="raw_activation",
                activation_feature_key="selected_choice_window_layer_21",
                metric_name="sealed_holdout_macro_f1",
                paper_probe=_probe_run(
                    source_report_id="paper",
                    probe_architecture="mlp_128_64_1",
                    training_loss="bce_with_l1_softplus_weight_sparsity",
                    model_bundle_id="paper-bundle",
                    metric_value=1.0,
                ),
                candidate_probe=_probe_run(
                    source_report_id="linear",
                    probe_architecture="linear_logistic_regression",
                    training_loss="regularized_logistic_loss",
                    model_bundle_id="linear-bundle",
                    metric_value=1.0,
                ),
                higher_is_better=True,
                created_at="2026-06-24T00:00:00Z",
            )
        )

        method = live_promotion_paper_method_from_probe_competition(report)

        self.assertEqual("mlp_128_64_1", method.probe_architecture)
        self.assertEqual("bce_with_l1_softplus_weight_sparsity", method.training_loss)
        self.assertIsNone(method.paper_faithfulness_exception)

    def test_live_sealed_competition_treats_freeform_readout_key_as_raw_activation(self) -> None:
        report = compare_cift_live_probe_candidates(
            replace(
                _config(),
                feature_representation="final_token_layer_12",
                activation_feature_key="final_token_layer_12",
            )
        )

        method = live_promotion_paper_method_from_probe_competition(report)

        self.assertEqual("raw_activation", method.feature_representation)
        self.assertEqual("not_applicable", method.covariance_estimator)
        self.assertEqual("not_applicable", method.layer_weighting)
        self.assertEqual(0.0, method.ridge)
        self.assertIn("final_token_layer_12", method.paper_faithfulness_exception or "")

    def test_materialize_live_competition_writes_self_identifying_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "live_head_to_head.json"

            report = materialize_cift_live_probe_competition(config=_config(), output_path=output_path)
            decoded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report.report_id, decoded["report_id"])
        self.assertEqual(report.schema_version, decoded["schema_version"])


def _config() -> CiftLiveProbeCompetitionConfig:
    return CiftLiveProbeCompetitionConfig(
        report_id="synthetic-live-head-to-head",
        training_dataset_id="synthetic-cift-lab",
        task_name="safe_secret_vs_exfiltration",
        evaluation_split_id="synthetic-cift-lab/sealed",
        evaluation_split_manifest_id="synthetic-manifest",
        evaluation_split_sha256="c" * 64,
        feature_representation="raw_activation",
        activation_feature_key="selected_choice_window_layer_21",
        metric_name="sealed_holdout_macro_f1",
        paper_probe=_probe_run(
            source_report_id="paper",
            probe_architecture="mlp_128_64_1",
            training_loss="bce_with_l1_softplus_weight_sparsity",
            model_bundle_id="paper-bundle",
            metric_value=0.9979166576243821,
        ),
        candidate_probe=_probe_run(
            source_report_id="linear",
            probe_architecture="linear_logistic_regression",
            training_loss="regularized_logistic_loss",
            model_bundle_id="linear-bundle",
            metric_value=1.0,
        ),
        higher_is_better=True,
        created_at="2026-06-24T00:00:00Z",
    )


def _probe_run(
    source_report_id: str,
    probe_architecture: str,
    training_loss: str,
    model_bundle_id: str,
    metric_value: float,
) -> CiftLiveProbeRun:
    return CiftLiveProbeRun(
        source_report_id=source_report_id,
        probe_architecture=probe_architecture,
        training_loss=training_loss,
        model_bundle_id=model_bundle_id,
        metric_value=metric_value,
        false_negative_count=0,
        false_positive_count=0,
        false_negative_rate=0.0,
        false_positive_rate=0.0,
        operating_threshold=0.5,
    )


if __name__ == "__main__":
    unittest.main()
