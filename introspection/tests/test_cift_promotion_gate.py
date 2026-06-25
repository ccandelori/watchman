from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from aegis_introspection.cift_model_bundle import CandidateStatus, CiftModelBundle, CiftModelBundleMetadata
from aegis_introspection.cift_paper_mlp import CiftPaperMlpClassifier, CiftPaperMlpConfig
from aegis_introspection.cift_probe_competition import (
    CiftProbeCompetitionConfig,
    CiftProbeRun,
    compare_cift_probe_candidates,
    promotion_paper_method_from_probe_competition,
)
from aegis_introspection.cift_promotion_gate import (
    CiftPaperMethodContract,
    CiftPromotionEvidence,
    CiftPromotionGateError,
    CiftPromotionReportArtifact,
    assert_cift_runtime_promotion_eligible,
    cift_paper_method_contract_to_json,
    cift_promotion_evidence_from_mapping,
    cift_promotion_evidence_to_json,
    cift_promotion_gate_result_to_json,
    evaluate_cift_promotion_gate,
)
from numpy.typing import NDArray

_REQUIRED_REPORT_IDS = (
    "synthetic-metric-report",
    "synthetic-sealed-holdout-report",
    "synthetic-calibration-report",
    "synthetic-ablation-report",
    "synthetic-patching-report",
    "synthetic-failure-case-report",
    "synthetic-runtime-prevention-report",
    "synthetic-gateway-smoke-report",
    "synthetic-lineage-report",
)


class _FakeScaler:
    mean_: list[float]
    scale_: list[float]

    def __init__(self) -> None:
        self.mean_ = [0.0, 0.0]
        self.scale_ = [1.0, 1.0]


class _FakeLogisticRegression:
    classes_: list[int]
    coef_: list[list[float]]
    intercept_: list[float]

    def __init__(self) -> None:
        self.classes_ = [0, 1]
        self.coef_ = [[2.0, 2.0]]
        self.intercept_ = [-3.0]


class _FakePipeline:
    classes_: list[int]
    named_steps: dict[str, object]

    def __init__(self) -> None:
        self.classes_ = [0, 1]
        self.named_steps = {
            "standardscaler": _FakeScaler(),
            "logisticregression": _FakeLogisticRegression(),
        }

    def predict_proba(self, matrix: NDArray[np.float32]) -> NDArray[np.float64]:
        probabilities = tuple((0.9, 0.1) if row[0] < 1.0 else (0.1, 0.9) for row in matrix)
        return np.asarray(probabilities, dtype=np.float64)


def _trained_classifier() -> _FakePipeline:
    return _FakePipeline()


def _metadata(candidate_status: CandidateStatus) -> CiftModelBundleMetadata:
    return CiftModelBundleMetadata(
        schema_version="cift_model_bundle/v1",
        source_model_id="Qwen/Qwen3-test",
        source_revision="main",
        source_selected_device="cpu",
        source_hidden_size=4096,
        source_layer_count=36,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        training_dataset_id="synthetic-cift-lab",
        source_artifact_path="data/activations/synthetic.pt",
        source_artifact_sha256="a" * 64,
        evaluation_report_ids=_REQUIRED_REPORT_IDS,
        task_name="safe_secret_vs_exfiltration",
        activation_feature_key="readout_window_layer_15",
        feature_count=2,
        label_names=("secret_present_safe", "exfiltration_intent"),
        positive_label="exfiltration_intent",
        decision_threshold=0.5,
        score_semantics="full_train_classifier_probability",
        created_at="2026-06-21T00:00:00Z",
        candidate_status=candidate_status,
    )


def _bundle(metadata: CiftModelBundleMetadata) -> CiftModelBundle:
    return CiftModelBundle(metadata=metadata, classifier=_trained_classifier(), calibrator=None)


def _mlp_bundle(metadata: CiftModelBundleMetadata) -> CiftModelBundle:
    classifier = CiftPaperMlpClassifier(
        CiftPaperMlpConfig(
            input_dim=2,
            hidden_layer_sizes=(128, 64),
            learning_rate=0.01,
            max_epochs=2,
            batch_size=4,
            l1_softplus_weight=0.0,
            random_seed=7,
        )
    )
    classifier.fit(
        np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [2.0, 1.5],
                [2.5, 2.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    return CiftModelBundle(metadata=metadata, classifier=classifier, calibrator=None)


def _report_artifacts(report_ids: tuple[str, ...]) -> tuple[CiftPromotionReportArtifact, ...]:
    return tuple(
        CiftPromotionReportArtifact(
            report_id=report_id,
            path=f"introspection/data/reports/{report_id}.json",
            sha256=f"{index + 1:064x}",
            schema_version="synthetic_report/v1",
        )
        for index, report_id in enumerate(report_ids)
    )


def _probe_run(
    source_report_id: str,
    probe_architecture: str,
    training_loss: str,
    metric_value: float,
    false_positive_rate: float,
    true_positive_rate: float,
) -> CiftProbeRun:
    return CiftProbeRun(
        source_report_id=source_report_id,
        probe_architecture=probe_architecture,
        training_loss=training_loss,
        training_dataset_id="synthetic-cift-lab",
        training_dataset_sha256="a" * 64,
        task_name="safe_secret_vs_exfiltration",
        evaluation_split_id="synthetic-cift-lab/sealed-holdout",
        evaluation_split_manifest_id="synthetic-cift-lab/sealed-holdout/manifest",
        evaluation_split_sha256="b" * 64,
        metric_name="sealed_holdout_macro_f1",
        metric_value=metric_value,
        metric_confidence_interval_low=metric_value - 0.01,
        metric_confidence_interval_high=metric_value + 0.01,
        random_seeds=(11, 17, 23),
        hyperparameter_search_trials=8,
        operating_threshold=0.5,
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
    )


def _promotion_evidence() -> CiftPromotionEvidence:
    return CiftPromotionEvidence(
        schema_version="cift_promotion_evidence/v1",
        evidence_id="synthetic-promotion-evidence",
        behavior_id="secret-exfiltration-intent",
        behavior_description="User request attempts to move a protected secret into an external channel.",
        training_dataset_id="synthetic-cift-lab",
        train_split_id="synthetic-cift-lab/train",
        calibration_split_id="synthetic-cift-lab/calibration",
        heldout_split_id="synthetic-cift-lab/heldout",
        sealed_holdout_split_id="synthetic-cift-lab/sealed-holdout",
        sealed_holdout_report_id="synthetic-sealed-holdout-report",
        metric_report_id="synthetic-metric-report",
        metric_name="sealed_holdout_macro_f1",
        metric_value=0.91,
        metric_threshold=0.9,
        calibration_report_id="synthetic-calibration-report",
        ablation_report_id="synthetic-ablation-report",
        ablation_delta=0.18,
        ablation_delta_threshold=0.1,
        patching_report_id="synthetic-patching-report",
        failure_case_report_id="synthetic-failure-case-report",
        runtime_prevention_report_id="synthetic-runtime-prevention-report",
        gateway_smoke_report_id="synthetic-gateway-smoke-report",
        lineage_report_id="synthetic-lineage-report",
        report_artifacts=_report_artifacts(_REQUIRED_REPORT_IDS),
        paper_method=_paper_method_evidence(),
        created_at="2026-06-23T00:00:00Z",
    )


def _paper_method_evidence() -> CiftPaperMethodContract:
    return CiftPaperMethodContract(
        readout_position_contract="post_secret_post_query_causal_readout",
        monitored_layer_policy="last_quarter_transformer_layers",
        feature_representation="diagonal_mahalanobis_cci",
        covariance_estimator="diagonal_covariance",
        ridge=0.001,
        layer_weighting="softplus_nonnegative_cfs",
        probe_architecture="mlp_128_64_1",
        training_loss="bce_with_l1_softplus_weight_sparsity",
        pre_output=True,
        uses_static_secret_token_positions=False,
        head_to_head_report_id=None,
        paper_probe_metric_value=None,
        candidate_probe_metric_value=None,
        paper_faithfulness_exception=None,
    )


def _paper_method_contract() -> dict[str, object]:
    return cift_paper_method_contract_to_json(_paper_method_evidence())


class CiftPromotionGateTest(unittest.TestCase):
    def test_runtime_candidate_with_complete_lab_evidence_passes_gate(self) -> None:
        bundle = _mlp_bundle(_metadata("runtime_candidate"))
        evidence = _promotion_evidence()

        decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)
        result = cift_promotion_gate_result_to_json(evidence=evidence, decision=decision)

        self.assertTrue(decision.eligible)
        self.assertEqual((), decision.failed_requirements)
        self.assertEqual((), decision.missing_report_ids)
        self.assertEqual("cift_promotion_gate_result/v1", result["schema_version"])
        self.assertEqual("synthetic-promotion-evidence", result["evidence_id"])
        self.assertEqual("runtime_candidate_promotion_only", result["eligibility_scope"])
        self.assertFalse(result["production_release_eligible"])
        self.assertTrue(result["requires_certification_binding"])
        self.assertEqual(list(_REQUIRED_REPORT_IDS), result["required_report_ids"])
        self.assertEqual(9, len(result["report_artifacts"]))

    def test_gate_rejects_linear_bundle_claiming_paper_mlp_method(self) -> None:
        bundle = _bundle(_metadata("runtime_candidate"))
        evidence = _promotion_evidence()

        decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertIn(
            "paper_method.probe_architecture mlp_128_64_1 requires CiftPaperMlpClassifier",
            decision.failed_requirements,
        )

    def test_gate_rejects_paper_mlp_bundle_claiming_challenger_method(self) -> None:
        metadata = replace(
            _metadata("runtime_candidate"),
            evaluation_report_ids=(*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report"),
        )
        evidence = replace(
            _promotion_evidence(),
            report_artifacts=_report_artifacts((*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report")),
            paper_method=replace(
                _paper_method_evidence(),
                probe_architecture="linear_logistic_regression",
                training_loss="regularized_logistic_loss",
                head_to_head_report_id="synthetic-linear-vs-mlp-report",
                paper_probe_metric_value=0.91,
                candidate_probe_metric_value=0.93,
            ),
        )

        decision = evaluate_cift_promotion_gate(bundle=_mlp_bundle(metadata), evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertIn(
            "CiftPaperMlpClassifier requires paper_method.probe_architecture mlp_128_64_1",
            decision.failed_requirements,
        )

    def test_gate_rejects_missing_report_artifact_manifest_entry(self) -> None:
        bundle = _bundle(_metadata("runtime_candidate"))
        evidence = replace(_promotion_evidence(), report_artifacts=_report_artifacts(_REQUIRED_REPORT_IDS[:-1]))

        decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertIn("report_artifacts must cover required_report_ids", decision.failed_requirements)

    def test_gate_rejects_alternative_probe_without_head_to_head_evidence(self) -> None:
        bundle = _bundle(_metadata("runtime_candidate"))
        record = cift_promotion_evidence_to_json(_promotion_evidence())
        record["paper_method"] = _paper_method_contract()
        paper_method = record["paper_method"]
        if not isinstance(paper_method, dict):
            raise AssertionError("paper_method must be an object.")
        paper_method["probe_architecture"] = "linear_logistic_regression"
        evidence = cift_promotion_evidence_from_mapping(record)

        decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertIn(
            "alternative probe architecture requires head_to_head_report_id",
            decision.failed_requirements,
        )

    def test_gate_allows_challenger_probe_with_head_to_head_win(self) -> None:
        metadata = replace(
            _metadata("runtime_candidate"),
            evaluation_report_ids=(*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report"),
        )
        evidence = replace(
            _promotion_evidence(),
            report_artifacts=_report_artifacts((*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report")),
            paper_method=replace(
                _paper_method_evidence(),
                probe_architecture="linear_logistic_regression",
                training_loss="regularized_logistic_loss",
                head_to_head_report_id="synthetic-linear-vs-mlp-report",
                paper_probe_metric_value=0.91,
                candidate_probe_metric_value=0.93,
            ),
        )

        decision = evaluate_cift_promotion_gate(bundle=_bundle(metadata), evidence=evidence)

        self.assertTrue(decision.eligible)
        self.assertEqual((), decision.failed_requirements)
        self.assertIn("synthetic-linear-vs-mlp-report", decision.required_report_ids)

    def test_gate_allows_raw_activation_challenger_only_as_explicit_head_to_head_exception(self) -> None:
        metadata = replace(
            _metadata("runtime_candidate"),
            evaluation_report_ids=(*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report"),
        )
        evidence = replace(
            _promotion_evidence(),
            report_artifacts=_report_artifacts((*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report")),
            paper_method=replace(
                _paper_method_evidence(),
                feature_representation="raw_activation",
                covariance_estimator="not_applicable",
                ridge=0.0,
                layer_weighting="not_applicable",
                probe_architecture="linear_logistic_regression",
                training_loss="regularized_logistic_loss",
                head_to_head_report_id="synthetic-linear-vs-mlp-report",
                paper_probe_metric_value=0.68,
                candidate_probe_metric_value=1.0,
                paper_faithfulness_exception=(
                    "Raw selected-choice activations outperformed the paper MLP in head-to-head grouped CV."
                ),
            ),
        )

        decision = evaluate_cift_promotion_gate(bundle=_bundle(metadata), evidence=evidence)

        self.assertTrue(decision.eligible)
        self.assertEqual((), decision.failed_requirements)

    def test_gate_rejects_raw_activation_challenger_without_faithfulness_exception(self) -> None:
        metadata = replace(
            _metadata("runtime_candidate"),
            evaluation_report_ids=(*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report"),
        )
        evidence = replace(
            _promotion_evidence(),
            report_artifacts=_report_artifacts((*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report")),
            paper_method=replace(
                _paper_method_evidence(),
                feature_representation="raw_activation",
                covariance_estimator="not_applicable",
                ridge=0.0,
                layer_weighting="not_applicable",
                probe_architecture="linear_logistic_regression",
                training_loss="regularized_logistic_loss",
                head_to_head_report_id="synthetic-linear-vs-mlp-report",
                paper_probe_metric_value=0.68,
                candidate_probe_metric_value=1.0,
                paper_faithfulness_exception=None,
            ),
        )

        decision = evaluate_cift_promotion_gate(bundle=_bundle(metadata), evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertIn(
            "paper_method.paper_faithfulness_exception must explain non-paper feature_representation",
            decision.failed_requirements,
        )

    def test_gate_rejects_raw_activation_challenger_that_only_ties_paper(self) -> None:
        metadata = replace(
            _metadata("runtime_candidate"),
            evaluation_report_ids=(*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report"),
        )
        evidence = replace(
            _promotion_evidence(),
            report_artifacts=_report_artifacts((*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report")),
            paper_method=replace(
                _paper_method_evidence(),
                feature_representation="raw_activation",
                covariance_estimator="not_applicable",
                ridge=0.0,
                layer_weighting="not_applicable",
                probe_architecture="linear_logistic_regression",
                training_loss="regularized_logistic_loss",
                head_to_head_report_id="synthetic-linear-vs-mlp-report",
                paper_probe_metric_value=0.93,
                candidate_probe_metric_value=0.93,
                paper_faithfulness_exception="Raw activation tied the paper MLP.",
            ),
        )

        decision = evaluate_cift_promotion_gate(bundle=_bundle(metadata), evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertIn(
            "raw_activation candidate_probe_metric_value must exceed paper_probe_metric_value",
            decision.failed_requirements,
        )

    def test_gate_accepts_method_contract_from_probe_competition_report(self) -> None:
        report_id = "synthetic-linear-vs-mlp-report"
        metadata = replace(
            _metadata("runtime_candidate"),
            evaluation_report_ids=(*_REQUIRED_REPORT_IDS, report_id),
        )
        competition_report = compare_cift_probe_candidates(
            CiftProbeCompetitionConfig(
                report_id=report_id,
                paper_probe=_probe_run(
                    source_report_id="synthetic-paper-mlp-report",
                    probe_architecture="mlp_128_64_1",
                    training_loss="bce_with_l1_softplus_weight_sparsity",
                    metric_value=0.91,
                    false_positive_rate=0.04,
                    true_positive_rate=0.91,
                ),
                candidate_probe=_probe_run(
                    source_report_id="synthetic-linear-report",
                    probe_architecture="linear_logistic_regression",
                    training_loss="bce",
                    metric_value=0.93,
                    false_positive_rate=0.03,
                    true_positive_rate=0.93,
                ),
                higher_is_better=True,
                created_at="2026-06-24T00:00:00Z",
            )
        )
        evidence = replace(
            _promotion_evidence(),
            report_artifacts=_report_artifacts((*_REQUIRED_REPORT_IDS, report_id)),
            paper_method=promotion_paper_method_from_probe_competition(competition_report),
        )

        decision = evaluate_cift_promotion_gate(bundle=_bundle(metadata), evidence=evidence)

        self.assertTrue(decision.eligible)
        self.assertEqual((), decision.failed_requirements)

    def test_gate_rejects_challenger_probe_that_underperforms_paper_probe(self) -> None:
        metadata = replace(
            _metadata("runtime_candidate"),
            evaluation_report_ids=(*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report"),
        )
        evidence = replace(
            _promotion_evidence(),
            report_artifacts=_report_artifacts((*_REQUIRED_REPORT_IDS, "synthetic-linear-vs-mlp-report")),
            paper_method=replace(
                _paper_method_evidence(),
                probe_architecture="linear_logistic_regression",
                training_loss="regularized_logistic_loss",
                head_to_head_report_id="synthetic-linear-vs-mlp-report",
                paper_probe_metric_value=0.94,
                candidate_probe_metric_value=0.91,
            ),
        )

        decision = evaluate_cift_promotion_gate(bundle=_bundle(metadata), evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertIn(
            "candidate_probe_metric_value must meet or exceed paper_probe_metric_value",
            decision.failed_requirements,
        )

    def test_gate_rejects_missing_report_lineage(self) -> None:
        metadata = replace(
            _metadata("runtime_candidate"),
            evaluation_report_ids=tuple(
                report_id for report_id in _REQUIRED_REPORT_IDS if report_id != "synthetic-patching-report"
            ),
        )
        bundle = _bundle(metadata)
        evidence = _promotion_evidence()

        decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertEqual(("synthetic-patching-report",), decision.missing_report_ids)
        with self.assertRaisesRegex(CiftPromotionGateError, "synthetic-patching-report"):
            assert_cift_runtime_promotion_eligible(bundle=bundle, evidence=evidence)

    def test_gate_rejects_overlapping_lab_splits(self) -> None:
        bundle = _bundle(_metadata("runtime_candidate"))
        evidence = replace(_promotion_evidence(), heldout_split_id="synthetic-cift-lab/calibration")

        decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)

        self.assertFalse(decision.eligible)
        self.assertIn("promotion split ids must be distinct", decision.failed_requirements)

    def test_gate_rejects_incomplete_or_failing_evidence(self) -> None:
        cases = (
            (
                "offline candidate",
                _bundle(_metadata("offline_research_candidate")),
                _promotion_evidence(),
                "candidate_status must be runtime_candidate",
            ),
            (
                "bad evidence schema",
                _bundle(_metadata("runtime_candidate")),
                replace(_promotion_evidence(), schema_version="cift_promotion_evidence/v0"),
                "schema_version must be cift_promotion_evidence/v1",
            ),
            (
                "mismatched dataset",
                _bundle(_metadata("runtime_candidate")),
                replace(_promotion_evidence(), training_dataset_id="other-dataset"),
                "training_dataset_id must match bundle metadata",
            ),
            (
                "missing sealed holdout",
                _bundle(_metadata("runtime_candidate")),
                replace(_promotion_evidence(), sealed_holdout_split_id=None),
                "sealed_holdout_split_id must not be empty",
            ),
            (
                "metric below threshold",
                _bundle(_metadata("runtime_candidate")),
                replace(_promotion_evidence(), metric_value=0.72, metric_threshold=0.9),
                "metric_value must meet or exceed metric_threshold",
            ),
            (
                "ablation below threshold",
                _bundle(_metadata("runtime_candidate")),
                replace(_promotion_evidence(), ablation_delta=0.02, ablation_delta_threshold=0.1),
                "ablation_delta must meet or exceed ablation_delta_threshold",
            ),
            (
                "missing runtime prevention report",
                _bundle(_metadata("runtime_candidate")),
                replace(_promotion_evidence(), runtime_prevention_report_id=""),
                "runtime_prevention_report_id must not be empty",
            ),
            (
                "non-finite metric",
                _bundle(_metadata("runtime_candidate")),
                replace(_promotion_evidence(), metric_value=float("inf")),
                "metric_value must be finite",
            ),
        )
        for case_name, bundle, evidence, expected_failure in cases:
            with self.subTest(case_name=case_name):
                decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)

                self.assertFalse(decision.eligible)
                self.assertIn(expected_failure, decision.failed_requirements)


if __name__ == "__main__":
    unittest.main()
