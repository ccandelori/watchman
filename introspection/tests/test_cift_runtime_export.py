from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from aegis_introspection.cift_model_bundle import (
    CandidateStatus,
    CiftModelBundle,
    CiftModelBundleMetadata,
    predict_cift_model_bundle,
    save_cift_model_bundle,
)
from aegis_introspection.cift_model_training import CiftLinearLogisticClassifier
from aegis_introspection.cift_paper_mlp import CiftPaperMlpClassifier, CiftPaperMlpConfig
from aegis_introspection.cift_promotion_gate import (
    CiftPaperMethodContract,
    CiftPromotionEvidence,
    CiftPromotionReportArtifact,
    cift_promotion_evidence_to_json,
)
from aegis_introspection.cift_runtime_export import (
    Action,
    CiftRuntimeModelExportError,
    ExportCiftRuntimeModelConfig,
    export_cift_runtime_model,
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
    return CiftModelBundle(metadata=metadata, classifier=_FakePipeline(), calibrator=None)


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


def _load_cift_runtime_model(path: Path) -> object:
    from aegis.detectors.cift_runtime import load_cift_runtime_model

    return load_cift_runtime_model(path)


def _predict_cift_runtime_model(model: object, feature_vector: tuple[float, ...]) -> object:
    from aegis.detectors.cift_runtime import predict_cift_runtime_model

    return predict_cift_runtime_model(model=model, feature_vector=feature_vector)


class CiftRuntimeExportTest(unittest.TestCase):
    def test_internal_linear_bundle_exports_runtime_scorable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.pkl"
            output_path = root / "runtime_model.json"
            feature_vector = (2.0, 1.5)
            classifier = CiftLinearLogisticClassifier(
                input_dim=2,
                max_epochs=250,
                regularization_c=1.0,
                random_seed=17,
            ).fit(
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
            bundle = CiftModelBundle(
                metadata=_metadata("offline_research_candidate"),
                classifier=classifier,
                calibrator=None,
            )
            expected_probability = predict_cift_model_bundle(
                bundle=bundle,
                feature_matrix=np.asarray([feature_vector], dtype=np.float32),
            )[0].positive_probability
            save_cift_model_bundle(path=bundle_path, bundle=bundle)

            exported = export_cift_runtime_model(
                ExportCiftRuntimeModelConfig(
                    bundle_path=bundle_path,
                    output_path=output_path,
                    model_bundle_id="synthetic-linear-cift",
                    confidence=0.86,
                    negative_action=Action.ALLOW,
                    positive_action=Action.WARN,
                    promotion_evidence_path=None,
                    allow_preview_without_promotion=False,
                )
            )
            record = json.loads(output_path.read_text(encoding="utf-8"))
            loaded = _load_cift_runtime_model(output_path)
            prediction = _predict_cift_runtime_model(model=loaded, feature_vector=feature_vector)

        self.assertEqual(exported, loaded)
        self.assertEqual("main", record["source_revision"])
        self.assertEqual(4096, record["source_hidden_size"])
        self.assertEqual(36, record["source_layer_count"])
        self.assertEqual("b" * 64, record["tokenizer_fingerprint_sha256"])
        self.assertEqual("c" * 64, record["special_tokens_map_sha256"])
        self.assertEqual("d" * 64, record["chat_template_sha256"])
        self.assertAlmostEqual(expected_probability, prediction.score, places=6)

    def test_paper_mlp_bundle_exports_runtime_scorable_mlp_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.pkl"
            output_path = root / "runtime_model.json"
            bundle = _mlp_bundle(_metadata("offline_research_candidate"))
            feature_vector = (2.0, 1.5)
            expected_probability = float(
                bundle.classifier.predict_proba(np.asarray([feature_vector], dtype=np.float32))[0, 1]
            )
            save_cift_model_bundle(path=bundle_path, bundle=bundle)

            exported = export_cift_runtime_model(
                ExportCiftRuntimeModelConfig(
                    bundle_path=bundle_path,
                    output_path=output_path,
                    model_bundle_id="synthetic-paper-mlp-cift",
                    confidence=0.86,
                    negative_action=Action.ALLOW,
                    positive_action=Action.WARN,
                    promotion_evidence_path=None,
                    allow_preview_without_promotion=False,
                )
            )
            record = json.loads(output_path.read_text(encoding="utf-8"))
            loaded = _load_cift_runtime_model(output_path)
            prediction = _predict_cift_runtime_model(model=loaded, feature_vector=feature_vector)

        self.assertEqual(exported, loaded)
        self.assertEqual("aegis.cift_runtime_mlp/v1", record["schema_version"])
        self.assertEqual("main", record["source_revision"])
        self.assertEqual("mlp_128_64_1", record["probe_architecture"])
        self.assertEqual(128, len(record["first_bias"]))
        self.assertEqual(64, len(record["second_bias"]))
        self.assertAlmostEqual(expected_probability, prediction.score, places=6)

    def test_runtime_candidate_export_writes_lab_promotion_metadata_without_changing_runtime_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.pkl"
            evidence_path = root / "promotion_evidence.json"
            output_path = root / "runtime_model.json"
            bundle = _mlp_bundle(_metadata("runtime_candidate"))
            evidence = _promotion_evidence()
            save_cift_model_bundle(path=bundle_path, bundle=bundle)
            evidence_path.write_text(json.dumps(cift_promotion_evidence_to_json(evidence), indent=2), encoding="utf-8")

            exported = export_cift_runtime_model(
                ExportCiftRuntimeModelConfig(
                    bundle_path=bundle_path,
                    output_path=output_path,
                    model_bundle_id="synthetic-runtime-cift",
                    confidence=0.86,
                    negative_action=Action.ALLOW,
                    positive_action=Action.BLOCK,
                    promotion_evidence_path=evidence_path,
                    allow_preview_without_promotion=False,
                )
            )
            record = json.loads(output_path.read_text(encoding="utf-8"))
            loaded = _load_cift_runtime_model(output_path)

        self.assertEqual(exported, loaded)
        self.assertEqual("runtime_candidate", record["candidate_status"])
        self.assertEqual("synthetic-runtime-cift", record["model_bundle_id"])
        self.assertEqual("main", record["source_revision"])
        self.assertEqual("cift_promotion_gates/v1", record["promotion_gates"]["schema_version"])
        self.assertEqual(
            "synthetic-promotion-evidence",
            record["promotion_gates"]["runtime_candidate"]["evidence_id"],
        )
        self.assertEqual(
            "runtime_candidate_promotion_only",
            record["promotion_gates"]["runtime_candidate"]["eligibility_scope"],
        )
        self.assertFalse(record["promotion_gates"]["runtime_candidate"]["production_release_eligible"])
        self.assertTrue(record["promotion_gates"]["runtime_candidate"]["requires_certification_binding"])
        self.assertEqual(9, len(record["promotion_gates"]["runtime_candidate"]["report_artifacts"]))

    def test_runtime_candidate_export_rejects_linear_bundle_claiming_paper_mlp_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.pkl"
            evidence_path = root / "promotion_evidence.json"
            output_path = root / "runtime_model.json"
            bundle = _bundle(_metadata("runtime_candidate"))
            evidence = _promotion_evidence()
            save_cift_model_bundle(path=bundle_path, bundle=bundle)
            evidence_path.write_text(json.dumps(cift_promotion_evidence_to_json(evidence), indent=2), encoding="utf-8")

            with self.assertRaisesRegex(CiftRuntimeModelExportError, "mlp_128_64_1"):
                export_cift_runtime_model(
                    ExportCiftRuntimeModelConfig(
                        bundle_path=bundle_path,
                        output_path=output_path,
                        model_bundle_id="synthetic-runtime-cift",
                        confidence=0.86,
                        negative_action=Action.ALLOW,
                        positive_action=Action.BLOCK,
                        promotion_evidence_path=evidence_path,
                        allow_preview_without_promotion=False,
                    )
                )

            self.assertFalse(output_path.exists())

    def test_runtime_candidate_export_requires_promotion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.pkl"
            output_path = root / "runtime_model.json"
            bundle: CiftModelBundle = _bundle(_metadata("runtime_candidate"))
            save_cift_model_bundle(path=bundle_path, bundle=bundle)

            with self.assertRaisesRegex(CiftRuntimeModelExportError, "promotion-evidence"):
                export_cift_runtime_model(
                    ExportCiftRuntimeModelConfig(
                        bundle_path=bundle_path,
                        output_path=output_path,
                        model_bundle_id="synthetic-runtime-cift",
                        confidence=0.86,
                        negative_action=Action.ALLOW,
                        positive_action=Action.BLOCK,
                        promotion_evidence_path=None,
                        allow_preview_without_promotion=False,
                    )
                )

    def test_runtime_candidate_preview_export_skips_promotion_metadata_for_benchmark_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.pkl"
            output_path = root / "runtime_model.json"
            bundle: CiftModelBundle = _bundle(_metadata("runtime_candidate"))
            save_cift_model_bundle(path=bundle_path, bundle=bundle)

            exported = export_cift_runtime_model(
                ExportCiftRuntimeModelConfig(
                    bundle_path=bundle_path,
                    output_path=output_path,
                    model_bundle_id="synthetic-runtime-cift-preview",
                    confidence=0.42,
                    negative_action=Action.ALLOW,
                    positive_action=Action.BLOCK,
                    promotion_evidence_path=None,
                    allow_preview_without_promotion=True,
                )
            )
            record = json.loads(output_path.read_text(encoding="utf-8"))
            loaded = _load_cift_runtime_model(output_path)

        self.assertEqual(exported, loaded)
        self.assertEqual("offline_research_candidate", record["candidate_status"])
        self.assertEqual("block", record["positive_action"])
        self.assertNotIn("promotion_gates", record)

    def test_offline_research_candidate_export_does_not_require_promotion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.pkl"
            output_path = root / "runtime_model.json"
            bundle: CiftModelBundle = _bundle(_metadata("offline_research_candidate"))
            save_cift_model_bundle(path=bundle_path, bundle=bundle)

            exported = export_cift_runtime_model(
                ExportCiftRuntimeModelConfig(
                    bundle_path=bundle_path,
                    output_path=output_path,
                    model_bundle_id="synthetic-offline-cift",
                    confidence=0.42,
                    negative_action=Action.ALLOW,
                    positive_action=Action.WARN,
                    promotion_evidence_path=None,
                    allow_preview_without_promotion=False,
                )
            )
            record = json.loads(output_path.read_text(encoding="utf-8"))
            loaded = _load_cift_runtime_model(output_path)

        self.assertEqual(exported, loaded)
        self.assertEqual("offline_research_candidate", record["candidate_status"])
        self.assertNotIn("promotion_gates", record)

    def test_runtime_candidate_export_rejects_failing_promotion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.pkl"
            evidence_path = root / "promotion_evidence.json"
            output_path = root / "runtime_model.json"
            bundle: CiftModelBundle = _mlp_bundle(_metadata("runtime_candidate"))
            evidence = replace(_promotion_evidence(), metric_value=0.2, metric_threshold=0.9)
            save_cift_model_bundle(path=bundle_path, bundle=bundle)
            evidence_path.write_text(json.dumps(cift_promotion_evidence_to_json(evidence), indent=2), encoding="utf-8")

            with self.assertRaisesRegex(CiftRuntimeModelExportError, "metric_value"):
                export_cift_runtime_model(
                    ExportCiftRuntimeModelConfig(
                        bundle_path=bundle_path,
                        output_path=output_path,
                        model_bundle_id="synthetic-runtime-cift",
                        confidence=0.86,
                        negative_action=Action.ALLOW,
                        positive_action=Action.BLOCK,
                        promotion_evidence_path=evidence_path,
                        allow_preview_without_promotion=False,
                    )
                )

            self.assertFalse(output_path.exists())

    def test_runtime_candidate_export_rejects_non_preventive_positive_action(self) -> None:
        for positive_action in (Action.WARN, Action.SANITIZE):
            with self.subTest(positive_action=positive_action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle_path = root / "bundle.pkl"
                evidence_path = root / "promotion_evidence.json"
                output_path = root / "runtime_model.json"
                bundle: CiftModelBundle = _bundle(_metadata("runtime_candidate"))
                evidence = _promotion_evidence()
                save_cift_model_bundle(path=bundle_path, bundle=bundle)
                evidence_path.write_text(
                    json.dumps(cift_promotion_evidence_to_json(evidence), indent=2),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(CiftRuntimeModelExportError, "positive_action"):
                    export_cift_runtime_model(
                        ExportCiftRuntimeModelConfig(
                            bundle_path=bundle_path,
                            output_path=output_path,
                            model_bundle_id="synthetic-runtime-cift",
                            confidence=0.86,
                            negative_action=Action.ALLOW,
                            positive_action=positive_action,
                            promotion_evidence_path=evidence_path,
                            allow_preview_without_promotion=False,
                        )
                    )

                self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
