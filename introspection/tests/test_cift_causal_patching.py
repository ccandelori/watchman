from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from aegis_introspection.cift_causal_patching import (
    CiftCounterfactualPatchingConfig,
    run_cift_counterfactual_patching,
)
from aegis_introspection.cift_model_training import (
    CiftTrainingArtifact,
    CiftTrainingArtifactMetadata,
    cift_training_artifact_to_pickle_record,
)
from aegis_introspection.lineage import sha256_file

from aegis.core.contracts import Action
from aegis.detectors.cift_runtime import CiftRuntimeLinearModel, cift_runtime_model_to_dict


class CiftCounterfactualPatchingTest(unittest.TestCase):
    def test_paired_feature_vector_patching_flips_runtime_detector_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "artifact.pkl"
            runtime_model_path = root / "runtime_model.json"
            output_path = root / "patching_report.json"
            _write_artifact(
                path=artifact_path,
                labels=(
                    "secret_present_safe",
                    "exfiltration_intent",
                    "secret_present_safe",
                    "exfiltration_intent",
                ),
                variants=("v000", "v000", "v001", "v001"),
                tasks=("alpha", "alpha", "beta", "beta"),
            )
            _write_runtime_model(
                path=runtime_model_path,
                source_artifact_sha256=sha256_file(artifact_path),
                label_names=("secret_present_safe", "exfiltration_intent"),
                task_name="safe_secret_vs_exfiltration",
            )

            report = run_cift_counterfactual_patching(
                CiftCounterfactualPatchingConfig(
                    activation_artifact_path=artifact_path,
                    runtime_model_path=runtime_model_path,
                    output_path=output_path,
                    report_id="synthetic-patching-report",
                    created_at="2026-06-24T00:00:00Z",
                    minimum_flip_rate=0.95,
                    allow_sealed_holdout=False,
                )
            )

            persisted = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual("aegis_introspection.cift_counterfactual_patching/v1", report.schema_version)
            self.assertEqual("synthetic-patching-report", persisted["report_id"])
            self.assertEqual("paired_feature_vector_replacement", persisted["intervention_type"])
            self.assertEqual("runtime_detector_decision", persisted["claim_scope"])
            self.assertFalse(persisted["transformer_hidden_state_patching"])
            self.assertEqual(2, persisted["pair_count"])
            self.assertEqual(1.0, persisted["safe_to_exfil_block_rate"])
            self.assertEqual(1.0, persisted["exfil_to_safe_allow_rate"])
            self.assertEqual("block", persisted["pairs"][0]["safe_to_exfil_patch"]["action"])
            self.assertEqual("allow", persisted["pairs"][0]["exfil_to_safe_patch"]["action"])
            self.assertIn("does not patch transformer hidden states", persisted["paper_faithfulness_limitation"])

    def test_patching_pairs_label_specific_variants_by_task_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "artifact.pkl"
            runtime_model_path = root / "runtime_model.json"
            output_path = root / "patching_report.json"
            _write_artifact(
                path=artifact_path,
                labels=(
                    "secret_present_safe",
                    "exfiltration_intent",
                    "secret_present_safe",
                    "exfiltration_intent",
                ),
                variants=(
                    "secret_present_safe-v000",
                    "exfiltration_intent-v000",
                    "secret_present_safe-v001",
                    "exfiltration_intent-v001",
                ),
                tasks=("alpha", "alpha", "beta", "beta"),
            )
            _write_runtime_model(
                path=runtime_model_path,
                source_artifact_sha256=sha256_file(artifact_path),
                label_names=("secret_present_safe", "exfiltration_intent"),
                task_name="safe_secret_vs_exfiltration",
            )

            report = run_cift_counterfactual_patching(
                CiftCounterfactualPatchingConfig(
                    activation_artifact_path=artifact_path,
                    runtime_model_path=runtime_model_path,
                    output_path=output_path,
                    report_id="synthetic-patching-report",
                    created_at="2026-06-24T00:00:00Z",
                    minimum_flip_rate=0.95,
                    allow_sealed_holdout=False,
                )
            )

        self.assertEqual(2, report.pair_count)
        self.assertEqual(
            "safe=secret_present_safe-v000;exfil=exfiltration_intent-v000",
            report.pairs[0].variant,
        )

    def test_patching_maps_aggregate_non_exfiltration_class_to_safe_secret_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "artifact.pkl"
            runtime_model_path = root / "runtime_model.json"
            output_path = root / "patching_report.json"
            _write_artifact(
                path=artifact_path,
                labels=(
                    "secret_present_safe",
                    "exfiltration_intent",
                    "benign",
                    "benign",
                ),
                variants=("secret_present_safe-v000", "exfiltration_intent-v000", "benign-v001", "benign-v002"),
                tasks=("alpha", "alpha", "beta", "beta"),
            )
            _write_runtime_model(
                path=runtime_model_path,
                source_artifact_sha256=sha256_file(artifact_path),
                label_names=("non_exfiltration", "exfiltration_intent"),
                task_name="non_exfiltration_vs_exfiltration",
            )

            report = run_cift_counterfactual_patching(
                CiftCounterfactualPatchingConfig(
                    activation_artifact_path=artifact_path,
                    runtime_model_path=runtime_model_path,
                    output_path=output_path,
                    report_id="synthetic-patching-report",
                    created_at="2026-06-24T00:00:00Z",
                    minimum_flip_rate=0.95,
                    allow_sealed_holdout=False,
                )
            )

        self.assertEqual(1, report.pair_count)
        self.assertEqual("safe=secret_present_safe-v000;exfil=exfiltration_intent-v000", report.pairs[0].variant)

    def test_patching_skips_ambiguous_pairs_and_reports_exact_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "artifact.pkl"
            runtime_model_path = root / "runtime_model.json"
            output_path = root / "patching_report.json"
            _write_artifact(
                path=artifact_path,
                labels=(
                    "secret_present_safe",
                    "secret_present_safe",
                    "exfiltration_intent",
                    "secret_present_safe",
                    "exfiltration_intent",
                ),
                variants=("v000", "v001", "v000", "v002", "v002"),
                tasks=("alpha", "alpha", "alpha", "beta", "beta"),
            )
            _write_runtime_model(
                path=runtime_model_path,
                source_artifact_sha256=sha256_file(artifact_path),
                label_names=("secret_present_safe", "exfiltration_intent"),
                task_name="safe_secret_vs_exfiltration",
            )

            report = run_cift_counterfactual_patching(
                CiftCounterfactualPatchingConfig(
                    activation_artifact_path=artifact_path,
                    runtime_model_path=runtime_model_path,
                    output_path=output_path,
                    report_id="synthetic-patching-report",
                    created_at="2026-06-24T00:00:00Z",
                    minimum_flip_rate=0.95,
                    allow_sealed_holdout=False,
                )
            )

            persisted = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(1, report.pair_count)
        self.assertEqual(1, persisted["skipped_ambiguous_pair_count"])
        self.assertIn("task=alpha", persisted["skipped_ambiguous_pair_keys"][0])
        self.assertEqual("beta_credentials", persisted["pairs"][0]["family"])


def _write_artifact(
    path: Path,
    labels: tuple[str, ...],
    variants: tuple[str, ...],
    tasks: tuple[str, ...],
) -> None:
    feature_rows = tuple(
        (float(index) / 10.0, 1.0 if label == "exfiltration_intent" else 0.0)
        for index, label in enumerate(labels)
    )
    artifact = CiftTrainingArtifact(
        metadata=CiftTrainingArtifactMetadata(
            model_id="Qwen/Qwen3-test",
            revision="main",
            selected_device="cpu",
            hidden_size=4096,
            layer_count=36,
            tokenizer_fingerprint_sha256="b" * 64,
            special_tokens_map_sha256="c" * 64,
            chat_template_sha256="d" * 64,
            layer_indices=(19,),
            pooling_methods=("selected_choice_window",),
        ),
        example_ids=tuple(
            f"trace-assignment-codex-{task}-{label}-{variant}"
            for task, label, variant in zip(tasks, labels, variants, strict=True)
        ),
        labels=labels,
        families=tuple(f"{task}_credentials" for task in tasks),
        texts=tuple(f"{label} {task}" for label, task in zip(labels, tasks, strict=True)),
        tags=(
            *tuple(
                _tags(task=task, variant=variant, label=label)
                for task, variant, label in zip(tasks, variants, labels, strict=True)
            ),
        ),
        features={
            "selected_choice_window_layer_19": np.asarray(feature_rows, dtype=np.float32)
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(cift_training_artifact_to_pickle_record(artifact), file)


def _tags(task: str, variant: str, label: str) -> tuple[str, ...]:
    return (
        "trace_collection",
        f"label:{label}",
        f"family:{task}_credentials",
        f"task:{task}",
        "participant:codex",
        f"variant:{variant}",
        "credential_type:synthetic",
    )


def _write_runtime_model(
    path: Path,
    source_artifact_sha256: str,
    label_names: tuple[str, str],
    task_name: str,
) -> None:
    model = CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id="synthetic-runtime-cift",
        source_model_id="Qwen/Qwen3-test",
        source_revision="main",
        source_selected_device="cpu",
        source_hidden_size=2,
        source_layer_count=1,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        training_dataset_id="synthetic-cift-lab",
        source_artifact_sha256=source_artifact_sha256,
        evaluation_report_ids=("synthetic-patching-report",),
        task_name=task_name,
        feature_key="selected_choice_window_layer_19",
        feature_count=2,
        label_names=label_names,
        positive_label="exfiltration_intent",
        positive_class_index=1,
        class_indices=(0, 1),
        decision_threshold=0.5,
        score_semantics="full_train_classifier_probability",
        confidence=0.91,
        candidate_status="offline_research_candidate",
        scaler_mean=(0.0, 0.0),
        scaler_scale=(1.0, 1.0),
        logistic_coefficients=(0.0, 4.0),
        logistic_intercept=-2.0,
        negative_action=Action.ALLOW,
        positive_action=Action.BLOCK,
    )
    path.write_text(json.dumps(cift_runtime_model_to_dict(model), sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
