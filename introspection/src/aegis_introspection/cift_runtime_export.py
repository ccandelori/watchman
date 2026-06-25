from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

RUNTIME_SRC_PATH = Path(__file__).resolve().parents[3] / "src"
if str(RUNTIME_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC_PATH))

from aegis.core.contracts import Action, JsonValue
from aegis.detectors.cift_runtime import (
    CiftRuntimeLinearModel,
    CiftRuntimeMlpModel,
    CiftRuntimeModel,
    cift_runtime_model_to_dict,
)
from aegis_introspection.cift_model_bundle import CiftModelBundle, load_cift_model_bundle
from aegis_introspection.cift_model_training import CiftLinearLogisticClassifier
from aegis_introspection.cift_paper_mlp import CiftPaperMlpClassifier
from aegis_introspection.cift_promotion_gate import (
    CiftPromotionGateError,
    assert_cift_runtime_promotion_eligible,
    cift_promotion_gates_to_json,
    evaluate_cift_promotion_gate,
    load_cift_promotion_evidence,
)


class CiftRuntimeModelExportError(ValueError):
    """Raised when a trained CIFT bundle cannot be exported to the runtime artifact schema."""


@dataclass(frozen=True)
class ExportCiftRuntimeModelConfig:
    bundle_path: Path
    output_path: Path
    model_bundle_id: str
    confidence: float
    negative_action: Action
    positive_action: Action
    promotion_evidence_path: Path | None
    allow_preview_without_promotion: bool


def export_cift_runtime_model(config: ExportCiftRuntimeModelConfig) -> CiftRuntimeModel:
    _validate_config(config)
    bundle = load_cift_model_bundle(config.bundle_path)
    promotion_gates = _promotion_gates_from_config(bundle=bundle, config=config)
    model = _runtime_model_from_bundle(bundle=bundle, config=config)
    record = cift_runtime_model_to_dict(model)
    if promotion_gates is not None:
        record["promotion_gates"] = promotion_gates
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model


def _runtime_model_from_bundle(
    bundle: CiftModelBundle,
    config: ExportCiftRuntimeModelConfig,
) -> CiftRuntimeModel:
    classifier = bundle.classifier
    if isinstance(classifier, CiftPaperMlpClassifier):
        return _runtime_mlp_model_from_bundle(bundle=bundle, config=config, classifier=classifier)
    return _runtime_linear_model_from_bundle(bundle=bundle, config=config)


def _export_candidate_status(bundle: CiftModelBundle, config: ExportCiftRuntimeModelConfig) -> str:
    if (
        bundle.metadata.candidate_status == "runtime_candidate"
        and config.promotion_evidence_path is None
        and config.allow_preview_without_promotion
    ):
        return "offline_research_candidate"
    return bundle.metadata.candidate_status


def _runtime_linear_model_from_bundle(
    bundle: CiftModelBundle,
    config: ExportCiftRuntimeModelConfig,
) -> CiftRuntimeLinearModel:
    classifier = bundle.classifier
    if isinstance(classifier, CiftLinearLogisticClassifier):
        return _runtime_internal_linear_model_from_bundle(bundle=bundle, config=config, classifier=classifier)
    return _runtime_pipeline_linear_model_from_bundle(bundle=bundle, config=config)


def _runtime_internal_linear_model_from_bundle(
    bundle: CiftModelBundle,
    config: ExportCiftRuntimeModelConfig,
    classifier: CiftLinearLogisticClassifier,
) -> CiftRuntimeLinearModel:
    parameters = classifier.runtime_parameters()
    feature_count = bundle.metadata.feature_count
    label_names = _two_string_tuple(bundle.metadata.label_names)
    positive_class_index = label_names.index(bundle.metadata.positive_label)
    return CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id=config.model_bundle_id,
        source_model_id=bundle.metadata.source_model_id,
        source_revision=bundle.metadata.source_revision,
        source_selected_device=bundle.metadata.source_selected_device,
        source_hidden_size=bundle.metadata.source_hidden_size,
        source_layer_count=bundle.metadata.source_layer_count,
        tokenizer_fingerprint_sha256=bundle.metadata.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=bundle.metadata.special_tokens_map_sha256,
        chat_template_sha256=bundle.metadata.chat_template_sha256,
        training_dataset_id=bundle.metadata.training_dataset_id,
        source_artifact_sha256=bundle.metadata.source_artifact_sha256.lower(),
        evaluation_report_ids=bundle.metadata.evaluation_report_ids,
        task_name=bundle.metadata.task_name,
        feature_key=bundle.metadata.activation_feature_key,
        feature_count=feature_count,
        label_names=label_names,
        positive_label=bundle.metadata.positive_label,
        positive_class_index=positive_class_index,
        class_indices=_two_int_tuple_from_attribute(owner=classifier, attribute_name="classes_"),
        decision_threshold=bundle.metadata.decision_threshold,
        score_semantics=bundle.metadata.score_semantics,
        confidence=config.confidence,
        candidate_status=_export_candidate_status(bundle=bundle, config=config),
        scaler_mean=_float_tuple_from_sequence(
            values=parameters.mean,
            field_name="mean",
            expected_length=feature_count,
        ),
        scaler_scale=_float_tuple_from_sequence(
            values=parameters.scale,
            field_name="scale",
            expected_length=feature_count,
        ),
        logistic_coefficients=_float_tuple_from_sequence(
            values=parameters.coefficients,
            field_name="coefficients",
            expected_length=feature_count,
        ),
        logistic_intercept=parameters.intercept,
        negative_action=config.negative_action,
        positive_action=config.positive_action,
    )


def _runtime_pipeline_linear_model_from_bundle(
    bundle: CiftModelBundle,
    config: ExportCiftRuntimeModelConfig,
) -> CiftRuntimeLinearModel:
    classifier = bundle.classifier
    scaler = _pipeline_step(classifier=classifier, step_name="standardscaler")
    logistic = _pipeline_step(classifier=classifier, step_name="logisticregression")
    feature_count = bundle.metadata.feature_count
    class_indices = _two_int_tuple_from_attribute(owner=logistic, attribute_name="classes_")
    label_names = _two_string_tuple(bundle.metadata.label_names)
    positive_class_index = label_names.index(bundle.metadata.positive_label)
    return CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id=config.model_bundle_id,
        source_model_id=bundle.metadata.source_model_id,
        source_revision=bundle.metadata.source_revision,
        source_selected_device=bundle.metadata.source_selected_device,
        source_hidden_size=bundle.metadata.source_hidden_size,
        source_layer_count=bundle.metadata.source_layer_count,
        tokenizer_fingerprint_sha256=bundle.metadata.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=bundle.metadata.special_tokens_map_sha256,
        chat_template_sha256=bundle.metadata.chat_template_sha256,
        training_dataset_id=bundle.metadata.training_dataset_id,
        source_artifact_sha256=bundle.metadata.source_artifact_sha256.lower(),
        evaluation_report_ids=bundle.metadata.evaluation_report_ids,
        task_name=bundle.metadata.task_name,
        feature_key=bundle.metadata.activation_feature_key,
        feature_count=feature_count,
        label_names=label_names,
        positive_label=bundle.metadata.positive_label,
        positive_class_index=positive_class_index,
        class_indices=class_indices,
        decision_threshold=bundle.metadata.decision_threshold,
        score_semantics=bundle.metadata.score_semantics,
        confidence=config.confidence,
        candidate_status=_export_candidate_status(bundle=bundle, config=config),
        scaler_mean=_float_tuple_from_attribute(
            owner=scaler,
            attribute_name="mean_",
            expected_length=feature_count,
        ),
        scaler_scale=_float_tuple_from_attribute(
            owner=scaler,
            attribute_name="scale_",
            expected_length=feature_count,
        ),
        logistic_coefficients=_coefficient_tuple(logistic=logistic, expected_length=feature_count),
        logistic_intercept=_single_float_from_attribute(owner=logistic, attribute_name="intercept_"),
        negative_action=config.negative_action,
        positive_action=config.positive_action,
    )


def _runtime_mlp_model_from_bundle(
    bundle: CiftModelBundle,
    config: ExportCiftRuntimeModelConfig,
    classifier: CiftPaperMlpClassifier,
) -> CiftRuntimeMlpModel:
    parameters = classifier.runtime_parameters()
    feature_count = bundle.metadata.feature_count
    label_names = _two_string_tuple(bundle.metadata.label_names)
    positive_class_index = label_names.index(bundle.metadata.positive_label)
    return CiftRuntimeMlpModel(
        schema_version="aegis.cift_runtime_mlp/v1",
        model_bundle_id=config.model_bundle_id,
        source_model_id=bundle.metadata.source_model_id,
        source_revision=bundle.metadata.source_revision,
        source_selected_device=bundle.metadata.source_selected_device,
        source_hidden_size=bundle.metadata.source_hidden_size,
        source_layer_count=bundle.metadata.source_layer_count,
        tokenizer_fingerprint_sha256=bundle.metadata.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=bundle.metadata.special_tokens_map_sha256,
        chat_template_sha256=bundle.metadata.chat_template_sha256,
        training_dataset_id=bundle.metadata.training_dataset_id,
        source_artifact_sha256=bundle.metadata.source_artifact_sha256.lower(),
        evaluation_report_ids=bundle.metadata.evaluation_report_ids,
        task_name=bundle.metadata.task_name,
        feature_key=bundle.metadata.activation_feature_key,
        feature_count=feature_count,
        label_names=label_names,
        positive_label=bundle.metadata.positive_label,
        positive_class_index=positive_class_index,
        class_indices=_two_int_tuple_from_attribute(owner=classifier, attribute_name="classes_"),
        decision_threshold=bundle.metadata.decision_threshold,
        score_semantics=bundle.metadata.score_semantics,
        confidence=config.confidence,
        candidate_status=_export_candidate_status(bundle=bundle, config=config),
        probe_architecture="mlp_128_64_1",
        raw_layer_weights=_float_tuple_from_value(
            value=parameters.raw_layer_weights,
            field_name="raw_layer_weights",
            expected_length=feature_count,
        ),
        first_weights=_float_matrix_from_value(
            value=parameters.first_weights,
            field_name="first_weights",
            expected_rows=feature_count,
            expected_columns=128,
        ),
        first_bias=_float_tuple_from_value(
            value=parameters.first_bias,
            field_name="first_bias",
            expected_length=128,
        ),
        second_weights=_float_matrix_from_value(
            value=parameters.second_weights,
            field_name="second_weights",
            expected_rows=128,
            expected_columns=64,
        ),
        second_bias=_float_tuple_from_value(
            value=parameters.second_bias,
            field_name="second_bias",
            expected_length=64,
        ),
        output_weights=_single_column_tuple_from_value(
            value=parameters.output_weights,
            field_name="output_weights",
            expected_length=64,
        ),
        output_bias=_single_float_from_value(value=parameters.output_bias, field_name="output_bias"),
        negative_action=config.negative_action,
        positive_action=config.positive_action,
    )


def _promotion_gates_from_config(
    bundle: CiftModelBundle,
    config: ExportCiftRuntimeModelConfig,
) -> dict[str, JsonValue] | None:
    if bundle.metadata.candidate_status != "runtime_candidate":
        return None
    _validate_runtime_candidate_positive_action(config.positive_action)
    if config.promotion_evidence_path is None:
        if config.allow_preview_without_promotion:
            return None
        raise CiftRuntimeModelExportError("runtime_candidate export requires --promotion-evidence.")
    evidence = load_cift_promotion_evidence(config.promotion_evidence_path)
    try:
        assert_cift_runtime_promotion_eligible(bundle=bundle, evidence=evidence)
    except CiftPromotionGateError as exc:
        raise CiftRuntimeModelExportError(str(exc)) from exc
    decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)
    return cift_promotion_gates_to_json(evidence=evidence, decision=decision)


def _validate_config(config: ExportCiftRuntimeModelConfig) -> None:
    if config.model_bundle_id == "":
        raise CiftRuntimeModelExportError("model_bundle_id must not be empty.")
    if config.confidence < 0.0 or config.confidence > 1.0:
        raise CiftRuntimeModelExportError("confidence must be in [0.0, 1.0].")


def _validate_runtime_candidate_positive_action(positive_action: Action) -> None:
    if positive_action not in (Action.BLOCK, Action.ESCALATE):
        raise CiftRuntimeModelExportError("runtime_candidate positive_action must be block or escalate.")


def _pipeline_step(classifier: object, step_name: str) -> object:
    named_steps = getattr(classifier, "named_steps", None)
    if not isinstance(named_steps, dict):
        raise CiftRuntimeModelExportError("CIFT classifier must be a pipeline with named_steps.")
    step = named_steps.get(step_name)
    if step is None:
        raise CiftRuntimeModelExportError(f"CIFT classifier pipeline is missing step '{step_name}'.")
    return step


def _two_string_tuple(values: tuple[str, ...]) -> tuple[str, str]:
    if len(values) != 2:
        raise CiftRuntimeModelExportError("CIFT runtime export requires exactly two label names.")
    return (values[0], values[1])


def _two_int_tuple_from_attribute(owner: object, attribute_name: str) -> tuple[int, int]:
    values = _list_from_attribute(owner=owner, attribute_name=attribute_name)
    if len(values) != 2:
        raise CiftRuntimeModelExportError(f"{attribute_name} must contain exactly two class indices.")
    first = values[0]
    second = values[1]
    if isinstance(first, bool) or not isinstance(first, int):
        raise CiftRuntimeModelExportError(f"{attribute_name}[0] must be an integer.")
    if isinstance(second, bool) or not isinstance(second, int):
        raise CiftRuntimeModelExportError(f"{attribute_name}[1] must be an integer.")
    return (first, second)


def _float_tuple_from_attribute(owner: object, attribute_name: str, expected_length: int) -> tuple[float, ...]:
    values = _list_from_attribute(owner=owner, attribute_name=attribute_name)
    return _float_tuple_from_list(values=values, field_name=attribute_name, expected_length=expected_length)


def _float_tuple_from_value(value: object, field_name: str, expected_length: int) -> tuple[float, ...]:
    values = _list_from_value(value=value, field_name=field_name)
    return _float_tuple_from_list(values=values, field_name=field_name, expected_length=expected_length)


def _float_tuple_from_sequence(values: tuple[float, ...], field_name: str, expected_length: int) -> tuple[float, ...]:
    return _float_tuple_from_list(
        values=list(values),
        field_name=field_name,
        expected_length=expected_length,
    )


def _float_tuple_from_list(values: list[object], field_name: str, expected_length: int) -> tuple[float, ...]:
    if len(values) != expected_length:
        raise CiftRuntimeModelExportError(f"{field_name} has {len(values)} values, but expected {expected_length}.")
    return tuple(_float_item(value=value, field_name=f"{field_name}[{index}]") for index, value in enumerate(values))


def _float_matrix_from_value(
    value: object,
    field_name: str,
    expected_rows: int,
    expected_columns: int,
) -> tuple[tuple[float, ...], ...]:
    rows = _list_from_value(value=value, field_name=field_name)
    if len(rows) != expected_rows:
        raise CiftRuntimeModelExportError(f"{field_name} has {len(rows)} rows, but expected {expected_rows}.")
    converted_rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        row_values = _list_from_value(value=row, field_name=f"{field_name}[{row_index}]")
        if len(row_values) != expected_columns:
            raise CiftRuntimeModelExportError(
                f"{field_name}[{row_index}] has {len(row_values)} values, expected {expected_columns}."
            )
        converted_rows.append(
            tuple(
                _float_item(value=item, field_name=f"{field_name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row_values)
            )
        )
    return tuple(converted_rows)


def _single_column_tuple_from_value(value: object, field_name: str, expected_length: int) -> tuple[float, ...]:
    rows = _list_from_value(value=value, field_name=field_name)
    if len(rows) != expected_length:
        raise CiftRuntimeModelExportError(f"{field_name} has {len(rows)} rows, but expected {expected_length}.")
    values: list[float] = []
    for row_index, row in enumerate(rows):
        row_values = _list_from_value(value=row, field_name=f"{field_name}[{row_index}]")
        if len(row_values) != 1:
            raise CiftRuntimeModelExportError(f"{field_name}[{row_index}] must contain exactly one value.")
        values.append(_float_item(value=row_values[0], field_name=f"{field_name}[{row_index}][0]"))
    return tuple(values)


def _coefficient_tuple(logistic: object, expected_length: int) -> tuple[float, ...]:
    values = _list_from_attribute(owner=logistic, attribute_name="coef_")
    if len(values) != 1 or not isinstance(values[0], list):
        raise CiftRuntimeModelExportError("coef_ must contain one coefficient row for binary logistic regression.")
    coefficient_row = cast(list[object], values[0])
    if len(coefficient_row) != expected_length:
        raise CiftRuntimeModelExportError(f"coef_[0] has {len(coefficient_row)} values, expected {expected_length}.")
    return tuple(
        _float_item(value=value, field_name=f"coef_[0][{index}]") for index, value in enumerate(coefficient_row)
    )


def _single_float_from_attribute(owner: object, attribute_name: str) -> float:
    values = _list_from_attribute(owner=owner, attribute_name=attribute_name)
    if len(values) != 1:
        raise CiftRuntimeModelExportError(f"{attribute_name} must contain exactly one value.")
    return _float_item(value=values[0], field_name=f"{attribute_name}[0]")


def _single_float_from_value(value: object, field_name: str) -> float:
    values = _list_from_value(value=value, field_name=field_name)
    if len(values) != 1:
        raise CiftRuntimeModelExportError(f"{field_name} must contain exactly one value.")
    return _float_item(value=values[0], field_name=f"{field_name}[0]")


def _list_from_attribute(owner: object, attribute_name: str) -> list[object]:
    value = getattr(owner, attribute_name, None)
    if value is None:
        raise CiftRuntimeModelExportError(f"Object is missing attribute '{attribute_name}'.")
    return _list_from_value(value=value, field_name=attribute_name)


def _list_from_value(value: object, field_name: str) -> list[object]:
    converted = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(converted, list):
        raise CiftRuntimeModelExportError(f"{field_name} must convert to a list.")
    return cast(list[object], converted)


def _float_item(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftRuntimeModelExportError(f"{field_name} must be a number.")
    return float(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained CIFT bundle to a runtime-native JSON artifact.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-bundle-id", required=True)
    parser.add_argument("--confidence", required=True, type=float)
    parser.add_argument("--negative-action", required=True, choices=tuple(action.value for action in Action))
    parser.add_argument("--positive-action", required=True, choices=tuple(action.value for action in Action))
    parser.add_argument("--promotion-evidence", required=False)
    parser.add_argument(
        "--allow-preview-without-promotion",
        action="store_true",
        help="Export a runtime_candidate preview without promotion gates for runtime-prevention benchmark bootstrap.",
    )
    return parser


def _parse_args(argv: Sequence[str]) -> ExportCiftRuntimeModelConfig:
    namespace = _build_parser().parse_args(argv)
    promotion_evidence = namespace.promotion_evidence
    return ExportCiftRuntimeModelConfig(
        bundle_path=Path(namespace.bundle),
        output_path=Path(namespace.output),
        model_bundle_id=str(namespace.model_bundle_id),
        confidence=float(namespace.confidence),
        negative_action=Action(str(namespace.negative_action)),
        positive_action=Action(str(namespace.positive_action)),
        promotion_evidence_path=Path(promotion_evidence) if promotion_evidence is not None else None,
        allow_preview_without_promotion=bool(namespace.allow_preview_without_promotion),
    )


def main(argv: Sequence[str]) -> None:
    model = export_cift_runtime_model(_parse_args(argv))
    print(f"Exported CIFT runtime model: {model.model_bundle_id}")
    print(f"Feature key: {model.feature_key}")
    print(f"Feature count: {model.feature_count}")
