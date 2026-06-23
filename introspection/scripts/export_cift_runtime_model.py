from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis.core.contracts import Action
from aegis.detectors.cift_runtime import CiftRuntimeLinearModel, cift_runtime_model_to_dict
from aegis_introspection.cift_model_bundle import CiftModelBundle, load_cift_model_bundle


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained CIFT bundle to a runtime-native JSON artifact.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-bundle-id", required=True)
    parser.add_argument("--confidence", required=True, type=float)
    parser.add_argument("--negative-action", required=True, choices=tuple(action.value for action in Action))
    parser.add_argument("--positive-action", required=True, choices=tuple(action.value for action in Action))
    return parser


def _parse_args(argv: Sequence[str]) -> ExportCiftRuntimeModelConfig:
    namespace = _build_parser().parse_args(argv)
    return ExportCiftRuntimeModelConfig(
        bundle_path=Path(namespace.bundle),
        output_path=Path(namespace.output),
        model_bundle_id=str(namespace.model_bundle_id),
        confidence=float(namespace.confidence),
        negative_action=Action(str(namespace.negative_action)),
        positive_action=Action(str(namespace.positive_action)),
    )


def export_cift_runtime_model(config: ExportCiftRuntimeModelConfig) -> CiftRuntimeLinearModel:
    _validate_config(config)
    bundle = load_cift_model_bundle(config.bundle_path)
    model = _runtime_model_from_bundle(bundle=bundle, config=config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(cift_runtime_model_to_dict(model), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model


def _runtime_model_from_bundle(
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
        candidate_status=bundle.metadata.candidate_status,
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


def _validate_config(config: ExportCiftRuntimeModelConfig) -> None:
    if config.model_bundle_id == "":
        raise CiftRuntimeModelExportError("model_bundle_id must not be empty.")
    if config.confidence < 0.0 or config.confidence > 1.0:
        raise CiftRuntimeModelExportError("confidence must be in [0.0, 1.0].")


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
    if len(values) != expected_length:
        raise CiftRuntimeModelExportError(
            f"{attribute_name} has {len(values)} values, but expected {expected_length}."
        )
    return tuple(_float_item(value=value, field_name=f"{attribute_name}[{index}]") for index, value in enumerate(values))


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


def _list_from_attribute(owner: object, attribute_name: str) -> list[object]:
    value = getattr(owner, attribute_name, None)
    if value is None:
        raise CiftRuntimeModelExportError(f"Object is missing attribute '{attribute_name}'.")
    if hasattr(value, "tolist"):
        converted = value.tolist()
    else:
        converted = value
    if not isinstance(converted, list):
        raise CiftRuntimeModelExportError(f"{attribute_name} must convert to a list.")
    return cast(list[object], converted)


def _float_item(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftRuntimeModelExportError(f"{field_name} must be a number.")
    return float(value)


def main(argv: Sequence[str]) -> None:
    model = export_cift_runtime_model(_parse_args(argv))
    print(f"Exported CIFT runtime model: {model.model_bundle_id}")
    print(f"Feature key: {model.feature_key}")
    print(f"Feature count: {model.feature_count}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
