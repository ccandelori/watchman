from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias, cast


JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

_FEATURE_KEY_PATTERN = re.compile(r"^(?P<pooling_method>.+)_layer_(?P<layer_index>\d+)$")


class ProbeReportSummaryError(ValueError):
    """Raised when a probe report cannot be summarized."""


@dataclass(frozen=True)
class ProbeFeatureSummary:
    feature_key: str
    pooling_method: str
    layer_index: int
    accuracy_mean: float
    macro_f1_mean: float
    accuracy_std: float
    macro_f1_std: float


@dataclass(frozen=True)
class ProbeReportSummary:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    label_names: tuple[str, ...]
    fold_count: int
    best_feature_key: str
    features: tuple[ProbeFeatureSummary, ...]


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ProbeReportSummaryError(f"Expected {description} to be a mapping.")
    return cast(Mapping[str, object], value)


def _required_string(mapping: Mapping[str, object], field_name: str) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str):
        raise ProbeReportSummaryError(f"Expected field '{field_name}' to be a string.")
    return value


def _required_int(mapping: Mapping[str, object], field_name: str) -> int:
    value = mapping.get(field_name)
    if not isinstance(value, int):
        raise ProbeReportSummaryError(f"Expected field '{field_name}' to be an integer.")
    return value


def _required_float(mapping: Mapping[str, object], field_name: str) -> float:
    value = mapping.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProbeReportSummaryError(f"Expected field '{field_name}' to be numeric.")
    return float(value)


def _required_label_names(mapping: Mapping[str, object]) -> tuple[str, ...]:
    value = mapping.get("label_names")
    if not isinstance(value, list):
        raise ProbeReportSummaryError("Expected field 'label_names' to be a list.")
    labels: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ProbeReportSummaryError(f"Expected label_names item {index} to be a string.")
        labels.append(item)
    return tuple(labels)


def parse_feature_key(feature_key: str) -> tuple[str, int]:
    match = _FEATURE_KEY_PATTERN.match(feature_key)
    if match is None:
        raise ProbeReportSummaryError(f"Feature key '{feature_key}' does not match '<pooling>_layer_<index>'.")
    pooling_method = match.group("pooling_method")
    layer_index = int(match.group("layer_index"))
    return pooling_method, layer_index


def _parse_feature_summary(value: object) -> ProbeFeatureSummary:
    mapping = _as_mapping(value, "probe feature summary")
    feature_key = _required_string(mapping, "feature_key")
    pooling_method, layer_index = parse_feature_key(feature_key)
    return ProbeFeatureSummary(
        feature_key=feature_key,
        pooling_method=pooling_method,
        layer_index=layer_index,
        accuracy_mean=_required_float(mapping, "accuracy_mean"),
        macro_f1_mean=_required_float(mapping, "macro_f1_mean"),
        accuracy_std=_required_float(mapping, "accuracy_std"),
        macro_f1_std=_required_float(mapping, "macro_f1_std"),
    )


def parse_probe_report_summary(value: object) -> ProbeReportSummary:
    mapping = _as_mapping(value, "probe report")
    features_value = mapping.get("features")
    if not isinstance(features_value, list):
        raise ProbeReportSummaryError("Expected field 'features' to be a list.")
    features = tuple(_parse_feature_summary(feature) for feature in features_value)
    if len(features) == 0:
        raise ProbeReportSummaryError("Cannot summarize a probe report with no features.")

    return ProbeReportSummary(
        source_model_id=_required_string(mapping, "source_model_id"),
        source_revision=_required_string(mapping, "source_revision"),
        source_selected_device=_required_string(mapping, "source_selected_device"),
        label_names=_required_label_names(mapping),
        fold_count=_required_int(mapping, "fold_count"),
        best_feature_key=_required_string(mapping, "best_feature_key"),
        features=features,
    )


def load_probe_report_summary(path: Path) -> ProbeReportSummary:
    with path.open("r", encoding="utf-8") as file:
        decoded = json.load(file)
    return parse_probe_report_summary(decoded)


def _feature_sort_key(feature: ProbeFeatureSummary) -> tuple[float, float, int, str]:
    return (
        feature.macro_f1_mean,
        feature.accuracy_mean,
        -feature.layer_index,
        feature.pooling_method,
    )


def sorted_features_by_score(summary: ProbeReportSummary) -> tuple[ProbeFeatureSummary, ...]:
    return tuple(sorted(summary.features, key=_feature_sort_key, reverse=True))


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def render_probe_report_markdown(summary: ProbeReportSummary) -> str:
    sorted_features = sorted_features_by_score(summary)
    best_feature = sorted_features[0]
    lines = [
        "# Probe Layer Sweep Summary",
        "",
        "## Source",
        "",
        f"- Model: `{summary.source_model_id}`",
        f"- Revision: `{summary.source_revision}`",
        f"- Extraction device: `{summary.source_selected_device}`",
        f"- Fold count: `{summary.fold_count}`",
        f"- Labels: `{', '.join(summary.label_names)}`",
        "",
        "## Best Feature",
        "",
        f"- Feature: `{best_feature.feature_key}`",
        f"- Macro F1: `{_format_float(best_feature.macro_f1_mean)}`",
        f"- Accuracy: `{_format_float(best_feature.accuracy_mean)}`",
        "",
        "## Feature Ranking",
        "",
        "| Rank | Feature | Pooling | Layer | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]

    for rank, feature in enumerate(sorted_features, start=1):
        lines.append(
            "| "
            f"{rank} | `{feature.feature_key}` | `{feature.pooling_method}` | {feature.layer_index} | "
            f"{_format_float(feature.macro_f1_mean)} | {_format_float(feature.accuracy_mean)} | "
            f"{_format_float(feature.macro_f1_std)} | {_format_float(feature.accuracy_std)} |"
        )

    lines.extend(
        [
            "",
            "## Pooling Summary",
            "",
            "| Pooling | Best Layer | Best Macro F1 | Best Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )

    pooling_methods = tuple(sorted(set(feature.pooling_method for feature in sorted_features)))
    for pooling_method in pooling_methods:
        pooling_features = tuple(feature for feature in sorted_features if feature.pooling_method == pooling_method)
        best_pooling_feature = pooling_features[0]
        lines.append(
            "| "
            f"`{pooling_method}` | {best_pooling_feature.layer_index} | "
            f"{_format_float(best_pooling_feature.macro_f1_mean)} | "
            f"{_format_float(best_pooling_feature.accuracy_mean)} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_probe_report_markdown(path: Path, summary: ProbeReportSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_probe_report_markdown(summary), encoding="utf-8")
