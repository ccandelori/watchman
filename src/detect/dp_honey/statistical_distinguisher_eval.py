"""DP-HONEY statistical distinguisher evidence.

The paper's DP-HONEY section names several empirical distinguishers for
generated honeytokens: character entropy, bigram likelihood, numeric-substring
features, format validation, and a discriminator MLP. This module evaluates
those families against held-out same-format synthetic references without
serializing raw token values.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .bigram import (
    DEFAULT_CLIP,
    DEFAULT_CORPUS_SIZE,
    DEFAULT_EPSILON,
    DEFAULT_TRAIN_SEED,
    START,
    build_model,
    segment_char_state,
    segment_start_state,
)
from .errors import DPHoneyError
from .formats import REGISTRY_VERSION, list_formats
from .grammar import FormatSpec
from .realism import LOG_PROB_FLOOR, compute_report

STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION = "detect.dp_honey.statistical_distinguisher_eval/v1"
STATISTICAL_DISTINGUISHER_EVAL_STATUS = "statistical_distinguisher_suite_evaluated"
STATISTICAL_DISTINGUISHER_EVAL_MAX_PER_FORMAT = 1000
DEFAULT_MLP_HIDDEN_UNITS = 16
DEFAULT_MLP_EPOCHS = 350
DEFAULT_MLP_LEARNING_RATE = 0.035
DEFAULT_MLP_L2 = 0.0005
MAX_CHAR_ENTROPY_MEAN_ABS_DELTA_BITS = 0.75
MAX_CHAR_ENTROPY_FORMAT_ABS_DELTA_BITS = 1.25
MAX_BIGRAM_LIKELIHOOD_MEAN_ABS_DELTA = 1.25
MAX_BIGRAM_LIKELIHOOD_FORMAT_ABS_DELTA = 3.0
MAX_DIGIT_FRACTION_FORMAT_ABS_DELTA = 0.25
MAX_NUMERIC_RUN_COUNT_FORMAT_ABS_DELTA = 2.0
MAX_NUMERIC_RUN_AVG_LENGTH_FORMAT_ABS_DELTA = 4.0
MAX_NUMERIC_RUN_P95_LENGTH_FORMAT_ABS_DELTA = 8.0
MAX_DISCRIMINATOR_TEST_BALANCED_ACCURACY = 0.68
REQUIRED_TESTS = (
    "character_entropy_tests",
    "bigram_likelihood_tests",
    "numeric_substring_tests",
    "discriminator_mlp",
)
FEATURE_NAMES = (
    "token_length",
    "digit_fraction",
    "alpha_fraction",
    "uppercase_fraction",
    "lowercase_fraction",
    "symbol_fraction",
    "single_token_entropy_bits",
    "avg_bigram_log_likelihood",
    "numeric_run_count",
    "numeric_run_avg_length",
    "numeric_run_max_length",
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class DPHoneyStatisticalDistinguisherEvalError(DPHoneyError):
    """Raised when DP-HONEY statistical distinguisher evaluation input is invalid."""


@dataclass(frozen=True)
class DPHoneyStatisticalDistinguisherEvalConfig:
    train_count_per_format: int
    test_count_per_format: int
    seed: int
    alpha: float

    def __post_init__(self) -> None:
        _validate_count(self.train_count_per_format, "train_count_per_format")
        _validate_count(self.test_count_per_format, "test_count_per_format")
        if not 0.0 < self.alpha < 1.0:
            raise DPHoneyStatisticalDistinguisherEvalError("alpha must be in (0.0, 1.0).")


@dataclass(frozen=True)
class _NumericProfile:
    digit_fraction: float
    numeric_run_count_per_token: float
    numeric_run_avg_length: float
    numeric_run_p95_length: float
    numeric_run_max_length: float


@dataclass(frozen=True)
class _FormatSamples:
    spec: FormatSpec
    model: object
    generated_train: tuple[str, ...]
    reference_train: tuple[str, ...]
    generated_test: tuple[str, ...]
    reference_test: tuple[str, ...]


def build_statistical_distinguisher_eval_report(
    config: DPHoneyStatisticalDistinguisherEvalConfig,
) -> dict[str, JsonValue]:
    specs = tuple(list_formats())
    if len(specs) == 0:
        raise DPHoneyStatisticalDistinguisherEvalError("no DP-HONEY formats are registered.")

    samples = tuple(_sample_format(spec, config) for spec in specs)
    suite = {
        "character_entropy_tests": _character_entropy_tests(samples, config),
        "bigram_likelihood_tests": _bigram_likelihood_tests(samples, config),
        "numeric_substring_tests": _numeric_substring_tests(samples, config),
        "discriminator_mlp": _discriminator_mlp_test(samples, config),
    }
    all_required_tests_passed = all(_test_passed(suite[name]) for name in REQUIRED_TESTS)
    return {
        "schema_version": STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION,
        "status": STATISTICAL_DISTINGUISHER_EVAL_STATUS,
        "registry_version": REGISTRY_VERSION,
        "seed": config.seed,
        "alpha": config.alpha,
        "train_count_per_format": config.train_count_per_format,
        "test_count_per_format": config.test_count_per_format,
        "format_count": len(samples),
        "scannable_format_count": sum(1 for sample in samples if sample.spec.scannable),
        "generator_parameters": {
            "epsilon": DEFAULT_EPSILON,
            "clip": DEFAULT_CLIP,
            "corpus_size": DEFAULT_CORPUS_SIZE,
            "train_seed": DEFAULT_TRAIN_SEED,
        },
        "reference_source": "same_format_uniform_synthetic_holdout",
        "raw_values_serialized": False,
        "required_tests": list(REQUIRED_TESTS),
        "all_required_tests_passed": all_required_tests_passed,
        "paper_faithful_statistical_distinguisher": all_required_tests_passed,
        "statistical_distinguisher_suite": suite,
        "audit_safety": {
            "raw_secret_values_in_report": False,
            "finding_payload_redacted": True,
        },
        "limits": [
            "No generated or reference token values are serialized.",
            "Reference tokens are same-format synthetic holdout examples, not provider-valid production secrets.",
            "Passing means these bounded distinguishers did not separate generated tokens from the synthetic "
            "reference beyond configured thresholds; it is not a computational indistinguishability proof.",
        ],
    }


def render_statistical_distinguisher_eval_report_json(report: dict[str, JsonValue]) -> str:
    return json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"


def write_statistical_distinguisher_eval_report(path: Path, report: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_statistical_distinguisher_eval_report_json(report), encoding="utf-8")


def _validate_count(value: int, field_name: str) -> None:
    if value < 1:
        raise DPHoneyStatisticalDistinguisherEvalError(f"{field_name} must be positive.")
    if value > STATISTICAL_DISTINGUISHER_EVAL_MAX_PER_FORMAT:
        raise DPHoneyStatisticalDistinguisherEvalError(
            f"{field_name} must be <= {STATISTICAL_DISTINGUISHER_EVAL_MAX_PER_FORMAT}."
        )


def _sample_format(spec: FormatSpec, config: DPHoneyStatisticalDistinguisherEvalConfig) -> _FormatSamples:
    slug_offset = _stable_slug_offset(spec.slug)
    model = build_model(spec)
    generated_train = tuple(model.sample(config.train_count_per_format, seed=config.seed + 10_000 + slug_offset))
    generated_test = tuple(model.sample(config.test_count_per_format, seed=config.seed + 20_000 + slug_offset))
    train_rng = np.random.default_rng(config.seed + 30_000 + slug_offset)
    test_rng = np.random.default_rng(config.seed + 40_000 + slug_offset)
    reference_train = tuple(spec.random_example(train_rng) for _ in range(config.train_count_per_format))
    reference_test = tuple(spec.random_example(test_rng) for _ in range(config.test_count_per_format))
    return _FormatSamples(
        spec=spec,
        model=model,
        generated_train=generated_train,
        reference_train=reference_train,
        generated_test=generated_test,
        reference_test=reference_test,
    )


def _character_entropy_tests(
    samples: tuple[_FormatSamples, ...],
    config: DPHoneyStatisticalDistinguisherEvalConfig,
) -> dict[str, JsonValue]:
    metrics: list[dict[str, JsonValue]] = []
    deltas: list[float] = []
    for sample in samples:
        generated_report = compute_report(list(sample.generated_test), sample.model)
        reference_report = compute_report(list(sample.reference_test), sample.model)
        generated_entropy = _finite_report_float(generated_report, "char_entropy_bits", sample.spec.slug)
        reference_entropy = _finite_report_float(reference_report, "char_entropy_bits", sample.spec.slug)
        abs_delta = abs(generated_entropy - reference_entropy)
        deltas.append(abs_delta)
        metrics.append(
            {
                "format_slug": sample.spec.slug,
                "generated_char_entropy_bits": _round_float(generated_entropy),
                "reference_char_entropy_bits": _round_float(reference_entropy),
                "abs_delta_bits": _round_float(abs_delta),
                "status": _pass_fail(abs_delta <= MAX_CHAR_ENTROPY_FORMAT_ABS_DELTA_BITS),
            }
        )

    mean_abs_delta = _mean(deltas)
    max_abs_delta = max(deltas)
    passed = (
        mean_abs_delta <= MAX_CHAR_ENTROPY_MEAN_ABS_DELTA_BITS
        and max_abs_delta <= MAX_CHAR_ENTROPY_FORMAT_ABS_DELTA_BITS
    )
    return {
        "status": _pass_fail(passed),
        "alpha": config.alpha,
        "pass_criterion": (
            "mean_abs_delta_bits <= 0.75 and every format abs_delta_bits <= 1.25 on held-out samples"
        ),
        "aggregate": {
            "mean_abs_delta_bits": _round_float(mean_abs_delta),
            "max_abs_delta_bits": _round_float(max_abs_delta),
            "threshold_mean_abs_delta_bits": MAX_CHAR_ENTROPY_MEAN_ABS_DELTA_BITS,
            "threshold_format_abs_delta_bits": MAX_CHAR_ENTROPY_FORMAT_ABS_DELTA_BITS,
        },
        "format_metrics": metrics,
    }


def _bigram_likelihood_tests(
    samples: tuple[_FormatSamples, ...],
    config: DPHoneyStatisticalDistinguisherEvalConfig,
) -> dict[str, JsonValue]:
    metrics: list[dict[str, JsonValue]] = []
    deltas: list[float] = []
    for sample in samples:
        generated_report = compute_report(list(sample.generated_test), sample.model)
        reference_report = compute_report(list(sample.reference_test), sample.model)
        generated_likelihood = _finite_report_float(generated_report, "avg_log_likelihood", sample.spec.slug)
        reference_likelihood = _finite_report_float(reference_report, "avg_log_likelihood", sample.spec.slug)
        abs_delta = abs(generated_likelihood - reference_likelihood)
        deltas.append(abs_delta)
        metrics.append(
            {
                "format_slug": sample.spec.slug,
                "generated_avg_log_likelihood": _round_float(generated_likelihood),
                "reference_avg_log_likelihood": _round_float(reference_likelihood),
                "abs_delta": _round_float(abs_delta),
                "status": _pass_fail(abs_delta <= MAX_BIGRAM_LIKELIHOOD_FORMAT_ABS_DELTA),
            }
        )

    mean_abs_delta = _mean(deltas)
    max_abs_delta = max(deltas)
    passed = (
        mean_abs_delta <= MAX_BIGRAM_LIKELIHOOD_MEAN_ABS_DELTA
        and max_abs_delta <= MAX_BIGRAM_LIKELIHOOD_FORMAT_ABS_DELTA
    )
    return {
        "status": _pass_fail(passed),
        "alpha": config.alpha,
        "pass_criterion": "mean_abs_delta <= 1.25 and every format abs_delta <= 3.0 on held-out samples",
        "aggregate": {
            "mean_abs_delta": _round_float(mean_abs_delta),
            "max_abs_delta": _round_float(max_abs_delta),
            "threshold_mean_abs_delta": MAX_BIGRAM_LIKELIHOOD_MEAN_ABS_DELTA,
            "threshold_format_abs_delta": MAX_BIGRAM_LIKELIHOOD_FORMAT_ABS_DELTA,
        },
        "format_metrics": metrics,
    }


def _numeric_substring_tests(
    samples: tuple[_FormatSamples, ...],
    config: DPHoneyStatisticalDistinguisherEvalConfig,
) -> dict[str, JsonValue]:
    metrics: list[dict[str, JsonValue]] = []
    max_digit_delta = 0.0
    max_run_count_delta = 0.0
    max_avg_run_length_delta = 0.0
    max_p95_run_length_delta = 0.0
    max_observed_run_length_delta = 0.0
    for sample in samples:
        generated_profile = _numeric_profile(sample.generated_test)
        reference_profile = _numeric_profile(sample.reference_test)
        digit_delta = abs(generated_profile.digit_fraction - reference_profile.digit_fraction)
        run_count_delta = abs(
            generated_profile.numeric_run_count_per_token - reference_profile.numeric_run_count_per_token
        )
        avg_run_length_delta = abs(generated_profile.numeric_run_avg_length - reference_profile.numeric_run_avg_length)
        p95_run_length_delta = abs(generated_profile.numeric_run_p95_length - reference_profile.numeric_run_p95_length)
        observed_run_length_delta = abs(
            generated_profile.numeric_run_max_length - reference_profile.numeric_run_max_length
        )
        max_digit_delta = max(max_digit_delta, digit_delta)
        max_run_count_delta = max(max_run_count_delta, run_count_delta)
        max_avg_run_length_delta = max(max_avg_run_length_delta, avg_run_length_delta)
        max_p95_run_length_delta = max(max_p95_run_length_delta, p95_run_length_delta)
        max_observed_run_length_delta = max(max_observed_run_length_delta, observed_run_length_delta)
        metric_passed = (
            digit_delta <= MAX_DIGIT_FRACTION_FORMAT_ABS_DELTA
            and run_count_delta <= MAX_NUMERIC_RUN_COUNT_FORMAT_ABS_DELTA
            and avg_run_length_delta <= MAX_NUMERIC_RUN_AVG_LENGTH_FORMAT_ABS_DELTA
            and p95_run_length_delta <= MAX_NUMERIC_RUN_P95_LENGTH_FORMAT_ABS_DELTA
        )
        metrics.append(
            {
                "format_slug": sample.spec.slug,
                "generated_digit_fraction": _round_float(generated_profile.digit_fraction),
                "reference_digit_fraction": _round_float(reference_profile.digit_fraction),
                "digit_fraction_abs_delta": _round_float(digit_delta),
                "numeric_run_count_per_token_abs_delta": _round_float(run_count_delta),
                "numeric_run_avg_length_abs_delta": _round_float(avg_run_length_delta),
                "numeric_run_p95_length_abs_delta": _round_float(p95_run_length_delta),
                "numeric_run_max_length_abs_delta": _round_float(observed_run_length_delta),
                "status": _pass_fail(metric_passed),
            }
        )

    passed = (
        max_digit_delta <= MAX_DIGIT_FRACTION_FORMAT_ABS_DELTA
        and max_run_count_delta <= MAX_NUMERIC_RUN_COUNT_FORMAT_ABS_DELTA
        and max_avg_run_length_delta <= MAX_NUMERIC_RUN_AVG_LENGTH_FORMAT_ABS_DELTA
        and max_p95_run_length_delta <= MAX_NUMERIC_RUN_P95_LENGTH_FORMAT_ABS_DELTA
    )
    return {
        "status": _pass_fail(passed),
        "alpha": config.alpha,
        "pass_criterion": (
            "every format digit fraction, run count, average run length, and p95 run length must remain within "
            "configured numeric thresholds"
        ),
        "aggregate": {
            "max_digit_fraction_abs_delta": _round_float(max_digit_delta),
            "max_numeric_run_count_per_token_abs_delta": _round_float(max_run_count_delta),
            "max_numeric_run_avg_length_abs_delta": _round_float(max_avg_run_length_delta),
            "max_numeric_run_p95_length_abs_delta": _round_float(max_p95_run_length_delta),
            "max_observed_numeric_run_length_abs_delta": _round_float(max_observed_run_length_delta),
            "threshold_digit_fraction_abs_delta": MAX_DIGIT_FRACTION_FORMAT_ABS_DELTA,
            "threshold_numeric_run_count_per_token_abs_delta": MAX_NUMERIC_RUN_COUNT_FORMAT_ABS_DELTA,
            "threshold_numeric_run_avg_length_abs_delta": MAX_NUMERIC_RUN_AVG_LENGTH_FORMAT_ABS_DELTA,
            "threshold_numeric_run_p95_length_abs_delta": MAX_NUMERIC_RUN_P95_LENGTH_FORMAT_ABS_DELTA,
        },
        "format_metrics": metrics,
    }


def _discriminator_mlp_test(
    samples: tuple[_FormatSamples, ...],
    config: DPHoneyStatisticalDistinguisherEvalConfig,
) -> dict[str, JsonValue]:
    train_features, train_labels = _feature_matrix(samples, split_name="train")
    test_features, test_labels = _feature_matrix(samples, split_name="test")
    model = _train_mlp_classifier(train_features, train_labels, config.seed)
    train_probabilities = _mlp_predict(model, train_features)
    test_probabilities = _mlp_predict(model, test_features)
    train_metrics = _classification_metrics(train_probabilities, train_labels)
    test_metrics = _classification_metrics(test_probabilities, test_labels)
    passed = test_metrics["balanced_accuracy"] <= MAX_DISCRIMINATOR_TEST_BALANCED_ACCURACY
    return {
        "status": _pass_fail(passed),
        "alpha": config.alpha,
        "pass_criterion": "held-out balanced accuracy <= 0.68 when distinguishing generated from reference",
        "architecture": f"K->{DEFAULT_MLP_HIDDEN_UNITS}->1",
        "feature_names": list(FEATURE_NAMES),
        "train_count": int(train_labels.size),
        "test_count": int(test_labels.size),
        "epochs": DEFAULT_MLP_EPOCHS,
        "learning_rate": DEFAULT_MLP_LEARNING_RATE,
        "threshold_max_test_balanced_accuracy": MAX_DISCRIMINATOR_TEST_BALANCED_ACCURACY,
        "aggregate": {
            "test_balanced_accuracy": _round_float(test_metrics["balanced_accuracy"]),
            "test_accuracy": _round_float(test_metrics["accuracy"]),
            "threshold_max_test_balanced_accuracy": MAX_DISCRIMINATOR_TEST_BALANCED_ACCURACY,
        },
        "train_metrics": _json_float_mapping(train_metrics),
        "test_metrics": _json_float_mapping(test_metrics),
    }


def _feature_matrix(
    samples: tuple[_FormatSamples, ...],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    for sample in samples:
        if split_name == "train":
            generated_tokens = sample.generated_train
            reference_tokens = sample.reference_train
        elif split_name == "test":
            generated_tokens = sample.generated_test
            reference_tokens = sample.reference_test
        else:
            raise DPHoneyStatisticalDistinguisherEvalError(f"unknown split_name: {split_name}")
        for token in generated_tokens:
            features.append(_token_features(token, sample.model))
            labels.append(1)
        for token in reference_tokens:
            features.append(_token_features(token, sample.model))
            labels.append(0)
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _token_features(token: str, model: object) -> list[float]:
    numeric = _numeric_profile((token,))
    length = len(token)
    digit_count = sum(1 for char in token if char.isdigit())
    alpha_count = sum(1 for char in token if char.isalpha())
    uppercase_count = sum(1 for char in token if char.isupper())
    lowercase_count = sum(1 for char in token if char.islower())
    symbol_count = length - sum(1 for char in token if char.isalnum())
    denominator = max(float(length), 1.0)
    return [
        float(length),
        digit_count / denominator,
        alpha_count / denominator,
        uppercase_count / denominator,
        lowercase_count / denominator,
        symbol_count / denominator,
        _single_token_entropy_bits(token),
        _token_avg_log_likelihood(model, token),
        numeric.numeric_run_count_per_token,
        numeric.numeric_run_avg_length,
        numeric.numeric_run_max_length,
    ]


def _train_mlp_classifier(features: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    standardized = (features - mean) / scale
    y = labels.reshape((-1, 1))
    rng = np.random.default_rng(seed + 70_000)
    w1 = rng.normal(loc=0.0, scale=0.08, size=(standardized.shape[1], DEFAULT_MLP_HIDDEN_UNITS))
    b1 = np.zeros((1, DEFAULT_MLP_HIDDEN_UNITS), dtype=np.float64)
    w2 = rng.normal(loc=0.0, scale=0.08, size=(DEFAULT_MLP_HIDDEN_UNITS, 1))
    b2 = np.zeros((1, 1), dtype=np.float64)

    for _ in range(DEFAULT_MLP_EPOCHS):
        z1 = standardized @ w1 + b1
        hidden = np.tanh(z1)
        logits = hidden @ w2 + b2
        probabilities = _sigmoid(logits)
        grad_logits = (probabilities - y) / float(y.shape[0])
        grad_w2 = hidden.T @ grad_logits + DEFAULT_MLP_L2 * w2
        grad_b2 = grad_logits.sum(axis=0, keepdims=True)
        grad_hidden = grad_logits @ w2.T
        grad_z1 = grad_hidden * (1.0 - hidden * hidden)
        grad_w1 = standardized.T @ grad_z1 + DEFAULT_MLP_L2 * w1
        grad_b1 = grad_z1.sum(axis=0, keepdims=True)
        w1 = w1 - DEFAULT_MLP_LEARNING_RATE * grad_w1
        b1 = b1 - DEFAULT_MLP_LEARNING_RATE * grad_b1
        w2 = w2 - DEFAULT_MLP_LEARNING_RATE * grad_w2
        b2 = b2 - DEFAULT_MLP_LEARNING_RATE * grad_b2

    return {"mean": mean, "scale": scale, "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _mlp_predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    hidden = np.tanh(standardized @ model["w1"] + model["b1"])
    logits = hidden @ model["w2"] + model["b2"]
    return _sigmoid(logits).reshape((-1,))


def _classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.float64)
    generated_mask = labels == 1.0
    reference_mask = labels == 0.0
    true_generated = float(np.sum((predictions == 1.0) & generated_mask))
    false_reference = float(np.sum((predictions == 0.0) & generated_mask))
    true_reference = float(np.sum((predictions == 0.0) & reference_mask))
    false_generated = float(np.sum((predictions == 1.0) & reference_mask))
    generated_count = max(float(np.sum(generated_mask)), 1.0)
    reference_count = max(float(np.sum(reference_mask)), 1.0)
    generated_recall = true_generated / generated_count
    reference_recall = true_reference / reference_count
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": (generated_recall + reference_recall) / 2.0,
        "generated_recall": generated_recall,
        "reference_recall": reference_recall,
        "reference_false_positive_rate": false_generated / reference_count,
        "generated_false_negative_rate": false_reference / generated_count,
    }


def _numeric_profile(tokens: tuple[str, ...]) -> _NumericProfile:
    digit_count = 0
    char_count = 0
    runs: list[int] = []
    for token in tokens:
        current_run = 0
        for char in token:
            char_count += 1
            if char.isdigit():
                digit_count += 1
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
    return _NumericProfile(
        digit_fraction=(digit_count / char_count) if char_count > 0 else 0.0,
        numeric_run_count_per_token=(len(runs) / len(tokens)) if len(tokens) > 0 else 0.0,
        numeric_run_avg_length=(sum(runs) / len(runs)) if len(runs) > 0 else 0.0,
        numeric_run_p95_length=_percentile(runs, 95.0),
        numeric_run_max_length=float(max(runs)) if len(runs) > 0 else 0.0,
    )


def _single_token_entropy_bits(token: str) -> float:
    if token == "":
        return 0.0
    counter: Counter[str] = Counter(token)
    entropy = 0.0
    for occurrences in counter.values():
        probability = occurrences / len(token)
        entropy -= probability * math.log2(probability)
    return entropy


def _token_avg_log_likelihood(model: object, token: str) -> float:
    spec = model.format_spec
    variables = spec.extract_variables(token)
    if variables is None:
        return 0.0
    total_log = 0.0
    total_chars = 0
    for segment_index, (chunk, segment) in enumerate(zip(variables, spec.variable_segments(), strict=True)):
        state = segment_start_state(segment_index, segment)
        fallback_state = START
        for char in chunk:
            probability = _char_probability(model, state, char, segment.alphabet, fallback_state=fallback_state)
            total_log += math.log(max(probability, LOG_PROB_FLOOR))
            total_chars += 1
            state = segment_char_state(segment_index, segment, char)
            fallback_state = char
    return (total_log / total_chars) if total_chars > 0 else 0.0


def _char_probability(model: object, state: str, char: str, alphabet: str, *, fallback_state: str | None) -> float:
    row = model.transitions.get(state, {})
    if not row and fallback_state is not None:
        row = model.transitions.get(fallback_state, {})
    masked_sum = sum(row.get(symbol, 0.0) for symbol in alphabet)
    if masked_sum > 0.0:
        return row.get(char, 0.0) / masked_sum
    return 1.0 / len(alphabet)


def _stable_slug_offset(slug: str) -> int:
    total = 0
    for index, character in enumerate(slug):
        total += (index + 1) * ord(character)
    return total


def _finite_report_float(report: dict[str, object], field_name: str, format_slug: str) -> float:
    value = report.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DPHoneyStatisticalDistinguisherEvalError(f"{format_slug}.{field_name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DPHoneyStatisticalDistinguisherEvalError(f"{format_slug}.{field_name} must be finite.")
    return numeric


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _test_passed(result: JsonValue) -> bool:
    return isinstance(result, dict) and result.get("status") == "passed"


def _pass_fail(passed: bool) -> str:
    return "passed" if passed else "failed"


def _mean(values: list[float]) -> float:
    if len(values) == 0:
        raise DPHoneyStatisticalDistinguisherEvalError("cannot compute mean over empty values.")
    return sum(values) / len(values)


def _percentile(values: list[int], percentile: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _json_float_mapping(metrics: dict[str, float]) -> dict[str, JsonValue]:
    return {key: _round_float(value) for key, value in metrics.items()}
