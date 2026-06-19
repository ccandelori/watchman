from __future__ import annotations

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskError
from aegis_introspection.cift import last_quarter_readout_feature_keys
from aegis_introspection.cift_meta_ablation import CiftMetaAblationVariant
from aegis_introspection.cift_meta_head import CiftMetaDecisionRule


def _variant(
    variant_id: str,
    source_feature_keys: tuple[str, ...],
    calibration_source_labels: tuple[str, ...],
    ridge: float,
    risk_label: str,
    inner_fold_count: int,
    decision_rule: CiftMetaDecisionRule,
) -> CiftMetaAblationVariant:
    if len(source_feature_keys) == 0:
        raise BinaryTaskError(f"CIFT source-ablation variant '{variant_id}' has no source features.")
    return CiftMetaAblationVariant(
        variant_id=variant_id,
        feature_name=f"cift_meta_source_ablation_{variant_id}",
        source_feature_keys=source_feature_keys,
        calibration_source_labels=calibration_source_labels,
        ridge=ridge,
        risk_label=risk_label,
        inner_fold_count=inner_fold_count,
        decision_rule=decision_rule,
    )


def _candidate_source_sets(
    final_token_keys: tuple[str, ...],
    mean_pool_keys: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if len(final_token_keys) == 0:
        raise BinaryTaskError("CIFT source ablation requires final-token source features.")
    if len(mean_pool_keys) == 0:
        raise BinaryTaskError("CIFT source ablation requires mean-pool source features.")

    candidates: list[tuple[str, tuple[str, ...]]] = [
        ("full_dual_readout", final_token_keys + mean_pool_keys),
        ("drop_last_mean_pool", final_token_keys + mean_pool_keys[:-1]),
        ("drop_last_two_mean_pool", final_token_keys + mean_pool_keys[:-2]),
        ("drop_last_final_token", final_token_keys[:-1] + mean_pool_keys),
        ("drop_last_dual_readout_layer", final_token_keys[:-1] + mean_pool_keys[:-1]),
        ("final_token_only", final_token_keys),
        ("mean_pool_only", mean_pool_keys),
    ]
    deduped_candidates: list[tuple[str, tuple[str, ...]]] = []
    seen_source_sets: set[tuple[str, ...]] = set()
    for variant_id, source_feature_keys in candidates:
        if len(source_feature_keys) == 0:
            continue
        if source_feature_keys in seen_source_sets:
            continue
        deduped_candidates.append((variant_id, source_feature_keys))
        seen_source_sets.add(source_feature_keys)
    return tuple(deduped_candidates)


def build_targeted_cift_meta_source_ablation_variants(
    artifact: ActivationArtifact,
    calibration_source_labels: tuple[str, ...],
    ridge: float,
    risk_label: str,
    inner_fold_count: int,
    decision_rule: CiftMetaDecisionRule,
) -> tuple[CiftMetaAblationVariant, ...]:
    final_token_keys = last_quarter_readout_feature_keys(artifact, "final_token")
    mean_pool_keys = last_quarter_readout_feature_keys(artifact, "mean_pool")
    return tuple(
        _variant(
            variant_id=variant_id,
            source_feature_keys=source_feature_keys,
            calibration_source_labels=calibration_source_labels,
            ridge=ridge,
            risk_label=risk_label,
            inner_fold_count=inner_fold_count,
            decision_rule=decision_rule,
        )
        for variant_id, source_feature_keys in _candidate_source_sets(
            final_token_keys=final_token_keys,
            mean_pool_keys=mean_pool_keys,
        )
    )
