from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from aegis.core.contracts import Action, DetectorComponent, DetectorResult, JsonValue, PolicyDecision
from detect.dp_honey import scanner as dp_honey_scanner


@dataclass(frozen=True)
class DPHoneyAutoDecoyResult:
    """Result of explicit response-layer DP-HONEY auto-decoy remediation."""

    output_text: str
    applied: bool
    reason: str
    findings: tuple[dict[str, JsonValue], ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "output_text": self.output_text,
            "applied": self.applied,
            "reason": self.reason,
            "findings": [dict(finding) for finding in self.findings],
        }


def apply_dp_honey_auto_decoy(
    output_text: str,
    policy_decision: PolicyDecision,
    detector_results: tuple[DetectorResult, ...],
    *,
    seed: int = 0,
) -> DPHoneyAutoDecoyResult:
    """Apply DP-HONEY auto-decoy only after policy selects sanitize for DP-HONEY evidence."""
    if policy_decision.final_action != Action.SANITIZE:
        return _unchanged(output_text, "policy_action_not_sanitize")
    if not _dp_honey_triggered(policy_decision=policy_decision, detector_results=detector_results):
        return _unchanged(output_text, "dp_honey_not_triggered")

    result = dp_honey_scanner.auto_decoy(output_text, seed=seed)
    raw_findings = cast(list[dict[str, int | str]], result["findings"])
    findings = tuple(_finding_to_evidence(finding) for finding in raw_findings)
    if len(findings) == 0:
        return _unchanged(output_text, "no_findings_to_replace")
    return DPHoneyAutoDecoyResult(
        output_text=cast(str, result["swapped_text"]),
        applied=True,
        reason="dp_honey_auto_decoy_applied",
        findings=findings,
    )


def _dp_honey_triggered(policy_decision: PolicyDecision, detector_results: tuple[DetectorResult, ...]) -> bool:
    triggered = set(policy_decision.triggered_detectors)
    return any(
        result.detector_name in triggered
        and result.component == DetectorComponent.DP_HONEY
        and result.recommended_action == Action.SANITIZE
        for result in detector_results
    )


def _finding_to_evidence(finding: object) -> dict[str, JsonValue]:
    if not isinstance(finding, dict):
        return {}
    return {
        "format": str(finding["format"]),
        "confidence": str(finding["confidence"]),
        "char_start": int(finding["start"]),
        "char_end": int(finding["end"]),
    }


def _unchanged(output_text: str, reason: str) -> DPHoneyAutoDecoyResult:
    return DPHoneyAutoDecoyResult(output_text=output_text, applied=False, reason=reason, findings=())
