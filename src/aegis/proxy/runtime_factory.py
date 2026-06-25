from __future__ import annotations

from dataclasses import dataclass

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import CapabilityMode
from aegis.core.orchestrator import AegisRuntime, Detector, ModelProvider, TurnAnnotator
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.detectors.canary import (
    CanaryRecord,
    EncodedCanaryDetector,
    InMemoryCanaryRegistry,
    NoopCanaryDetector,
    TextCanaryDetector,
)
from aegis.detectors.egress import ProviderEgressGuardDetector
from aegis.detectors.nimbus import NimbusDetector
from aegis.policy.engine import SeverityPolicyEngine


@dataclass(frozen=True)
class ProxyCiftCapability:
    capability_mode: CapabilityMode
    turn_annotators: tuple[TurnAnnotator, ...]
    pre_generation_detectors: tuple[Detector, ...]
    detector_names: tuple[str, ...]


@dataclass(frozen=True)
class ProxyRuntimeFactory:
    audit_sink: InMemoryAuditSink
    nimbus_detector: NimbusDetector
    cift_capability: ProxyCiftCapability
    model_provider: ModelProvider

    def build(self, canary_records: tuple[CanaryRecord, ...]) -> AegisRuntime:
        return AegisRuntime(
            turn_annotators=self.cift_capability.turn_annotators,
            pre_generation_detectors=(
                *self.cift_capability.pre_generation_detectors,
                ProviderEgressGuardDetector(),
            ),
            post_generation_detectors=_post_generation_detectors(canary_records),
            session_detectors=(self.nimbus_detector,),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=self.audit_sink,
            model_provider=self.model_provider,
        )


def black_box_cift_capability() -> ProxyCiftCapability:
    return ProxyCiftCapability(
        capability_mode=CapabilityMode.BLACK_BOX,
        turn_annotators=(),
        pre_generation_detectors=(ActivationUnavailableDetector(),),
        detector_names=("activation_unavailable",),
    )


def _post_generation_detectors(canary_records: tuple[CanaryRecord, ...]) -> tuple[Detector, ...]:
    if len(canary_records) == 0:
        return (NoopCanaryDetector(),)
    registry = InMemoryCanaryRegistry(records=canary_records)
    return (
        TextCanaryDetector(detector_name="text_canary", registry=registry),
        EncodedCanaryDetector(detector_name="encoded_canary", registry=registry, partial_match_threshold=0.75),
    )
