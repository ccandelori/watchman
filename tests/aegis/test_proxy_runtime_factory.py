from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import cast

from aegis.audit.memory import InMemoryAuditSink
from aegis.cift_contract import (
    CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)
from aegis.core.contracts import Action, CapabilityMode, Message, ModelInfo, NormalizedTurn
from aegis.core.orchestrator import RuntimeRequest
from aegis.detectors.canary import CanaryRecord, canary_sha256
from aegis.detectors.cift_runtime import (
    CiftRuntimeLinearModel,
    CiftRuntimeMlpModel,
    cift_runtime_model_to_dict,
    load_cift_runtime_model,
)
from aegis.detectors.nimbus import (
    BaselineNimbusCritic,
    InMemoryNimbusStateStore,
    NimbusConfig,
    NimbusDetector,
    NimbusToolEgressDetector,
)
from aegis.providers.mock import MockModelProvider
from aegis.proxy.config import CiftCertificationMode, CiftProfile, ProxyCiftConfig, ProxyConfigError
from aegis.proxy.runtime_factory import ProxyRuntimeFactory, black_box_cift_capability, cift_capability_from_config

_IMMUTABLE_MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"


def test_proxy_runtime_factory_builds_canary_aware_runtime() -> None:
    canary = CanaryRecord(
        canary_id="canary-test-1",
        credential_type="github_pat",
        value="canary_test_value_123",
        sha256=canary_sha256("canary_test_value_123"),
        source="test",
        metadata={"slot_name": "repo_pat", "turn_planted": 0},
    )
    runtime = ProxyRuntimeFactory(
        audit_sink=InMemoryAuditSink(),
        nimbus_detector=_nimbus_detector(),
        nimbus_tool_egress_detector=_nimbus_tool_egress_detector(),
        cift_capability=black_box_cift_capability(),
        model_provider=MockModelProvider(default_content=canary.value),
    ).build(canary_records=(canary,))

    response = runtime.evaluate_turn(_runtime_request())
    detector_names = tuple(result.detector_name for result in response.detector_results)

    assert detector_names == (
        "activation_unavailable",
        "tool_call_canary",
        "nimbus_tool_egress",
        "provider_egress_guard",
        "text_canary",
        "encoded_canary",
        "nimbus",
    )
    assert response.policy_decision.final_action == Action.ESCALATE


def test_cift_capability_from_config_defaults_to_black_box() -> None:
    capability = cift_capability_from_config(
        config=ProxyCiftConfig(
            profile=CiftProfile.BLACK_BOX,
            certification_mode=CiftCertificationMode.STRICT,
            detector_name="cift_runtime",
            selected_choice_model_path=None,
            fallback_model_path=None,
            certification_manifest_path=None,
            certification_report_path=None,
            certification_artifact_root=None,
            certification_manifest_sha256=None,
            certification_report_sha256=None,
            release_gate_report_path=None,
            release_gate_report_sha256=None,
            required_device=None,
            selected_choice_readout_token_count=None,
            extractor_id=None,
            extractor_base_url=None,
            extractor_api_key=None,
            extractor_timeout_seconds=None,
            feature_source="",
        ),
        extractors={},
    )

    assert capability.capability_mode == CapabilityMode.BLACK_BOX
    assert capability.detector_names == ("activation_unavailable",)


def test_cift_capability_from_config_requires_registered_extractor() -> None:
    config = ProxyCiftConfig(
        profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
        certification_mode=CiftCertificationMode.STRICT,
        detector_name="cift_runtime",
        selected_choice_model_path=Path("selected.json"),
        fallback_model_path=Path("fallback.json"),
        certification_manifest_path=Path("certification.json"),
        certification_report_path=Path("certification-run.json"),
        certification_artifact_root=Path("."),
        certification_manifest_sha256="0" * 64,
        certification_report_sha256="1" * 64,
        release_gate_report_path=Path("release-gate.json"),
        release_gate_report_sha256="2" * 64,
        required_device="mps",
        selected_choice_readout_token_count=4,
        extractor_id="trusted-activation-sidecar",
        extractor_base_url=None,
        extractor_api_key=None,
        extractor_timeout_seconds=None,
        feature_source="self_hosted_activation_extractor",
    )

    try:
        cift_capability_from_config(config=config, extractors={})
    except ProxyConfigError as exc:
        assert "trusted-activation-sidecar" in str(exc)
    else:
        raise AssertionError("missing extractor should fail closed.")


def test_cift_capability_from_config_builds_self_hosted_window_selector() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        selected_model_sha256 = _sha256_file(selected_model_path)
        release_gate_report_sha256 = _sha256_file(_release_gate_report_path(certification_report_path))

        capability = cift_capability_from_config(
            config=ProxyCiftConfig(
                profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                certification_mode=CiftCertificationMode.STRICT,
                detector_name="cift_runtime",
                selected_choice_model_path=selected_model_path,
                fallback_model_path=None,
                certification_manifest_path=certification_manifest_path,
                certification_report_path=certification_report_path,
                certification_artifact_root=root,
                certification_manifest_sha256=_sha256_file(certification_manifest_path),
                certification_report_sha256=_sha256_file(certification_report_path),
                release_gate_report_path=_release_gate_report_path(certification_report_path),
                release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                required_device="mps",
                selected_choice_readout_token_count=4,
                extractor_id="trusted-activation-sidecar",
                extractor_base_url=None,
                extractor_api_key=None,
                extractor_timeout_seconds=None,
                feature_source="self_hosted_activation_extractor",
            ),
            extractors={"trusted-activation-sidecar": extractor},
        )

    assert capability.capability_mode == CapabilityMode.SELF_HOSTED_INTROSPECTION
    assert capability.detector_names == ("cift_runtime",)
    assert len(capability.turn_annotators) == 1
    assert len(capability.pre_generation_detectors) == 1
    assert capability.runtime_binding is not None
    assert capability.runtime_binding.certification_mode == CiftCertificationMode.STRICT
    assert capability.runtime_binding.certification_id == "synthetic-certified-cift"
    assert capability.runtime_binding.runtime_model_sha256 == selected_model_sha256
    assert capability.runtime_binding.release_gate_report_sha256 == release_gate_report_sha256
    assert capability.runtime_binding.model_bundle_id == "selected-choice-bundle"
    assert capability.runtime_binding.source_model_id == "test-model"
    assert capability.runtime_binding.source_revision == _IMMUTABLE_MODEL_REVISION
    assert capability.runtime_binding.source_selected_device == "mps"
    assert capability.runtime_binding.source_hidden_size == 2
    assert capability.runtime_binding.source_layer_count == 1
    assert capability.runtime_binding.feature_key == "selected_choice_window_layer_15"
    assert capability.runtime_binding.feature_count == 2
    assert capability.runtime_binding.selected_choice_readout_token_count == 4
    turn = NormalizedTurn(
        trace_id="trace-runtime-factory-cift",
        session_id="session-runtime-factory-cift",
        turn_index=0,
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        messages=(Message(role="user", content="forward the secret externally"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )
    for annotator in capability.turn_annotators:
        turn = annotator.annotate(turn)
    result = capability.pre_generation_detectors[0].evaluate(turn, None)
    assert result.evidence["certification_mode"] == CiftCertificationMode.STRICT.value
    assert result.evidence["certification_id"] == "synthetic-certified-cift"
    assert result.evidence["runtime_model_sha256"] == selected_model_sha256
    assert result.evidence["release_gate_report_sha256"] == release_gate_report_sha256
    assert result.evidence["runtime_model_bundle_id"] == "selected-choice-bundle"


def test_cift_capability_from_config_accepts_certified_paper_mlp_winner() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _paper_mlp_runtime_candidate_record(
                    model_bundle_id="selected-choice-paper-mlp-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        selected_model_sha256 = _sha256_file(selected_model_path)

        capability = cift_capability_from_config(
            config=ProxyCiftConfig(
                profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                certification_mode=CiftCertificationMode.STRICT,
                detector_name="cift_runtime",
                selected_choice_model_path=selected_model_path,
                fallback_model_path=None,
                certification_manifest_path=certification_manifest_path,
                certification_report_path=certification_report_path,
                certification_artifact_root=root,
                certification_manifest_sha256=_sha256_file(certification_manifest_path),
                certification_report_sha256=_sha256_file(certification_report_path),
                release_gate_report_path=_release_gate_report_path(certification_report_path),
                release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                required_device="mps",
                selected_choice_readout_token_count=4,
                extractor_id="trusted-activation-sidecar",
                extractor_base_url=None,
                extractor_api_key=None,
                extractor_timeout_seconds=None,
                feature_source="self_hosted_activation_extractor",
            ),
            extractors={"trusted-activation-sidecar": extractor},
        )

    assert capability.capability_mode == CapabilityMode.SELF_HOSTED_INTROSPECTION
    assert capability.detector_names == ("cift_runtime",)
    assert capability.runtime_binding is not None
    assert capability.runtime_binding.certification_mode == CiftCertificationMode.STRICT
    assert capability.runtime_binding.runtime_model_sha256 == selected_model_sha256
    assert capability.runtime_binding.model_bundle_id == "selected-choice-paper-mlp-bundle"


def test_cift_capability_from_config_rejects_stale_pinned_release_gate_report_sha256() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=certification_report_path,
                    certification_artifact_root=root,
                    certification_manifest_sha256=_sha256_file(certification_manifest_path),
                    certification_report_sha256=_sha256_file(certification_report_path),
                    release_gate_report_path=_release_gate_report_path(certification_report_path),
                    release_gate_report_sha256="0" * 64,
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "release_gate_report_path sha256" in str(exc)
        else:
            raise AssertionError("stale release-gate report SHA should fail closed.")


def test_cift_capability_from_config_rejects_ineligible_release_gate_report() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        release_gate_report_path = _release_gate_report_path(certification_report_path)
        release_gate_report = _read_json_object(release_gate_report_path)
        release_gate_report["production_release_eligible"] = False
        _write_json(release_gate_report_path, release_gate_report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="release_gate_report.production_release_eligible",
        )


def test_cift_capability_from_config_rejects_release_gate_artifact_root_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        release_gate_report_path = _release_gate_report_path(certification_report_path)
        release_gate_report = _read_json_object(release_gate_report_path)
        certification_binding = cast(dict[str, object], release_gate_report["certification_binding"])
        certification_binding["certification_artifact_root"] = str(root / "wrong-root")
        _write_json(release_gate_report_path, release_gate_report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="release_gate_report.certification_binding.certification_artifact_root",
        )


def test_cift_capability_from_config_rejects_runtime_candidate_without_certification_scope() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        runtime_record = _runtime_candidate_record(
            model_bundle_id="selected-choice-bundle",
            feature_key="selected_choice_window_layer_15",
        )
        promotion_gates = cast(dict[str, object], runtime_record["promotion_gates"])
        runtime_candidate = cast(dict[str, object], promotion_gates["runtime_candidate"])
        del runtime_candidate["requires_certification_binding"]
        selected_model_path.write_text(json.dumps(runtime_record), encoding="utf-8")
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="runtime_model.promotion_gates.runtime_candidate.requires_certification_binding",
        )


def test_cift_capability_from_config_rejects_strict_fallback_model_path() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        fallback_model_path = root / "fallback.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        fallback_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="fallback-bundle",
                    feature_key="readout_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=fallback_model_path,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=certification_report_path,
                    certification_artifact_root=root,
                    certification_manifest_sha256=_sha256_file(certification_manifest_path),
                    certification_report_sha256=_sha256_file(certification_report_path),
                    release_gate_report_path=_release_gate_report_path(certification_report_path),
                    release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "fallback_model_path" in str(exc)
        else:
            raise AssertionError("strict CIFT should reject fallback model paths.")


def test_cift_capability_from_config_rejects_untrusted_strict_feature_source() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="expected_feature_source",
            feature_source="offline_replay",
        )


def test_cift_capability_from_config_requires_certified_runtime_binding() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=root / "missing-certification.json",
                    certification_report_path=root / "missing-certification-run.json",
                    certification_artifact_root=root,
                    certification_manifest_sha256="0" * 64,
                    certification_report_sha256="1" * 64,
                    release_gate_report_path=root / "release-gate.json",
                    release_gate_report_sha256="2" * 64,
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "certification_manifest_path" in str(exc)
        else:
            raise AssertionError("uncertified CIFT runtime should fail closed.")


def test_cift_capability_from_config_builds_gateway_smoke_bootstrap_from_preview_candidate() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        selected_model_path.write_text(
            json.dumps(
                _offline_preview_record(
                    model_bundle_id="selected-choice-preview-bundle",
                    feature_key="selected_choice_window_layer_15",
                    source_model_id="Qwen/Qwen3-4B",
                    source_selected_device="mps",
                )
            ),
            encoding="utf-8",
        )
        expected_runtime_sha256 = hashlib.sha256(selected_model_path.read_bytes()).hexdigest()

        capability = cift_capability_from_config(
            config=_gateway_smoke_bootstrap_config(
                selected_model_path=selected_model_path,
                fallback_model_path=None,
                required_device="mps",
            ),
            extractors={"trusted-activation-sidecar": extractor},
        )

    assert capability.capability_mode == CapabilityMode.SELF_HOSTED_INTROSPECTION
    assert capability.detector_names == ("cift_runtime",)
    assert len(capability.turn_annotators) == 1
    assert len(capability.pre_generation_detectors) == 1
    assert capability.runtime_binding is not None
    assert capability.runtime_binding.certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP
    assert capability.runtime_binding.certification_id is None
    assert capability.runtime_binding.release_gate_report_sha256 is None
    assert capability.runtime_binding.runtime_model_sha256 == expected_runtime_sha256
    assert capability.runtime_binding.model_bundle_id == "selected-choice-preview-bundle"
    assert capability.runtime_binding.source_model_id == "Qwen/Qwen3-4B"
    assert capability.runtime_binding.source_selected_device == "mps"
    assert capability.runtime_binding.selected_choice_readout_token_count == 4


def test_cift_capability_from_config_rejects_gateway_smoke_bootstrap_mutable_revision() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        runtime_record = _offline_preview_record(
            model_bundle_id="selected-choice-preview-bundle",
            feature_key="selected_choice_window_layer_15",
            source_model_id="Qwen/Qwen3-4B",
            source_selected_device="mps",
        )
        runtime_record["source_revision"] = "main"
        selected_model_path.write_text(json.dumps(runtime_record), encoding="utf-8")

        try:
            cift_capability_from_config(
                config=_gateway_smoke_bootstrap_config(
                    selected_model_path=selected_model_path,
                    fallback_model_path=None,
                    required_device="mps",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "source_revision must be an immutable model revision" in str(exc)
        else:
            raise AssertionError("gateway smoke bootstrap must reject mutable model revisions.")


def test_cift_capability_from_config_rejects_gateway_smoke_bootstrap_device_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        selected_model_path.write_text(
            json.dumps(
                _offline_preview_record(
                    model_bundle_id="selected-choice-preview-bundle",
                    feature_key="selected_choice_window_layer_15",
                    source_model_id="Qwen/Qwen3-4B",
                    source_selected_device="cpu",
                )
            ),
            encoding="utf-8",
        )

        try:
            cift_capability_from_config(
                config=_gateway_smoke_bootstrap_config(
                    selected_model_path=selected_model_path,
                    fallback_model_path=None,
                    required_device="mps",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "source_selected_device must match required_device" in str(exc)
        else:
            raise AssertionError("gateway smoke bootstrap must reject device-mismatched artifacts.")


def test_cift_capability_from_config_rejects_gateway_smoke_bootstrap_non_blocking_positive_action() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        runtime_record = _offline_preview_record(
            model_bundle_id="selected-choice-preview-bundle",
            feature_key="selected_choice_window_layer_15",
            source_model_id="Qwen/Qwen3-4B",
            source_selected_device="mps",
        )
        runtime_record["positive_action"] = Action.ALLOW.value
        selected_model_path.write_text(json.dumps(runtime_record), encoding="utf-8")

        try:
            cift_capability_from_config(
                config=_gateway_smoke_bootstrap_config(
                    selected_model_path=selected_model_path,
                    fallback_model_path=None,
                    required_device="mps",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "positive_action must be block or escalate" in str(exc)
        else:
            raise AssertionError("gateway smoke bootstrap must reject non-blocking positive actions.")


def test_cift_capability_from_config_rejects_certification_device_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="cpu",
        )

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=certification_report_path,
                    certification_artifact_root=root,
                    certification_manifest_sha256=_sha256_file(certification_manifest_path),
                    certification_report_sha256=_sha256_file(certification_report_path),
                    release_gate_report_path=_release_gate_report_path(certification_report_path),
                    release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "requested_device" in str(exc)
        else:
            raise AssertionError("CIFT runtime certified for a different device should fail closed.")


def test_cift_capability_from_config_rejects_qwen3_4b_cpu_certification() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                    source_model_id="Qwen/Qwen3-4B",
                    source_selected_device="cpu",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="cpu",
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="Qwen/Qwen3-4B certification requires required_device mps",
            required_device="cpu",
        )


def test_cift_capability_from_config_rejects_mutable_model_revision() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        runtime_record = _runtime_candidate_record(
            model_bundle_id="selected-choice-bundle",
            feature_key="selected_choice_window_layer_15",
        )
        runtime_record["source_revision"] = "main"
        selected_model_path.write_text(json.dumps(runtime_record), encoding="utf-8")
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="runtime_model.source_revision must be an immutable",
        )


def test_cift_capability_from_config_requires_certified_selected_choice_contract() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        training = cast(dict[str, object], manifest["training"])
        del training["selected_choice_readout_token_count"]
        _write_json(certification_manifest_path, manifest)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="selected_choice_readout_token_count",
        )


def test_cift_capability_from_config_rejects_runtime_artifact_drift_after_certification() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="tampered-selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=certification_report_path,
                    certification_artifact_root=root,
                    certification_manifest_sha256=_sha256_file(certification_manifest_path),
                    certification_report_sha256=_sha256_file(certification_report_path),
                    release_gate_report_path=_release_gate_report_path(certification_report_path),
                    release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "sha256" in str(exc)
        else:
            raise AssertionError("CIFT runtime drift after certification should fail closed.")


def test_cift_capability_from_config_requires_certification_report_path() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=None,
                    certification_artifact_root=root,
                    certification_manifest_sha256="0" * 64,
                    certification_report_sha256="1" * 64,
                    release_gate_report_path=root / "release-gate.json",
                    release_gate_report_sha256="2" * 64,
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "certification_report_path" in str(exc)
        else:
            raise AssertionError("missing certification report path should fail closed.")


def test_cift_capability_from_config_requires_certification_artifact_root() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=certification_report_path,
                    certification_artifact_root=None,
                    certification_manifest_sha256="0" * 64,
                    certification_report_sha256="1" * 64,
                    release_gate_report_path=root / "release-gate.json",
                    release_gate_report_sha256="2" * 64,
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "certification_artifact_root" in str(exc)
        else:
            raise AssertionError("missing certification artifact root should fail closed.")


def test_cift_capability_from_config_rejects_ineligible_certification_report() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        report["certification_eligible"] = False
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="certification_eligible",
        )


def test_cift_capability_from_config_rejects_dry_run_certification_report() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        report["mode"] = "dry_run"
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="mode",
        )


def test_cift_capability_from_config_rejects_failed_certification_report() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        report["failed_requirements"] = ["runtime prevention evidence is missing"]
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="failed_requirements",
        )


def test_cift_capability_from_config_rejects_unbounded_certification_report() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        del report["command_timeout_seconds"]
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="command_timeout_seconds",
        )


def test_cift_capability_from_config_rejects_nonpositive_certification_report_timeout() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        report["command_timeout_seconds"] = 0.0
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="command_timeout_seconds",
        )


def test_cift_capability_from_config_rejects_certification_report_sha_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        artifact = _single_test_artifact(report, "promoted_runtime")
        artifact["actual_sha256"] = "f" * 64
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="actual_sha256",
        )


def test_cift_capability_from_config_rejects_stale_pinned_certification_report_sha256() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        stale_report_sha256 = _sha256_file(certification_report_path)
        report = _read_json_object(certification_report_path)
        report["created_at"] = "2026-06-26T00:00:00Z"
        _write_json(certification_report_path, report)

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=certification_report_path,
                    certification_artifact_root=root,
                    certification_manifest_sha256=_sha256_file(certification_manifest_path),
                    certification_report_sha256=stale_report_sha256,
                    release_gate_report_path=_release_gate_report_path(certification_report_path),
                    release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "certification_report_path sha256" in str(exc)
        else:
            raise AssertionError("stale pinned certification report digest should fail closed.")


def test_cift_capability_from_config_rejects_stale_pinned_certification_manifest_sha256() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        stale_manifest_sha256 = _sha256_file(certification_manifest_path)
        manifest = _read_json_object(certification_manifest_path)
        manifest["created_at"] = "2026-06-26T00:00:00Z"
        _write_json(certification_manifest_path, manifest)

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=certification_report_path,
                    certification_artifact_root=root,
                    certification_manifest_sha256=stale_manifest_sha256,
                    certification_report_sha256=_sha256_file(certification_report_path),
                    release_gate_report_path=_release_gate_report_path(certification_report_path),
                    release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "certification_manifest_path sha256" in str(exc)
        else:
            raise AssertionError("stale pinned certification manifest digest should fail closed.")


def test_cift_capability_from_config_rejects_duplicate_promoted_runtime_certification_artifacts() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        artifacts = _test_artifacts(report)
        artifacts.append(dict(_single_test_artifact(report, "promoted_runtime")))
        report["artifacts"] = artifacts
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="exactly one artifact",
        )


def test_cift_capability_from_config_rejects_certification_without_gateway_smoke_artifact() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        manifest["required_evidence_artifacts"] = [
            artifact
            for artifact in _required_evidence_artifacts(manifest)
            if artifact.get("role") != "linear_gateway_smoke"
        ]
        _write_json(certification_manifest_path, manifest)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="linear_gateway_smoke",
        )


def test_cift_capability_from_config_rejects_unverified_evidence_chain_artifact() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        artifact = _single_test_artifact(report, "evidence_chain_verification")
        artifact["actual_status"] = "missing"
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="evidence_chain_verification.actual_status",
        )


def test_cift_capability_from_config_rejects_report_missing_evidence_chain_artifact() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        report["artifacts"] = [
            artifact for artifact in _test_artifacts(report) if artifact.get("role") != "evidence_chain_verification"
        ]
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="evidence_chain_verification",
        )


def test_cift_capability_from_config_rejects_certification_artifact_path_outside_root() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "linear_gateway_smoke")
        outside_gateway_path = str(root.parent / "outside-gateway-smoke.json")
        artifact["path"] = outside_gateway_path
        _write_json(certification_manifest_path, manifest)
        report = _read_json_object(certification_report_path)
        report_artifact = _single_test_artifact(report, "linear_gateway_smoke")
        report_artifact["path"] = outside_gateway_path
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="certification artifact path must resolve under artifact root",
        )


def test_cift_capability_from_config_rejects_missing_gateway_smoke_file() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "linear_gateway_smoke")
        artifact_path = Path(str(artifact["path"]))
        artifact_path.unlink()

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="certification manifest.linear_gateway_smoke.path does not exist",
        )


def test_cift_capability_from_config_rejects_gateway_smoke_without_readiness() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        gateway_artifact = _single_required_evidence_artifact(manifest, "linear_gateway_smoke")
        gateway_path = Path(str(gateway_artifact["path"]))
        gateway_smoke = _read_json_object(gateway_path)
        checks = cast(dict[str, object], gateway_smoke["checks"])
        del checks["gateway_readiness"]
        _write_json(gateway_path, gateway_smoke)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="linear_gateway_smoke",
            sha256=_sha256_file(gateway_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="gateway_smoke.gateway_readiness",
        )


def test_cift_capability_from_config_rejects_missing_non_smoke_certification_artifact_file() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "calibration")
        artifact_path = Path(str(artifact["path"]))
        artifact_path.unlink()

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="certification manifest.calibration.path does not exist",
        )


def test_cift_capability_from_config_rejects_non_smoke_certification_artifact_sha_drift() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "calibration")
        artifact_path = Path(str(artifact["path"]))
        calibration = _read_json_object(artifact_path)
        calibration["status"] = "drifted"
        _write_json(artifact_path, calibration)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="certification manifest.calibration.path sha256 must match certification manifest",
        )


def test_cift_capability_from_config_rejects_offline_runtime_prevention_evidence() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "linear_live_runtime_prevention")
        artifact_path = Path(str(artifact["path"]))
        runtime_prevention = _read_json_object(artifact_path)
        runtime_prevention["benchmark_mode"] = "offline_replay"
        _write_json(artifact_path, runtime_prevention)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="linear_live_runtime_prevention",
            sha256=_sha256_file(artifact_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="linear_live_runtime_prevention.benchmark_mode",
        )


def test_cift_capability_from_config_rejects_runtime_prevention_row_without_selected_choice_proof() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "linear_live_runtime_prevention")
        artifact_path = Path(str(artifact["path"]))
        runtime_prevention = _read_json_object(artifact_path)
        rows = runtime_prevention["rows"]
        if not isinstance(rows, list) or not isinstance(rows[0], dict):
            raise AssertionError("runtime-prevention rows must contain objects.")
        rows[0]["window_family"] = "fallback"
        _write_json(artifact_path, runtime_prevention)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="linear_live_runtime_prevention",
            sha256=_sha256_file(artifact_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="linear_live_runtime_prevention.rows.window_family",
        )


def test_cift_capability_from_config_rejects_runtime_prevention_receipt_digest_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "linear_live_runtime_prevention")
        artifact_path = Path(str(artifact["path"]))
        runtime_prevention = _read_json_object(artifact_path)
        rows = runtime_prevention["rows"]
        if not isinstance(rows, list) or not isinstance(rows[0], dict):
            raise AssertionError("runtime-prevention rows must contain objects.")
        rows[0]["extractor_selected_choice_readout_token_indices_sha256"] = "b" * 64
        _write_json(artifact_path, runtime_prevention)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="linear_live_runtime_prevention",
            sha256=_sha256_file(artifact_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="extractor_selected_choice_readout_token_indices_sha256",
        )


def test_cift_capability_from_config_rejects_sealed_holdout_device_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "linear_sealed_holdout_metric")
        artifact_path = Path(str(artifact["path"]))
        sealed_holdout = _read_json_object(artifact_path)
        sealed_holdout["source_selected_device"] = "cpu"
        _write_json(artifact_path, sealed_holdout)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="linear_sealed_holdout_metric",
            sha256=_sha256_file(artifact_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="linear_sealed_holdout_metric.source_selected_device",
        )


def test_cift_capability_from_config_rejects_failed_live_linear_vs_paper_mlp_evidence() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "live_sealed_linear_vs_paper_mlp")
        artifact_path = Path(str(artifact["path"]))
        head_to_head = _read_json_object(artifact_path)
        head_to_head["candidate_strictly_outperforms_paper"] = False
        _write_json(artifact_path, head_to_head)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="live_sealed_linear_vs_paper_mlp",
            sha256=_sha256_file(artifact_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="live_sealed_linear_vs_paper_mlp.candidate_strictly_outperforms_paper",
        )


def test_cift_capability_from_config_rejects_grouped_cv_without_false_negative_rate() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "grouped_cv_linear_vs_paper_mlp")
        artifact_path = Path(str(artifact["path"]))
        grouped_cv = _read_json_object(artifact_path)
        candidate_probe = grouped_cv["candidate_probe"]
        if not isinstance(candidate_probe, dict):
            raise AssertionError("candidate_probe must be an object.")
        del candidate_probe["false_negative_rate"]
        _write_json(artifact_path, grouped_cv)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="grouped_cv_linear_vs_paper_mlp",
            sha256=_sha256_file(artifact_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="grouped_cv_linear_vs_paper_mlp.candidate_probe.false_negative_rate",
        )


def test_cift_capability_from_config_rejects_gateway_smoke_detector_name_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )

        try:
            cift_capability_from_config(
                config=ProxyCiftConfig(
                    profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                    certification_mode=CiftCertificationMode.STRICT,
                    detector_name="renamed_cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    fallback_model_path=None,
                    certification_manifest_path=certification_manifest_path,
                    certification_report_path=certification_report_path,
                    certification_artifact_root=root,
                    certification_manifest_sha256=_sha256_file(certification_manifest_path),
                    certification_report_sha256=_sha256_file(certification_report_path),
                    release_gate_report_path=_release_gate_report_path(certification_report_path),
                    release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                    required_device="mps",
                    selected_choice_readout_token_count=4,
                    extractor_id="trusted-activation-sidecar",
                    extractor_base_url=None,
                    extractor_api_key=None,
                    extractor_timeout_seconds=None,
                    feature_source="self_hosted_activation_extractor",
                ),
                extractors={"trusted-activation-sidecar": extractor},
            )
        except ProxyConfigError as exc:
            assert "gateway_smoke.detector_name" in str(exc)
        else:
            raise AssertionError("detector name must be bound to certified CIFT smoke evidence.")


def test_cift_capability_from_config_rejects_gateway_smoke_extractor_identity_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        gateway_artifact = _single_required_evidence_artifact(manifest, "linear_gateway_smoke")
        gateway_path = Path(str(gateway_artifact["path"]))
        gateway_smoke = _read_json_object(gateway_path)
        expected = cast(dict[str, object], gateway_smoke["expected"])
        expected["extractor_id"] = "untrusted-sidecar"
        _write_json(gateway_path, gateway_smoke)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="linear_gateway_smoke",
            sha256=_sha256_file(gateway_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="gateway_smoke.expected.extractor_id",
        )


def test_cift_capability_from_config_rejects_gateway_smoke_tokenizer_fingerprint_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        gateway_artifact = _single_required_evidence_artifact(manifest, "linear_gateway_smoke")
        gateway_path = Path(str(gateway_artifact["path"]))
        gateway_smoke = _read_json_object(gateway_path)
        expected = cast(dict[str, object], gateway_smoke["expected"])
        expected["sidecar_tokenizer_fingerprint_sha256"] = "e" * 64
        _write_json(gateway_path, gateway_smoke)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="linear_gateway_smoke",
            sha256=_sha256_file(gateway_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="gateway_smoke.expected.sidecar_tokenizer_fingerprint_sha256",
        )


def test_cift_capability_from_config_rejects_evidence_chain_device_mismatch() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        evidence_artifact = _single_required_evidence_artifact(manifest, "evidence_chain_verification")
        evidence_path = Path(str(evidence_artifact["path"]))
        evidence_chain = _read_json_object(evidence_path)
        evidence_chain["required_runtime_prevention_device"] = "cpu"
        _write_json(evidence_path, evidence_chain)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="evidence_chain_verification",
            sha256=_sha256_file(evidence_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="evidence_chain_verification.required_runtime_prevention_device",
        )


def test_cift_capability_from_config_rejects_planned_promotion_evidence_artifact() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        artifact = _single_required_evidence_artifact(manifest, "promotion_evidence")
        artifact["status"] = "planned"
        artifact["sha256"] = None
        _write_json(certification_manifest_path, manifest)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="promotion_evidence.status",
        )


def test_cift_capability_from_config_rejects_promotion_evidence_report_artifact_drift() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        manifest = _read_json_object(certification_manifest_path)
        promotion_artifact = _single_required_evidence_artifact(manifest, "promotion_evidence")
        promotion_path = Path(str(promotion_artifact["path"]))
        stale_gateway_path = root / "stale_gateway_smoke.json"
        _write_json(
            stale_gateway_path,
            {
                "schema_version": "aegis.proxy.cift_gateway_smoke/v1",
                "report_id": "synthetic-gateway-smoke-report",
            },
        )
        promotion_evidence = _read_json_object(promotion_path)
        raw_report_artifacts = promotion_evidence.get("report_artifacts")
        if not isinstance(raw_report_artifacts, list):
            raise AssertionError("Expected promotion evidence report_artifacts list.")
        for raw_artifact in raw_report_artifacts:
            if not isinstance(raw_artifact, dict):
                raise AssertionError("Expected promotion evidence report artifact object.")
            artifact = cast(dict[str, object], raw_artifact)
            if artifact["report_id"] == "synthetic-gateway-smoke-report":
                artifact["path"] = str(stale_gateway_path)
                artifact["sha256"] = _sha256_file(stale_gateway_path)
        _write_json(promotion_path, promotion_evidence)
        _replace_artifact_sha256(
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            role="promotion_evidence",
            sha256=_sha256_file(promotion_path),
        )

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="promotion_evidence.report_artifacts[synthetic-gateway-smoke-report].path",
        )


def test_cift_capability_from_config_rejects_sealed_holdout_sha_mismatch_between_manifest_and_report() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )
        report = _read_json_object(certification_report_path)
        artifact = _single_test_artifact(report, "linear_sealed_holdout_metric")
        artifact["actual_sha256"] = "f" * 64
        _write_json(certification_report_path, report)

        _assert_self_hosted_cift_rejected(
            extractor=extractor,
            selected_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            expected_message="linear_sealed_holdout_metric.actual_sha256",
        )


def test_cift_capability_from_config_builds_self_hosted_window_selector_without_fallback() -> None:
    extractor = StaticFeatureExtractor()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selected_model_path = root / "selected.json"
        certification_manifest_path = root / "certification.json"
        certification_report_path = root / "certification-run.json"
        selected_model_path.write_text(
            json.dumps(
                _runtime_candidate_record(
                    model_bundle_id="selected-choice-bundle",
                    feature_key="selected_choice_window_layer_15",
                )
            ),
            encoding="utf-8",
        )
        _write_certification_binding(
            runtime_model_path=selected_model_path,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device="mps",
        )

        capability = cift_capability_from_config(
            config=ProxyCiftConfig(
                profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                certification_mode=CiftCertificationMode.STRICT,
                detector_name="cift_runtime",
                selected_choice_model_path=selected_model_path,
                fallback_model_path=None,
                certification_manifest_path=certification_manifest_path,
                certification_report_path=certification_report_path,
                certification_artifact_root=root,
                certification_manifest_sha256=_sha256_file(certification_manifest_path),
                certification_report_sha256=_sha256_file(certification_report_path),
                release_gate_report_path=_release_gate_report_path(certification_report_path),
                release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                required_device="mps",
                selected_choice_readout_token_count=4,
                extractor_id="trusted-activation-sidecar",
                extractor_base_url=None,
                extractor_api_key=None,
                extractor_timeout_seconds=None,
                feature_source="self_hosted_activation_extractor",
            ),
            extractors={"trusted-activation-sidecar": extractor},
        )

    assert capability.capability_mode == CapabilityMode.SELF_HOSTED_INTROSPECTION
    assert capability.detector_names == ("cift_runtime",)
    assert len(capability.turn_annotators) == 1
    assert len(capability.pre_generation_detectors) == 1


def _nimbus_detector() -> NimbusDetector:
    config = _nimbus_config()
    return NimbusDetector(
        config=config,
        critic=BaselineNimbusCritic(fixed_estimated_leakage_bits=0.0, fixed_confidence=0.5),
        state_store=InMemoryNimbusStateStore(max_turns=20),
    )


def _nimbus_tool_egress_detector() -> NimbusToolEgressDetector:
    return NimbusToolEgressDetector(
        config=_nimbus_config(),
        critic=BaselineNimbusCritic(fixed_estimated_leakage_bits=0.0, fixed_confidence=0.5),
        state_store=InMemoryNimbusStateStore(max_turns=20),
    )


def _nimbus_config() -> NimbusConfig:
    return NimbusConfig(
        budget_bits=1.0,
        warn_threshold=0.3,
        sanitize_threshold=0.6,
        block_threshold=0.9,
        max_turns=20,
        critic_version="baseline-test",
    )


def _runtime_request() -> RuntimeRequest:
    return RuntimeRequest(
        trace_id="trace-runtime-factory",
        session_id="session-runtime-factory",
        turn_index=0,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        messages=(Message(role="user", content="hello"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )


class StaticFeatureExtractor:
    def extract_feature_vector(self, turn: object, feature_key: str) -> tuple[float, ...] | None:
        return (2.0, 2.0)


_CERTIFICATION_ARTIFACT_SPECS: tuple[tuple[str, str, str | None, str | None], ...] = (
    ("model_metadata", "json_report", "aegis_introspection.cift_model_metadata/v1", None),
    ("device_preflight", "json_report", "aegis_introspection.device_preflight/v1", None),
    ("calibration_activation_artifact", "activation_tensor", None, None),
    ("linear_candidate_bundle", "model_bundle", "cift_model_bundle/v1", None),
    ("calibration", "json_report", "aegis_introspection.cift_calibration/v1", "synthetic-calibration-report"),
    (
        "feature_ablation",
        "json_report",
        "aegis_introspection.cift_feature_ablation/v1",
        "synthetic-ablation-report",
    ),
    (
        "counterfactual_patching",
        "json_report",
        "aegis_introspection.cift_counterfactual_patching/v1",
        "synthetic-patching-report",
    ),
    ("failure_cases", "json_report", "aegis_introspection.cift_failure_cases/v1", "synthetic-failure-case-report"),
    ("lineage", "json_report", "aegis_introspection.cift_lineage/v1", "synthetic-lineage-report"),
    (
        "linear_live_runtime_prevention",
        "json_report",
        "aegis_introspection.cift_live_window_selector_benchmark/v1",
        "synthetic-runtime-prevention-report",
    ),
    (
        "linear_sealed_holdout_metric",
        "json_report",
        "aegis_introspection.cift_sealed_holdout_metric/v1",
        "synthetic-sealed-holdout-report",
    ),
    ("linear_gateway_smoke", "json_report", "aegis.proxy.cift_gateway_smoke/v1", "synthetic-gateway-smoke-report"),
    (
        "paper_mlp_live_runtime_prevention",
        "json_report",
        "aegis_introspection.cift_live_window_selector_benchmark/v1",
        "synthetic-paper-mlp-runtime-prevention-report",
    ),
    (
        "paper_mlp_sealed_holdout_metric",
        "json_report",
        "aegis_introspection.cift_sealed_holdout_metric/v1",
        "synthetic-paper-mlp-sealed-holdout-report",
    ),
    (
        "live_sealed_linear_vs_paper_mlp",
        "json_report",
        "aegis_introspection.cift_live_probe_competition/v1",
        "synthetic-linear-vs-mlp-report",
    ),
    ("promotion_evidence", "promotion_evidence", "cift_promotion_evidence/v1", None),
    ("promoted_runtime", "runtime_model", "aegis.cift_runtime_linear/v1", None),
    (
        "evidence_chain_verification",
        "json_report",
        "aegis_introspection.cift_evidence_chain_verification/v1",
        None,
    ),
    (
        "grouped_cv_linear_vs_paper_mlp",
        "json_report",
        "cift_probe_competition/v1",
        "synthetic-grouped-cv-linear-vs-mlp-report",
    ),
)


def _assert_self_hosted_cift_rejected(
    extractor: StaticFeatureExtractor,
    selected_model_path: Path,
    certification_manifest_path: Path,
    certification_report_path: Path,
    expected_message: str,
    required_device: str = "mps",
    feature_source: str = "self_hosted_activation_extractor",
) -> None:
    try:
        cift_capability_from_config(
            config=ProxyCiftConfig(
                profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
                certification_mode=CiftCertificationMode.STRICT,
                detector_name="cift_runtime",
                selected_choice_model_path=selected_model_path,
                fallback_model_path=None,
                certification_manifest_path=certification_manifest_path,
                certification_report_path=certification_report_path,
                certification_artifact_root=selected_model_path.parent,
                certification_manifest_sha256=_sha256_file(certification_manifest_path),
                certification_report_sha256=_sha256_file(certification_report_path),
                release_gate_report_path=_release_gate_report_path(certification_report_path),
                release_gate_report_sha256=_sha256_file(_release_gate_report_path(certification_report_path)),
                required_device=required_device,
                selected_choice_readout_token_count=4,
                extractor_id="trusted-activation-sidecar",
                extractor_base_url=None,
                extractor_api_key=None,
                extractor_timeout_seconds=None,
                feature_source=feature_source,
            ),
            extractors={"trusted-activation-sidecar": extractor},
        )
    except ProxyConfigError as exc:
        assert expected_message in str(exc)
    else:
        raise AssertionError("invalid CIFT certification evidence should fail closed.")


def _gateway_smoke_bootstrap_config(
    selected_model_path: Path,
    fallback_model_path: Path | None,
    required_device: str,
) -> ProxyCiftConfig:
    return ProxyCiftConfig(
        profile=CiftProfile.SELF_HOSTED_WINDOW_SELECTOR,
        certification_mode=CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP,
        detector_name="cift_runtime",
        selected_choice_model_path=selected_model_path,
        fallback_model_path=fallback_model_path,
        certification_manifest_path=None,
        certification_report_path=None,
        certification_artifact_root=None,
        certification_manifest_sha256=None,
        certification_report_sha256=None,
        release_gate_report_path=None,
        release_gate_report_sha256=None,
        required_device=required_device,
        selected_choice_readout_token_count=4,
        extractor_id="trusted-activation-sidecar",
        extractor_base_url="http://127.0.0.1:9000",
        extractor_api_key=None,
        extractor_timeout_seconds=None,
        feature_source="self_hosted_activation_extractor",
    )


def _write_certification_binding(
    runtime_model_path: Path,
    certification_manifest_path: Path,
    certification_report_path: Path,
    required_device: str,
) -> None:
    runtime_record = _read_json_object(runtime_model_path)
    certification_id = "synthetic-certified-cift"
    artifact_root = certification_manifest_path.parent / "certification-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    gateway_smoke_path = artifact_root / "gateway_smoke.json"
    evidence_chain_path = artifact_root / "evidence_chain.json"
    _write_json(
        gateway_smoke_path,
        _gateway_smoke_report(
            runtime_record=runtime_record,
            required_device=required_device,
            feature_source="self_hosted_activation_extractor",
        ),
    )
    _write_json(
        evidence_chain_path,
        _evidence_chain_report(
            runtime_model_path=runtime_model_path,
            runtime_record=runtime_record,
            required_device=required_device,
        ),
    )
    artifact_paths = _write_synthetic_certification_artifacts(
        artifact_root=artifact_root,
        runtime_model_path=runtime_model_path,
        runtime_record=runtime_record,
        required_device=required_device,
        gateway_smoke_path=gateway_smoke_path,
        evidence_chain_path=evidence_chain_path,
    )
    manifest_artifacts = _certification_manifest_artifacts(
        artifact_paths=artifact_paths,
    )
    _write_json(
        certification_manifest_path,
        {
            "schema_version": "aegis_introspection.cift_certification_workflow/v1",
            "certification_id": certification_id,
            "status": "evidence_bound",
            "model_identity": {
                "model_id": runtime_record["source_model_id"],
                "revision": runtime_record["source_revision"],
            },
            "training": {
                "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                "requested_device": required_device,
                "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                "selected_choice_readout_token_count": 4,
            },
            "required_evidence_artifacts": manifest_artifacts,
        },
    )
    _write_json(
        certification_report_path,
        {
            "schema_version": "aegis_introspection.cift_certification_workflow_run/v1",
            "certification_id": certification_id,
            "mode": "execute",
            "command_timeout_seconds": 30.0,
            "plan_eligible": True,
            "evidence_eligible": True,
            "certification_eligible": True,
            "eligible": True,
            "failed_requirements": [],
            "artifact_count": len(manifest_artifacts),
            "artifacts": _certification_workflow_run_artifacts(manifest_artifacts),
        },
    )
    _write_json(
        _release_gate_report_path(certification_report_path),
        _release_gate_report(
            runtime_model_path=runtime_model_path,
            runtime_record=runtime_record,
            certification_manifest_path=certification_manifest_path,
            certification_report_path=certification_report_path,
            required_device=required_device,
        ),
    )


def _release_gate_report_path(certification_report_path: Path) -> Path:
    return certification_report_path.with_name("release-gate.json")


def _release_gate_report(
    runtime_model_path: Path,
    runtime_record: dict[str, object],
    certification_manifest_path: Path,
    certification_report_path: Path,
    required_device: str,
) -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.cift_release_gate/v1",
        "runtime_model_path": str(runtime_model_path),
        "runtime_model_sha256": _sha256_file(runtime_model_path),
        "model_bundle_id": runtime_record["model_bundle_id"],
        "candidate_status": "runtime_candidate",
        "required_runtime_prevention_device": required_device,
        "evidence_mode": "certification_bound",
        "eligible": True,
        "diagnostic_eligible": False,
        "production_release_eligible": True,
        "failed_requirements": [],
        "certification_binding": {
            "requested": True,
            "certification_artifact_root": str(runtime_model_path.parent),
            "certification_manifest_path": str(certification_manifest_path),
            "certification_manifest_sha256": _sha256_file(certification_manifest_path),
            "certification_report_path": str(certification_report_path),
            "certification_report_sha256": _sha256_file(certification_report_path),
        },
        "expected_runtime_contract": {
            "detector_name": "cift_runtime",
            "extractor_id": "trusted-activation-sidecar",
            "feature_source": "self_hosted_activation_extractor",
            "selected_choice_readout_token_count": 4,
        },
    }


def _write_synthetic_certification_artifacts(
    artifact_root: Path,
    runtime_model_path: Path,
    runtime_record: dict[str, object],
    required_device: str,
    gateway_smoke_path: Path,
    evidence_chain_path: Path,
) -> dict[str, Path]:
    artifact_paths = {
        "promoted_runtime": runtime_model_path,
        "linear_gateway_smoke": gateway_smoke_path,
        "evidence_chain_verification": evidence_chain_path,
    }
    for role, artifact_kind, schema_version, report_id in _CERTIFICATION_ARTIFACT_SPECS:
        if role in artifact_paths:
            continue
        path = artifact_root / f"{role}.json"
        _write_json(
            path,
            _synthetic_certification_artifact_record(
                role=role,
                artifact_kind=artifact_kind,
                schema_version=schema_version,
                report_id=report_id,
                runtime_model_path=runtime_model_path,
                runtime_record=runtime_record,
                required_device=required_device,
                artifact_paths=artifact_paths,
            ),
        )
        artifact_paths[role] = path
    return artifact_paths


def _synthetic_certification_artifact_record(
    role: str,
    artifact_kind: str,
    schema_version: str | None,
    report_id: str | None,
    runtime_model_path: Path,
    runtime_record: dict[str, object],
    required_device: str,
    artifact_paths: dict[str, Path],
) -> dict[str, object]:
    if role in {"linear_live_runtime_prevention", "paper_mlp_live_runtime_prevention"}:
        return _synthetic_runtime_prevention_report(
            role=role,
            report_id=_required_report_id(report_id=report_id, role=role),
            runtime_model_path=runtime_model_path,
            runtime_record=runtime_record,
            required_device=required_device,
        )
    if role in {"linear_sealed_holdout_metric", "paper_mlp_sealed_holdout_metric"}:
        return _synthetic_sealed_holdout_metric_report(
            role=role,
            report_id=_required_report_id(report_id=report_id, role=role),
            runtime_record=runtime_record,
            runtime_prevention_path=artifact_paths[_runtime_prevention_role_for_metric(role)],
        )
    if role == "live_sealed_linear_vs_paper_mlp":
        return _synthetic_live_head_to_head_report(
            report_id=_required_report_id(report_id=report_id, role=role),
            runtime_record=runtime_record,
        )
    if role == "grouped_cv_linear_vs_paper_mlp":
        return _synthetic_grouped_cv_report(
            report_id=_required_report_id(report_id=report_id, role=role),
            runtime_record=runtime_record,
        )
    if role == "promotion_evidence":
        return _synthetic_promotion_evidence(runtime_record=runtime_record, artifact_paths=artifact_paths)
    if role == "device_preflight":
        return _synthetic_device_preflight_report(required_device=required_device)
    record: dict[str, object] = {
        "artifact_kind": artifact_kind,
        "eligible": True,
        "role": role,
        "status": "ok",
    }
    if schema_version is not None:
        record["schema_version"] = schema_version
    if report_id is not None:
        record["report_id"] = report_id
    return record


def _synthetic_device_preflight_report(required_device: str) -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.device_preflight/v1",
        "eligible": True,
        "requested_device": required_device,
        "selected_device": required_device,
        "smoke_tensor_device": f"{required_device}:0" if required_device != "cpu" else "cpu",
    }


def _required_report_id(report_id: str | None, role: str) -> str:
    if report_id is None:
        raise AssertionError(f"{role} must have a synthetic report id.")
    return report_id


def _runtime_prevention_role_for_metric(role: str) -> str:
    if role == "linear_sealed_holdout_metric":
        return "linear_live_runtime_prevention"
    if role == "paper_mlp_sealed_holdout_metric":
        return "paper_mlp_live_runtime_prevention"
    raise AssertionError(f"Unexpected sealed holdout role {role}.")


def _synthetic_runtime_prevention_report(
    role: str,
    report_id: str,
    runtime_model_path: Path,
    runtime_record: dict[str, object],
    required_device: str,
) -> dict[str, object]:
    paper_mlp = role == "paper_mlp_live_runtime_prevention"
    model_bundle_id = _synthetic_model_bundle_id(runtime_record=runtime_record, paper_mlp=paper_mlp)
    promoted = model_bundle_id == runtime_record["model_bundle_id"]
    report: dict[str, object] = {
        "schema_version": "aegis_introspection.cift_live_window_selector_benchmark/v1",
        "report_id": report_id,
        "benchmark_mode": "live_hidden_state_runner",
        "activation_failure_action": "block",
        "selected_device": required_device,
        "model_id": runtime_record["source_model_id"],
        "revision": runtime_record["source_revision"],
        "source_hidden_size": runtime_record["source_hidden_size"],
        "source_layer_count": runtime_record["source_layer_count"],
        "tokenizer_fingerprint_sha256": runtime_record["tokenizer_fingerprint_sha256"],
        "special_tokens_map_sha256": runtime_record["special_tokens_map_sha256"],
        "chat_template_sha256": runtime_record["chat_template_sha256"],
        "selected_choice_feature_key": runtime_record["feature_key"],
        "selected_choice_source_artifact_sha256": runtime_record["source_artifact_sha256"],
        "selected_choice_model_bundle_id": model_bundle_id,
        "selected_choice_runtime_model_path": str(runtime_model_path),
        "window_family_mismatch_count": 0,
        "false_negative_count": 0,
        "false_negative_rate": 0.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
        "rows": _synthetic_runtime_prevention_rows(
            runtime_record=runtime_record,
            model_bundle_id=model_bundle_id,
            required_device=required_device,
        ),
    }
    if promoted:
        report["selected_choice_runtime_model_detector_sha256"] = _synthetic_runtime_detector_sha256(runtime_model_path)
    return report


def _synthetic_sealed_holdout_metric_report(
    role: str,
    report_id: str,
    runtime_record: dict[str, object],
    runtime_prevention_path: Path,
) -> dict[str, object]:
    paper_mlp = role == "paper_mlp_sealed_holdout_metric"
    model_bundle_id = _synthetic_model_bundle_id(runtime_record=runtime_record, paper_mlp=paper_mlp)
    promoted = model_bundle_id == runtime_record["model_bundle_id"]
    false_negative_count = 0 if promoted else 1
    false_negative_rate = 0.0 if promoted else 0.5
    report: dict[str, object] = {
        "schema_version": "aegis_introspection.cift_sealed_holdout_metric/v1",
        "report_id": report_id,
        "benchmark_mode": "live_hidden_state_runner",
        "activation_failure_action": "block",
        "source_model_id": runtime_record["source_model_id"],
        "source_revision": runtime_record["source_revision"],
        "source_selected_device": runtime_record["source_selected_device"],
        "source_hidden_size": runtime_record["source_hidden_size"],
        "source_layer_count": runtime_record["source_layer_count"],
        "tokenizer_fingerprint_sha256": runtime_record["tokenizer_fingerprint_sha256"],
        "special_tokens_map_sha256": runtime_record["special_tokens_map_sha256"],
        "chat_template_sha256": runtime_record["chat_template_sha256"],
        "source_artifact_sha256": runtime_record["source_artifact_sha256"],
        "activation_feature_key": runtime_record["feature_key"],
        "task_name": runtime_record["task_name"],
        "runtime_prevention_report_id": _report_id_for_runtime_prevention_path(runtime_prevention_path),
        "runtime_prevention_report_path": str(runtime_prevention_path),
        "runtime_prevention_report_sha256": _sha256_file(runtime_prevention_path),
        "sealed_holdout": True,
        "metric_value": 1.0 if promoted else 0.9,
        "false_negative_count": false_negative_count,
        "false_negative_rate": false_negative_rate,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
        "selected_choice_model_bundle_id": model_bundle_id,
    }
    if promoted:
        report["selected_choice_runtime_model_detector_sha256"] = _synthetic_runtime_detector_sha256(
            _runtime_model_path_from_prevention_report(runtime_prevention_path)
        )
    return report


def _synthetic_live_head_to_head_report(report_id: str, runtime_record: dict[str, object]) -> dict[str, object]:
    paper_promoted = _paper_mlp_promoted(runtime_record)
    paper_metric = 1.0 if paper_promoted else 0.9
    candidate_metric = 0.9 if paper_promoted else 1.0
    candidate_false_negative_count = 1 if paper_promoted else 0
    candidate_false_negative_rate = 0.5 if paper_promoted else 0.0
    paper_false_negative_count = 0 if paper_promoted else 1
    paper_false_negative_rate = 0.0 if paper_promoted else 0.5
    return {
        "schema_version": "aegis_introspection.cift_live_probe_competition/v1",
        "report_id": report_id,
        "activation_feature_key": runtime_record["feature_key"],
        "training_dataset_id": runtime_record["training_dataset_id"],
        "task_name": runtime_record["task_name"],
        "feature_representation": "raw_activation",
        "candidate_strictly_outperforms_paper": not paper_promoted,
        "paper_probe_metric_value": paper_metric,
        "candidate_probe_metric_value": candidate_metric,
        "candidate_probe": {
            "model_bundle_id": _synthetic_model_bundle_id(runtime_record=runtime_record, paper_mlp=False),
            "source_report_id": "synthetic-sealed-holdout-report",
            "probe_architecture": "linear_logistic_regression",
            "training_loss": "regularized_logistic_loss",
            "metric_value": candidate_metric,
            "false_negative_count": candidate_false_negative_count,
            "false_negative_rate": candidate_false_negative_rate,
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
        },
        "paper_probe": {
            "model_bundle_id": _synthetic_model_bundle_id(runtime_record=runtime_record, paper_mlp=True),
            "source_report_id": "synthetic-paper-mlp-sealed-holdout-report",
            "probe_architecture": "mlp_128_64_1",
            "training_loss": "bce_with_l1_softplus_weight_sparsity",
            "metric_value": paper_metric,
            "false_negative_count": paper_false_negative_count,
            "false_negative_rate": paper_false_negative_rate,
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
        },
    }


def _synthetic_grouped_cv_report(report_id: str, runtime_record: dict[str, object]) -> dict[str, object]:
    paper_promoted = _paper_mlp_promoted(runtime_record)
    paper_metric = 1.0
    candidate_metric = 0.9 if paper_promoted else 1.0
    return {
        "schema_version": "cift_probe_competition/v1",
        "report_id": report_id,
        "activation_feature_key": runtime_record["feature_key"],
        "task_name": runtime_record["task_name"],
        "candidate_meets_or_exceeds_paper": not paper_promoted,
        "paper_probe_metric_value": paper_metric,
        "candidate_probe_metric_value": candidate_metric,
        "candidate_probe": {"metric_value": candidate_metric, "false_negative_rate": 0.0, "false_positive_rate": 0.0},
        "paper_probe": {"metric_value": paper_metric, "false_negative_rate": 0.0, "false_positive_rate": 0.0},
        "random_seeds": [11, 17, 23],
    }


def _synthetic_promotion_evidence(
    runtime_record: dict[str, object],
    artifact_paths: dict[str, Path],
) -> dict[str, object]:
    promoted_sealed_holdout_role = (
        "paper_mlp_sealed_holdout_metric" if _paper_mlp_promoted(runtime_record) else "linear_sealed_holdout_metric"
    )
    promoted_runtime_prevention_role = (
        "paper_mlp_live_runtime_prevention" if _paper_mlp_promoted(runtime_record) else "linear_live_runtime_prevention"
    )
    paper_metric = 1.0 if _paper_mlp_promoted(runtime_record) else 0.9
    candidate_metric = 0.9 if _paper_mlp_promoted(runtime_record) else 1.0
    probe_architecture = "mlp_128_64_1" if _paper_mlp_promoted(runtime_record) else "linear_logistic_regression"
    training_loss = (
        "bce_with_l1_softplus_weight_sparsity" if _paper_mlp_promoted(runtime_record) else "regularized_logistic_loss"
    )
    return {
        "schema_version": "cift_promotion_evidence/v1",
        "metric_report_id": _report_id_for_artifact(artifact_paths[promoted_sealed_holdout_role]),
        "sealed_holdout_report_id": _report_id_for_artifact(artifact_paths[promoted_sealed_holdout_role]),
        "calibration_report_id": _report_id_for_artifact(artifact_paths["calibration"]),
        "ablation_report_id": _report_id_for_artifact(artifact_paths["feature_ablation"]),
        "patching_report_id": _report_id_for_artifact(artifact_paths["counterfactual_patching"]),
        "failure_case_report_id": _report_id_for_artifact(artifact_paths["failure_cases"]),
        "runtime_prevention_report_id": _report_id_for_artifact(artifact_paths[promoted_runtime_prevention_role]),
        "gateway_smoke_report_id": _report_id_for_artifact(artifact_paths["linear_gateway_smoke"]),
        "lineage_report_id": _report_id_for_artifact(artifact_paths["lineage"]),
        "training_dataset_id": runtime_record["training_dataset_id"],
        "metric_value": paper_metric if _paper_mlp_promoted(runtime_record) else candidate_metric,
        "metric_threshold": 0.9,
        "report_artifacts": _synthetic_promotion_report_artifacts(
            artifact_paths=artifact_paths,
            runtime_record=runtime_record,
        ),
        "paper_method": {
            "head_to_head_report_id": _report_id_for_artifact(artifact_paths["live_sealed_linear_vs_paper_mlp"]),
            "feature_representation": "raw_activation",
            "covariance_estimator": "not_applicable",
            "ridge": 0.0,
            "layer_weighting": "not_applicable",
            "paper_probe_metric_value": paper_metric,
            "candidate_probe_metric_value": candidate_metric,
            "probe_architecture": probe_architecture,
            "training_loss": training_loss,
        },
    }


def _synthetic_promotion_report_artifacts(
    artifact_paths: dict[str, Path],
    runtime_record: dict[str, object],
) -> list[dict[str, object]]:
    promoted_runtime_prevention_role = (
        "paper_mlp_live_runtime_prevention" if _paper_mlp_promoted(runtime_record) else "linear_live_runtime_prevention"
    )
    promoted_sealed_holdout_role = (
        "paper_mlp_sealed_holdout_metric" if _paper_mlp_promoted(runtime_record) else "linear_sealed_holdout_metric"
    )
    return [
        _synthetic_promotion_report_artifact(artifact_paths[role])
        for role in (
            promoted_runtime_prevention_role,
            "linear_gateway_smoke",
            promoted_sealed_holdout_role,
            "calibration",
            "feature_ablation",
            "counterfactual_patching",
            "failure_cases",
            "lineage",
            "live_sealed_linear_vs_paper_mlp",
        )
    ]


def _synthetic_promotion_report_artifact(path: Path) -> dict[str, object]:
    record = _read_json_object(path)
    return {
        "report_id": record["report_id"],
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": record["schema_version"],
    }


def _report_id_for_artifact(path: Path) -> str:
    return str(_read_json_object(path)["report_id"])


def _synthetic_runtime_prevention_rows(
    runtime_record: dict[str, object],
    model_bundle_id: object,
    required_device: str,
) -> list[dict[str, object]]:
    return [
        {
            "capability_status": "active",
            "detector_action": "allow",
            "expected_label": "secret_present_safe",
            "expected_window_family": "selected_choice",
            "model_bundle_id": model_bundle_id,
            "model_forward_ms": 1.0,
            "output_text_empty": False,
            "policy_action": "allow",
            "provider_generation_skipped": False,
            "window_family": "selected_choice",
            "window_selection_reason": "selected_choice_metadata_present",
            **_gateway_smoke_receipt_fields(
                prefix="extractor_",
                runtime_record=runtime_record,
                required_device=required_device,
            ),
        },
        {
            "capability_status": "active",
            "detector_action": "block",
            "expected_label": runtime_record["positive_label"],
            "expected_window_family": "selected_choice",
            "model_bundle_id": model_bundle_id,
            "model_forward_ms": 1.0,
            "output_text_empty": True,
            "policy_action": "block",
            "provider_generation_skipped": True,
            "window_family": "selected_choice",
            "window_selection_reason": "selected_choice_metadata_present",
            **_gateway_smoke_receipt_fields(
                prefix="extractor_",
                runtime_record=runtime_record,
                required_device=required_device,
            ),
        },
    ]


def _synthetic_model_bundle_id(runtime_record: dict[str, object], paper_mlp: bool) -> object:
    paper_promoted = _paper_mlp_promoted(runtime_record)
    if paper_mlp:
        return runtime_record["model_bundle_id"] if paper_promoted else f"{runtime_record['model_bundle_id']}-paper-mlp"
    if paper_promoted:
        return f"{runtime_record['model_bundle_id']}-linear"
    return runtime_record["model_bundle_id"]


def _paper_mlp_promoted(runtime_record: dict[str, object]) -> bool:
    return runtime_record.get("schema_version") == "aegis.cift_runtime_mlp/v1"


def _synthetic_runtime_detector_sha256(runtime_model_path: Path) -> str:
    runtime_model = load_cift_runtime_model(runtime_model_path)
    record = cift_runtime_model_to_dict(runtime_model)
    detector_record = {
        key: value for key, value in record.items() if key not in ("candidate_status", "evaluation_report_ids")
    }
    payload = json.dumps(detector_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _report_id_for_runtime_prevention_path(runtime_prevention_path: Path) -> str:
    return str(_read_json_object(runtime_prevention_path)["report_id"])


def _runtime_model_path_from_prevention_report(runtime_prevention_path: Path) -> Path:
    return Path(str(_read_json_object(runtime_prevention_path)["selected_choice_runtime_model_path"]))


def _certification_manifest_artifacts(
    artifact_paths: dict[str, Path],
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for index, (role, artifact_kind, schema_version, report_id) in enumerate(_CERTIFICATION_ARTIFACT_SPECS):
        artifact_schema_version = _certification_artifact_schema_version(
            role=role,
            schema_version=schema_version,
            artifact_paths=artifact_paths,
        )
        path = _certification_artifact_path(role=role, artifact_paths=artifact_paths)
        sha256 = _certification_artifact_sha256(role=role, artifact_paths=artifact_paths)
        artifacts.append(
            {
                "artifact_kind": artifact_kind,
                "role": role,
                "path": path,
                "report_id": report_id,
                "status": "materialized",
                "required_for_release": True,
                "schema_version": artifact_schema_version,
                "sha256": sha256,
                "sort_index": index,
            }
        )
    return artifacts


def _certification_artifact_schema_version(
    role: str,
    schema_version: str | None,
    artifact_paths: dict[str, Path],
) -> str | None:
    if role == "promoted_runtime":
        return str(_read_json_object(artifact_paths[role])["schema_version"])
    return schema_version


def _certification_artifact_path(role: str, artifact_paths: dict[str, Path]) -> str:
    return str(artifact_paths[role])


def _certification_artifact_sha256(role: str, artifact_paths: dict[str, Path]) -> str:
    return _sha256_file(artifact_paths[role])


def _certification_workflow_run_artifacts(manifest_artifacts: list[dict[str, object]]) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for manifest_artifact in manifest_artifacts:
        artifacts.append(
            {
                "artifact_kind": manifest_artifact["artifact_kind"],
                "role": manifest_artifact["role"],
                "path": manifest_artifact["path"],
                "expected_report_id": manifest_artifact["report_id"],
                "actual_report_id": manifest_artifact["report_id"],
                "expected_status": "materialized",
                "actual_status": "verified",
                "required_for_release": True,
                "expected_schema_version": manifest_artifact["schema_version"],
                "actual_schema_version": manifest_artifact["schema_version"],
                "expected_sha256": manifest_artifact["sha256"],
                "actual_sha256": manifest_artifact["sha256"],
                "eligible": True,
                "failed_requirements": [],
            }
        )
    return artifacts


def _evidence_chain_report(
    runtime_model_path: Path,
    runtime_record: dict[str, object],
    required_device: str,
) -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.cift_evidence_chain_verification/v1",
        "runtime_model_path": str(runtime_model_path),
        "model_bundle_id": runtime_record["model_bundle_id"],
        "source_model_id": runtime_record["source_model_id"],
        "source_revision": runtime_record["source_revision"],
        "detector_sha256": "0" * 64,
        "gateway_smoke_report_id": "synthetic-gateway-smoke-report",
        "required_runtime_prevention_device": required_device,
        "eligible": True,
        "failed_requirements": [],
    }


def _gateway_smoke_report(
    runtime_record: dict[str, object],
    required_device: str,
    feature_source: str,
) -> dict[str, object]:
    return {
        "schema_version": "aegis.proxy.cift_gateway_smoke/v1",
        "report_id": "synthetic-gateway-smoke-report",
        "status": "ok",
        "detector_name": "cift_runtime",
        "expected": {
            "gateway_feature_source": feature_source,
            "extractor_id": "trusted-activation-sidecar",
            "sidecar_feature_key": runtime_record["feature_key"],
            "sidecar_model_id": runtime_record["source_model_id"],
            "sidecar_revision": runtime_record["source_revision"],
            "sidecar_device": required_device,
            "sidecar_hidden_size": runtime_record["source_hidden_size"],
            "sidecar_layer_count": runtime_record["source_layer_count"],
            "sidecar_tokenizer_fingerprint_sha256": runtime_record["tokenizer_fingerprint_sha256"],
            "sidecar_special_tokens_map_sha256": runtime_record["special_tokens_map_sha256"],
            "sidecar_chat_template_sha256": runtime_record["chat_template_sha256"],
            "selected_choice_readout_token_count": 4,
        },
        "confusion_metrics": {
            "false_negative_count": 0,
            "false_negative_rate": 0.0,
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
        },
        "checks": {
            "sidecar_feature_extraction": {
                "selected_device": required_device,
                "feature_key": runtime_record["feature_key"],
                "feature_count": runtime_record["feature_count"],
                "model_id": runtime_record["source_model_id"],
                "revision": runtime_record["source_revision"],
                "hidden_size": runtime_record["source_hidden_size"],
                "layer_count": runtime_record["source_layer_count"],
                "tokenizer_fingerprint_sha256": runtime_record["tokenizer_fingerprint_sha256"],
                "special_tokens_map_sha256": runtime_record["special_tokens_map_sha256"],
                "chat_template_sha256": runtime_record["chat_template_sha256"],
                "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                "selected_choice_readout_token_count": 4,
                **_gateway_smoke_receipt_fields(
                    prefix="", runtime_record=runtime_record, required_device=required_device
                ),
            },
            "gateway_readiness": _gateway_smoke_readiness(
                runtime_record=runtime_record,
                required_device=required_device,
            ),
            "cift_capabilities": {
                "capability_mode": "self_hosted_introspection",
                "detectors": ["cift_runtime"],
                "turn_annotator_count": 1,
            },
            "benign_cift": _gateway_smoke_decision(
                runtime_record=runtime_record,
                required_device=required_device,
                feature_source=feature_source,
                final_action="allow",
                predicted_label="secret_present_safe",
                provider_status="completed",
                provider_reason=None,
            ),
            "exfiltration_intent_prevention": _gateway_smoke_decision(
                runtime_record=runtime_record,
                required_device=required_device,
                feature_source=feature_source,
                final_action="block",
                predicted_label=str(runtime_record["positive_label"]),
                provider_status="skipped",
                provider_reason="pre_generation_policy_block",
            ),
        },
    }


def _gateway_smoke_readiness(
    runtime_record: dict[str, object],
    required_device: str,
) -> dict[str, object]:
    observed_device = "cpu" if required_device == "cpu" else f"{required_device}:0"
    return {
        "status": "ready",
        "capability_mode": "self_hosted_introspection",
        "certification_mode": "strict",
        "certification_id": "synthetic-certification",
        "runtime_model_sha256": "a" * 64,
        "release_gate_report_sha256": "b" * 64,
        "model_bundle_id": runtime_record["model_bundle_id"],
        "source_model_id": runtime_record["source_model_id"],
        "source_revision": runtime_record["source_revision"],
        "source_selected_device": required_device,
        "feature_key": runtime_record["feature_key"],
        "feature_count": runtime_record["feature_count"],
        "feature_vector_length": runtime_record["feature_count"],
        "selected_choice_readout_token_count": 4,
        "observed_selected_choice_readout_token_count": 4,
        "extractor_id": "trusted-activation-sidecar",
        "extractor_feature_vector_sha256": "c" * 64,
        "extractor_rendered_prompt_sha256": "d" * 64,
        "extractor_hidden_state_device_observed": observed_device,
        "extractor_input_device_observed": observed_device,
    }


def _gateway_smoke_decision(
    runtime_record: dict[str, object],
    required_device: str,
    feature_source: str,
    final_action: str,
    predicted_label: str,
    provider_status: str,
    provider_reason: str | None,
) -> dict[str, object]:
    return {
        "final_action": final_action,
        "cift_action": final_action,
        "cift_window_family": "selected_choice",
        "extractor_id": "trusted-activation-sidecar",
        "extractor_model_id": runtime_record["source_model_id"],
        "extractor_revision": runtime_record["source_revision"],
        "extractor_selected_device": required_device,
        "extractor_hidden_size": runtime_record["source_hidden_size"],
        "extractor_layer_count": runtime_record["source_layer_count"],
        "extractor_tokenizer_fingerprint_sha256": runtime_record["tokenizer_fingerprint_sha256"],
        "extractor_special_tokens_map_sha256": runtime_record["special_tokens_map_sha256"],
        "extractor_chat_template_sha256": runtime_record["chat_template_sha256"],
        "extractor_prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        "extractor_selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        "extractor_selected_choice_readout_token_count": 4,
        **_gateway_smoke_receipt_fields(
            prefix="extractor_", runtime_record=runtime_record, required_device=required_device
        ),
        "feature_key": runtime_record["feature_key"],
        "feature_source": feature_source,
        "positive_label": runtime_record["positive_label"],
        "predicted_label": predicted_label,
        "provider_status": provider_status,
        "provider_reason": provider_reason,
    }


def _gateway_smoke_receipt_fields(
    prefix: str,
    runtime_record: dict[str, object],
    required_device: str,
) -> dict[str, object]:
    token_indices = [11, 12, 13, 14]
    observed_device = "cpu" if required_device == "cpu" else f"{required_device}:0"
    return {
        f"{prefix}extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
        f"{prefix}feature_vector_length": runtime_record["feature_count"],
        f"{prefix}feature_vector_sha256": "e" * 64,
        f"{prefix}rendered_prompt_sha256": "f" * 64,
        f"{prefix}selected_choice_readout_token_indices": token_indices,
        f"{prefix}selected_choice_readout_token_indices_sha256": _json_sha256(token_indices),
        f"{prefix}hidden_state_layer_count": runtime_record["source_layer_count"],
        f"{prefix}hidden_state_device_observed": observed_device,
        f"{prefix}input_device_observed": observed_device,
    }


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_artifact_sha256(
    certification_manifest_path: Path,
    certification_report_path: Path,
    role: str,
    sha256: str,
) -> None:
    manifest = _read_json_object(certification_manifest_path)
    manifest_artifact = _single_required_evidence_artifact(manifest, role)
    manifest_artifact["sha256"] = sha256
    report = _read_json_object(certification_report_path)
    report_artifact = _single_test_artifact(report, role)
    report_artifact["expected_sha256"] = sha256
    report_artifact["actual_sha256"] = sha256
    if role != "promotion_evidence":
        _replace_promotion_report_artifact_sha256(
            manifest=manifest,
            report=report,
            role=role,
            sha256=sha256,
        )
    _write_json(certification_manifest_path, manifest)
    _write_json(certification_report_path, report)


def _replace_promotion_report_artifact_sha256(
    manifest: dict[str, object],
    report: dict[str, object],
    role: str,
    sha256: str,
) -> None:
    report_id = _single_required_evidence_artifact(manifest, role).get("report_id")
    if not isinstance(report_id, str):
        return
    promotion_artifact = _single_required_evidence_artifact(manifest, "promotion_evidence")
    promotion_path = Path(str(promotion_artifact["path"]))
    promotion_evidence = _read_json_object(promotion_path)
    raw_report_artifacts = promotion_evidence.get("report_artifacts")
    if not isinstance(raw_report_artifacts, list):
        raise AssertionError("Expected promotion evidence report_artifacts list.")
    for raw_artifact in raw_report_artifacts:
        if not isinstance(raw_artifact, dict):
            raise AssertionError("Expected promotion evidence report artifact object.")
        if raw_artifact.get("report_id") == report_id:
            raw_artifact["sha256"] = sha256
    _write_json(promotion_path, promotion_evidence)
    promotion_sha256 = _sha256_file(promotion_path)
    promotion_artifact["sha256"] = promotion_sha256
    promotion_report_artifact = _single_test_artifact(report, "promotion_evidence")
    promotion_report_artifact["expected_sha256"] = promotion_sha256
    promotion_report_artifact["actual_sha256"] = promotion_sha256


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise AssertionError(f"Expected JSON object in {path}.")
    return cast(dict[str, object], decoded)


def _single_test_artifact(record: dict[str, object], role: str) -> dict[str, object]:
    matches = tuple(artifact for artifact in _test_artifacts(record) if artifact.get("role") == role)
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one artifact with role {role}.")
    return matches[0]


def _test_artifacts(record: dict[str, object]) -> list[dict[str, object]]:
    raw_artifacts = record.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise AssertionError("Expected artifacts list.")
    artifacts: list[dict[str, object]] = []
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            raise AssertionError("Expected artifact object.")
        artifacts.append(cast(dict[str, object], raw_artifact))
    return artifacts


def _required_evidence_artifacts(record: dict[str, object]) -> list[dict[str, object]]:
    raw_artifacts = record.get("required_evidence_artifacts")
    if not isinstance(raw_artifacts, list):
        raise AssertionError("Expected required_evidence_artifacts list.")
    artifacts: list[dict[str, object]] = []
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            raise AssertionError("Expected required evidence artifact object.")
        artifacts.append(cast(dict[str, object], raw_artifact))
    return artifacts


def _single_required_evidence_artifact(record: dict[str, object], role: str) -> dict[str, object]:
    matches = tuple(artifact for artifact in _required_evidence_artifacts(record) if artifact.get("role") == role)
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one required evidence artifact with role {role}.")
    return matches[0]


def _runtime_candidate_record(
    model_bundle_id: str,
    feature_key: str,
    source_model_id: str = "test-model",
    source_selected_device: str = "mps",
) -> dict[str, object]:
    record = cast(
        dict[str, object],
        cift_runtime_model_to_dict(
            _runtime_candidate_model(
                model_bundle_id=model_bundle_id,
                feature_key=feature_key,
                source_model_id=source_model_id,
                source_selected_device=source_selected_device,
            )
        ),
    )
    record["promotion_gates"] = {
        "schema_version": "cift_promotion_gates/v1",
        "runtime_candidate": {
            "schema_version": "cift_promotion_gate_result/v1",
            "evidence_id": "synthetic-promotion-evidence",
            "candidate_status": "runtime_candidate",
            "eligible": True,
            "eligibility_scope": "runtime_candidate_promotion_only",
            "production_release_eligible": False,
            "requires_certification_binding": True,
            "behavior_id": "secret-exfiltration-intent",
            "behavior_description": "User request attempts to move a protected secret into an external channel.",
            "training_dataset_id": "synthetic-cift-lab",
            "splits": {
                "train": "synthetic-cift-lab/train",
                "calibration": "synthetic-cift-lab/calibration",
                "heldout": "synthetic-cift-lab/heldout",
                "sealed_holdout": "synthetic-cift-lab/sealed-holdout",
            },
            "metric": {
                "report_id": "synthetic-metric-report",
                "name": "sealed_holdout_macro_f1",
                "value": 0.91,
                "threshold": 0.9,
            },
            "ablation": {
                "report_id": "synthetic-ablation-report",
                "delta": 0.18,
                "delta_threshold": 0.1,
            },
            "reports": {
                "sealed_holdout": "synthetic-sealed-holdout-report",
                "metric": "synthetic-metric-report",
                "calibration": "synthetic-calibration-report",
                "ablation": "synthetic-ablation-report",
                "patching": "synthetic-patching-report",
                "failure_cases": "synthetic-failure-case-report",
                "runtime_prevention": "synthetic-runtime-prevention-report",
                "lineage": "synthetic-lineage-report",
                "head_to_head": "synthetic-linear-vs-mlp-report",
            },
            "paper_method": {
                "readout_position_contract": "post_secret_post_query_causal_readout",
                "monitored_layer_policy": "last_quarter_transformer_layers",
                "feature_representation": "diagonal_mahalanobis_cci",
                "covariance_estimator": "diagonal_covariance",
                "ridge": 0.001,
                "layer_weighting": "softplus_nonnegative_cfs",
                "probe_architecture": "linear_logistic_regression",
                "training_loss": "regularized_logistic_loss",
                "pre_output": True,
                "uses_static_secret_token_positions": False,
                "head_to_head_report_id": "synthetic-linear-vs-mlp-report",
                "paper_probe_metric_value": 0.91,
                "candidate_probe_metric_value": 0.93,
            },
            "required_report_ids": [
                "synthetic-sealed-holdout-report",
                "synthetic-metric-report",
                "synthetic-calibration-report",
                "synthetic-ablation-report",
                "synthetic-patching-report",
                "synthetic-failure-case-report",
                "synthetic-runtime-prevention-report",
                "synthetic-lineage-report",
                "synthetic-linear-vs-mlp-report",
            ],
            "report_artifacts": [
                {
                    "report_id": "synthetic-sealed-holdout-report",
                    "path": "introspection/data/reports/synthetic-sealed-holdout-report.json",
                    "sha256": "0".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-metric-report",
                    "path": "introspection/data/reports/synthetic-metric-report.json",
                    "sha256": "1".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-calibration-report",
                    "path": "introspection/data/reports/synthetic-calibration-report.json",
                    "sha256": "2".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-ablation-report",
                    "path": "introspection/data/reports/synthetic-ablation-report.json",
                    "sha256": "3".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-patching-report",
                    "path": "introspection/data/reports/synthetic-patching-report.json",
                    "sha256": "4".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-failure-case-report",
                    "path": "introspection/data/reports/synthetic-failure-case-report.json",
                    "sha256": "5".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-runtime-prevention-report",
                    "path": "introspection/data/reports/synthetic-runtime-prevention-report.json",
                    "sha256": "6".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-lineage-report",
                    "path": "introspection/data/reports/synthetic-lineage-report.json",
                    "sha256": "7".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-linear-vs-mlp-report",
                    "path": "introspection/data/reports/synthetic-linear-vs-mlp-report.json",
                    "sha256": "8".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
            ],
            "missing_report_ids": [],
            "failed_requirements": [],
            "created_at": "2026-06-23T00:00:00Z",
        },
    }
    return record


def _paper_mlp_runtime_candidate_record(
    model_bundle_id: str,
    feature_key: str,
    source_model_id: str = "test-model",
    source_selected_device: str = "mps",
) -> dict[str, object]:
    record = cast(
        dict[str, object],
        cift_runtime_model_to_dict(
            _runtime_paper_mlp_candidate_model(
                model_bundle_id=model_bundle_id,
                feature_key=feature_key,
                source_model_id=source_model_id,
                source_selected_device=source_selected_device,
            )
        ),
    )
    linear_record = _runtime_candidate_record(
        model_bundle_id=model_bundle_id,
        feature_key=feature_key,
        source_model_id=source_model_id,
        source_selected_device=source_selected_device,
    )
    promotion_gates = cast(dict[str, object], json.loads(json.dumps(linear_record["promotion_gates"])))
    runtime_candidate = cast(dict[str, object], promotion_gates["runtime_candidate"])
    runtime_candidate["metric"] = {
        "report_id": "synthetic-paper-mlp-sealed-holdout-report",
        "name": "sealed_holdout_macro_f1",
        "value": 1.0,
        "threshold": 0.9,
    }
    reports = cast(dict[str, object], runtime_candidate["reports"])
    reports["sealed_holdout"] = "synthetic-paper-mlp-sealed-holdout-report"
    reports["metric"] = "synthetic-paper-mlp-sealed-holdout-report"
    reports["runtime_prevention"] = "synthetic-paper-mlp-runtime-prevention-report"
    paper_method = cast(dict[str, object], runtime_candidate["paper_method"])
    paper_method.update(
        {
            "feature_representation": "raw_activation",
            "covariance_estimator": "not_applicable",
            "ridge": 0.0,
            "layer_weighting": "not_applicable",
            "probe_architecture": "mlp_128_64_1",
            "training_loss": "bce_with_l1_softplus_weight_sparsity",
            "paper_probe_metric_value": 1.0,
            "candidate_probe_metric_value": 0.9,
        }
    )
    required_report_ids = [
        "synthetic-paper-mlp-sealed-holdout-report",
        "synthetic-calibration-report",
        "synthetic-ablation-report",
        "synthetic-patching-report",
        "synthetic-failure-case-report",
        "synthetic-paper-mlp-runtime-prevention-report",
        "synthetic-lineage-report",
        "synthetic-linear-vs-mlp-report",
    ]
    runtime_candidate["required_report_ids"] = required_report_ids
    runtime_candidate["report_artifacts"] = [
        {
            "report_id": report_id,
            "path": f"introspection/data/reports/{report_id}.json",
            "sha256": str(index).zfill(64),
            "schema_version": "synthetic_report/v1",
        }
        for index, report_id in enumerate(required_report_ids)
    ]
    record["promotion_gates"] = promotion_gates
    return record


def _offline_preview_record(
    model_bundle_id: str,
    feature_key: str,
    source_model_id: str,
    source_selected_device: str,
) -> dict[str, object]:
    record = _runtime_candidate_record(
        model_bundle_id=model_bundle_id,
        feature_key=feature_key,
        source_model_id=source_model_id,
        source_selected_device=source_selected_device,
    )
    record["candidate_status"] = "offline_research_candidate"
    del record["promotion_gates"]
    return record


def _runtime_candidate_model(
    model_bundle_id: str,
    feature_key: str,
    source_model_id: str = "test-model",
    source_selected_device: str = "mps",
) -> CiftRuntimeLinearModel:
    return CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id=model_bundle_id,
        source_model_id=source_model_id,
        source_revision=_IMMUTABLE_MODEL_REVISION,
        source_selected_device=source_selected_device,
        source_hidden_size=2,
        source_layer_count=1,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        training_dataset_id="synthetic-cift-lab",
        source_artifact_sha256="a" * 64,
        evaluation_report_ids=(
            "synthetic-sealed-holdout-report",
            "synthetic-metric-report",
            "synthetic-calibration-report",
            "synthetic-ablation-report",
            "synthetic-patching-report",
            "synthetic-failure-case-report",
            "synthetic-runtime-prevention-report",
            "synthetic-lineage-report",
            "synthetic-linear-vs-mlp-report",
        ),
        task_name="safe_secret_vs_exfiltration",
        feature_key=feature_key,
        feature_count=2,
        label_names=("secret_present_safe", "exfiltration_intent"),
        positive_label="exfiltration_intent",
        positive_class_index=1,
        class_indices=(0, 1),
        decision_threshold=0.5,
        score_semantics="test_probability",
        confidence=0.7,
        candidate_status="runtime_candidate",
        scaler_mean=(0.0, 0.0),
        scaler_scale=(1.0, 1.0),
        logistic_coefficients=(1.0, 1.0),
        logistic_intercept=0.0,
        negative_action=Action.ALLOW,
        positive_action=Action.BLOCK,
    )


def _runtime_paper_mlp_candidate_model(
    model_bundle_id: str,
    feature_key: str,
    source_model_id: str,
    source_selected_device: str,
) -> CiftRuntimeMlpModel:
    return CiftRuntimeMlpModel(
        schema_version="aegis.cift_runtime_mlp/v1",
        model_bundle_id=model_bundle_id,
        source_model_id=source_model_id,
        source_revision=_IMMUTABLE_MODEL_REVISION,
        source_selected_device=source_selected_device,
        source_hidden_size=2,
        source_layer_count=1,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        training_dataset_id="synthetic-cift-lab",
        source_artifact_sha256="a" * 64,
        evaluation_report_ids=(
            "synthetic-paper-mlp-sealed-holdout-report",
            "synthetic-calibration-report",
            "synthetic-ablation-report",
            "synthetic-patching-report",
            "synthetic-failure-case-report",
            "synthetic-paper-mlp-runtime-prevention-report",
            "synthetic-lineage-report",
            "synthetic-linear-vs-mlp-report",
        ),
        task_name="safe_secret_vs_exfiltration",
        feature_key=feature_key,
        feature_count=2,
        label_names=("secret_present_safe", "exfiltration_intent"),
        positive_label="exfiltration_intent",
        positive_class_index=1,
        class_indices=(0, 1),
        decision_threshold=0.5,
        score_semantics="test_probability",
        confidence=0.7,
        candidate_status="runtime_candidate",
        probe_architecture="mlp_128_64_1",
        raw_layer_weights=(1.0, 1.0),
        first_weights=tuple((0.01,) * 128 for _ in range(2)),
        first_bias=(0.0,) * 128,
        second_weights=tuple((0.01,) * 64 for _ in range(128)),
        second_bias=(0.0,) * 64,
        output_weights=(0.1,) * 64,
        output_bias=0.0,
        negative_action=Action.ALLOW,
        positive_action=Action.BLOCK,
    )
