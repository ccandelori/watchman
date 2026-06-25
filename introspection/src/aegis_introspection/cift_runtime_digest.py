from __future__ import annotations

import hashlib
import json

from aegis.detectors.cift_runtime import CiftRuntimeModel, cift_runtime_model_to_dict

_NON_DETECTOR_FIELDS = frozenset(("candidate_status", "evaluation_report_ids"))


def cift_runtime_detector_sha256(model: CiftRuntimeModel) -> str:
    record = cift_runtime_model_to_dict(model)
    detector_record = {key: value for key, value in record.items() if key not in _NON_DETECTOR_FIELDS}
    payload = json.dumps(detector_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
