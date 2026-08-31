import json

import pytest

from ozon_quality.vlm_contract import build_vlm_prompt, parse_vlm_decision


def test_vlm_contract_accepts_strict_json() -> None:
    payload = {
        "predicted_labels": ["restricted"],
        "confidence": {"restricted": 0.8},
        "evidence_text": ["explicit phrase"],
        "evidence_image": [],
        "reason_short": "Text signal only.",
    }
    decision = parse_vlm_decision(json.dumps(payload), {"safe", "restricted"})
    assert decision.predicted_labels == ["restricted"]
    assert "Allowed labels" in build_vlm_prompt("product", ["safe", "restricted"])


def test_vlm_contract_rejects_unknown_label_and_extra_fields() -> None:
    payload = {
        "predicted_labels": ["invented"],
        "confidence": {},
        "evidence_text": [],
        "evidence_image": [],
        "reason_short": "",
    }
    with pytest.raises(ValueError, match="Unknown"):
        parse_vlm_decision(json.dumps(payload), {"safe"})
    payload["extra"] = True
    with pytest.raises(ValueError, match="exactly"):
        parse_vlm_decision(json.dumps(payload), {"safe"})
