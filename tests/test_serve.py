import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from receiptx.config import ExtractionConfig, ExtractRequest, ServeConfig
from receiptx.schema import Item, Receipt
from receiptx.serve import ExtractionService, create_app

SAMPLE_OCR = "DONAT GULA 11,000\nSUBTOTAL 11,000\nTOTAL 11,000\nCASH 20,000\nCHANGE 9,000"


def test_config_validation_rejects_invalid_settings():
    with pytest.raises(ValidationError):
        ExtractionConfig(backend="unknown")
    with pytest.raises(ValidationError):
        ExtractionConfig(max_repair_rounds=5)
    with pytest.raises(ValidationError):
        ExtractRequest(text="")
    with pytest.raises(ValidationError):
        ServeConfig(port=0)


def test_service_rules_and_injected_llm_backend():
    svc = ExtractionService(
        config=ExtractionConfig(backend="rules"),
        llm_extractor=lambda _: Receipt(menu=[Item(nm="DONAT GULA", price="11,000")], total_price="11,000"),
    )
    r_rules = svc.extract(SAMPLE_OCR)
    assert r_rules.backend == "rules"
    assert r_rules.receipt.total_price == "11,000"
    assert r_rules.needs_review is False

    r_llm = svc.extract(SAMPLE_OCR, backend="llm")
    assert r_llm.backend == "llm"
    assert len(r_llm.receipt.menu) == 1
    assert r_llm.needs_review is False


def test_fastapi_extract_batch_validate_and_schema():
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok", "backend": "rules"}
    schema = client.get("/schema").json()
    assert "menu" in schema["required"]

    res = client.post("/extract", json={"text": SAMPLE_OCR})
    assert res.status_code == 200
    body = res.json()
    assert body["receipt"]["total_price"] == "11,000"
    assert body["needs_review"] is False

    batch = client.post("/extract/batch", json={"receipts": [{"text": SAMPLE_OCR}, {"text": "TOTAL 50,000\nCASH 60,000\nCHANGE 5,000"}]})
    assert batch.status_code == 200
    assert batch.json()["flagged_count"] == 1

    val_bad = client.post(
        "/validate",
        json={
            "text": SAMPLE_OCR,
            "receipt": {"menu": [{"nm": "INVENTED ITEM", "price": "99,000"}], "total_price": "11,000"},
        },
    )
    assert val_bad.status_code == 200
    assert val_bad.json()["valid"] is False
    assert len(val_bad.json()["issues"]) >= 2

    no_llm = client.post("/extract", json={"text": SAMPLE_OCR, "backend": "llm"})
    assert no_llm.status_code == 503
