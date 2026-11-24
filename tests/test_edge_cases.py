"""Edge-case, regression, and data-validation tests for receiptx."""

from __future__ import annotations

import pytest

from receiptx.llm import parse
from receiptx.metrics import parse_amount, per_field, score
from receiptx.rules import extract
from receiptx.run import Store
from receiptx.schema import Item, Receipt
from receiptx.validate import arithmetic, check, grounding


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rp. 125.000", 125000.0),
        ("(15,000)", 15000.0),
        ("45.5", 45.5),
        ("0", 0.0),
        ("---", None),
        ("", None),
    ],
)
def test_parse_amount_edge_cases(text, expected):
    assert parse_amount(text) == expected


def test_rules_next_line_amount_and_qty_exclusion():
    text = "TOTAL ITEMS 3\nSUBTOTAL\n25,000\nTAX 10%\n2,500\nGRAND TOTAL\n27,500\nTUNAI 30,000\nKEMBALI 2,500"
    r = extract(text)
    assert r.subtotal_price == "25,000"
    assert r.tax_price == "2,500"
    assert r.total_price == "27,500"
    assert r.cashprice == "30,000"
    assert r.changeprice == "2,500"


def test_rules_duplicate_keywords_keep_first_match():
    text = "TOTAL 15,000\nTOTAL 99,000\nCASH 20,000"
    r = extract(text)
    assert r.total_price == "15,000"
    assert r.cashprice == "20,000"


def test_rules_empty_and_whitespace_input():
    r = extract("   \n\n\t  ")
    assert r == Receipt()


def test_arithmetic_with_discount_and_service_charge():
    ok = Receipt(
        menu=[Item(nm="COFFEE", price="20,000"), Item(nm="CAKE", price="30,000")],
        subtotal_price="50,000",
        discount_price="-5,000",
        service_price="2,500",
        tax_price="4,750",
        total_price="52,250",
    )
    assert arithmetic(ok) == []

    bad = ok.model_copy(update={"total_price": "99,000"})
    issues = arithmetic(bad)
    assert len(issues) == 1
    assert "subtotal_price" in issues[0]


def test_arithmetic_skips_item_sum_when_tax_sits_between_items_and_total():
    r = Receipt(
        menu=[Item(nm="NOODLES", price="20,000")],
        tax_price="2,000",
        total_price="22,000",
    )
    assert arithmetic(r) == []


def test_grounding_handles_substring_digits_and_case_insensitivity():
    source = "Iced Latte  Rp18.500\nTOTAL 18.500"
    ok = Receipt(menu=[Item(nm="iced   latte", price="18,500")], total_price="18500")
    assert grounding(ok, source) == []
    bad = Receipt(menu=[Item(nm="Matcha", price="18,500")], total_price="19,000")
    issues = check(bad, source)
    assert any("Matcha" in i for i in issues)
    assert any("19,000" in i for i in issues)


def test_parse_handles_prose_and_null_fields():
    raw = 'Result:\n{"menu": [{"nm": "WATER", "cnt": null, "unitprice": null, "price": "5,000"}], "total_price": "5,000"}\nDone.'
    p = parse(raw)
    assert p.parse_ok and p.schema_ok and p.receipt is not None
    assert p.receipt.menu[0].nm == "WATER"
    assert p.receipt.menu[0].cnt is None


def test_metrics_handle_all_none_predictions():
    gold = Receipt(menu=[Item(nm="TEA", price="5,000")], total_price="5,000")
    s = score([None, None], [gold, gold], n_boot=20)
    assert s["f1"] == 0.0 and s["precision"] == 0.0 and s["recall"] == 0.0
    pf = per_field([None], [gold])
    assert pf["total_price"] == {"gold": 1, "precision": None, "recall": 0.0}


def test_store_persists_and_reloads_predictions(tmp_path):
    path = tmp_path / "preds.jsonl"
    store = Store(path)
    r = Receipt(total_price="11,000")
    store.add({"id": "test-0001", "receipt": r.model_dump(), "parse_ok": True, "schema_ok": True})
    reloaded = Store(path)
    assert "test-0001" in reloaded.rows
    assert reloaded.receipt("test-0001") == r
