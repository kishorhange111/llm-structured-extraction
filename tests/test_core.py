import pytest

from receiptx.data import annotated_text, gold_receipt
from receiptx.llm import parse
from receiptx.metrics import counts, flatten, normalise, paired_f1_diff, parse_amount, parse_count, score
from receiptx.ocr import to_lines
from receiptx.prompts import extraction, repair
from receiptx.rules import extract
from receiptx.schema import Item, Receipt, json_schema
from receiptx.validate import arithmetic, check, grounding


@pytest.mark.parametrize(
    ("text", "value"),
    [("11,000", 11000), ("60.000", 60000), ("1.234.567", 1234567), ("25.50", 25.5), ("Rp 45,500", 45500), ("-5.455", 5455), ("@11000", 11000), ("12,000.00", 12000), ("abc", None), (None, None)],
)
def test_parse_amount(text, value):
    assert parse_amount(text) == value


def test_parse_count():
    assert parse_count("1X") == 1 and parse_count("2.00") == 2 and parse_count("x3") == 3 and parse_count("-") is None


def test_normalise_makes_separator_styles_agree():
    assert normalise("price", "11,000") == normalise("price", "11.000") == normalise("price", "11000")
    assert normalise("nm", "Donat  GULA ") == "donat gula"
    assert normalise("cnt", "1X") == normalise("cnt", "1")


def receipt(**kw):
    return Receipt(**kw)


GOLD = receipt(menu=[Item(nm="DONAT GULA", cnt="1", price="11,000"), Item(nm="TEA", cnt="2", price="8,000")], total_price="19,000", cashprice="20,000", changeprice="1,000")


def test_flatten_ignores_item_order_and_nulls():
    swapped = receipt(menu=list(reversed(GOLD.menu)), total_price="19,000", cashprice="20,000", changeprice="1,000")
    assert flatten(swapped) == flatten(GOLD)
    assert ("menu.unitprice", None) not in flatten(GOLD)


def test_counts_and_micro_f1():
    pred = receipt(menu=[Item(nm="DONAT GULA", cnt="1", price="11.000")], total_price="19,000", cashprice="2,000")
    tp, npred, ngold = counts(pred, GOLD)
    assert (tp, npred, ngold) == (4, 5, 9)
    assert counts(pred, GOLD, normalised=False)[0] == 3  # "11.000" only matches once separators are normalised
    s = score([pred, None], [GOLD, GOLD], n_boot=50)
    assert s["precision"] == pytest.approx(4 / 5, abs=1e-4) and s["recall"] == pytest.approx(4 / 18, abs=1e-4)
    assert s["receipts_fully_correct"] == 0


def test_perfect_prediction_scores_one():
    s = score([GOLD], [GOLD], n_boot=20)
    assert s["f1"] == 1 and s["receipts_fully_correct"] == 1


def test_paired_diff_of_identical_systems_is_zero():
    d = paired_f1_diff([GOLD, None], [GOLD, None], [GOLD, GOLD], n_boot=50)
    assert d == {"diff": 0.0, "low": 0.0, "high": 0.0}


def test_gold_receipt_accepts_a_single_item_and_drops_sub_items():
    g = gold_receipt(
        {
            "menu": {"nm": "-TICKET CP", "num": "901016", "cnt": "2", "price": "60.000", "sub": {"nm": "x"}},
            "sub_total": {"subtotal_price": "60.000", "tax_price": "5.455"},
            "total": {"total_price": "60.000", "menuqty_cnt": "2.00"},
        }
    )
    assert g.menu == [Item(nm="-TICKET CP", cnt="2", unitprice=None, price="60.000")]
    assert g.subtotal_price == "60.000" and g.tax_price == "5.455" and g.total_price == "60.000"


def test_gold_receipt_joins_list_values():
    g = gold_receipt({"menu": [{"nm": ["ICE", "TEA"], "price": "5,000"}], "total": [{"total_price": "5,000"}, {"total_price": "9"}]})
    assert g.menu[0].nm == "ICE TEA" and g.total_price == "5,000"


def test_annotated_text_orders_rows_and_words():
    quad = lambda x, y: {"x1": x, "x3": x + 10, "y1": y, "y3": y + 10}  # noqa: E731
    lines = [
        {"words": [{"quad": quad(50, 100), "row_id": 2, "text": "11,000"}, {"quad": quad(0, 100), "row_id": 2, "text": "TOTAL"}]},
        {"words": [{"quad": quad(0, 10), "row_id": 1, "text": "TEA"}]},
    ]
    assert annotated_text(lines) == "TEA\nTOTAL 11,000"


def test_to_lines_groups_boxes_by_height():
    box = lambda x, y: [[x, y], [x + 40, y], [x + 40, y + 20], [x, y + 20]]  # noqa: E731
    assert to_lines([box(300, 102), box(10, 100), box(10, 140)], ["11,000", "TOTAL", "CASH"]) == "TOTAL  11,000\nCASH"


def test_rules_read_keyword_lines():
    text = "DONAT GULA 11,000\nSUBTOTAL 11,000\nTOTAL QTY 1\nTOTAL\n11,000\nCASH 20,000\nCHANGE 9,000"
    r = extract(text)
    assert r.subtotal_price == "11,000" and r.total_price == "11,000"
    assert r.cashprice == "20,000" and r.changeprice == "9,000"
    assert r.menu == []


def test_arithmetic_accepts_tax_inclusive_totals_and_flags_bad_change():
    inclusive = receipt(menu=[Item(nm="A", price="60.000")], subtotal_price="60.000", tax_price="5.455", total_price="60.000")
    assert arithmetic(inclusive) == []
    bad = receipt(menu=[Item(nm="A", price="10,000")], total_price="10,000", cashprice="20,000", changeprice="5,000")
    assert len(arithmetic(bad)) == 1
    wrong_items = receipt(menu=[Item(nm="A", price="10,000"), Item(nm="B", price="3,000")], subtotal_price="10,000")
    assert "add up to 13000" in arithmetic(wrong_items)[0]


def test_grounding_flags_values_missing_from_the_text():
    text = "DONAT GULA 11,000\nTOTAL 11.000"
    ok = receipt(menu=[Item(nm="Donat Gula", price="11000")], total_price="11,000")
    assert grounding(ok, text) == []
    invented = receipt(menu=[Item(nm="ROTI", price="11,000")], total_price="12,000")
    assert len(grounding(invented, text)) == 2
    assert check(invented, text)


def test_parse_reads_fenced_json_and_numbers():
    p = parse('Here it is:\n```json\n{"menu":[{"nm":"TEA","cnt":1,"price":5000}],"total_price":5000}\n```')
    assert p.schema_ok and p.receipt.menu[0].price == "5000" and p.receipt.total_price == "5000"


def test_parse_rejects_wrong_keys_and_broken_json():
    wrong = parse('{"items":[{"item_name":"TEA"}],"total":5}')
    assert wrong.parse_ok and not wrong.schema_ok and wrong.receipt is None
    broken = parse('{"menu": [')
    assert not broken.parse_ok


def test_schema_requires_every_key():
    s = json_schema()
    assert s["required"][0] == "menu" and "changeprice" in s["required"]
    assert s["$defs"]["Item"]["required"] == ["nm", "cnt", "unitprice", "price"]


def test_prompts_include_example_and_issues():
    msgs = extraction("TOTAL 5", ("TOTAL 9", GOLD))
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    fix = repair("TOTAL 5", GOLD, ["cash minus total isn't change"])
    assert fix[-2]["role"] == "assistant" and "cash minus total" in fix[-1]["content"]
