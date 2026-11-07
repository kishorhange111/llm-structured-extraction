"""Prompts. Fixed before the test run; the one-shot example comes from the training split."""

from __future__ import annotations

from .schema import Receipt

INSTRUCTIONS = """You read the OCR text of a shop or restaurant receipt and return its fields as JSON.

Fields:
- menu: the purchased line items, in order. Each has nm (item name), cnt (quantity as printed), unitprice (price of one unit, if printed) and price (the line's amount).
- subtotal_price, discount_price, service_price, tax_price, total_price: the summary amounts.
- cashprice (cash handed over), creditcardprice (amount paid by card), changeprice (change given back).

Rules:
- Copy every value exactly as printed, including separators such as "11,000" or "60.000".
- Use null for any field that is not on the receipt. Never calculate or invent a value.
- Reply with one JSON object on a single line and nothing else, with exactly these keys:
{"menu":[{"nm":...,"cnt":...,"unitprice":...,"price":...}],"subtotal_price":...,"discount_price":...,"service_price":...,"tax_price":...,"total_price":...,"cashprice":...,"creditcardprice":...,"changeprice":...}"""


def as_json(receipt: Receipt) -> str:
    return receipt.model_dump_json()


def extraction(text: str, example: tuple[str, Receipt] | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": INSTRUCTIONS}]
    if example is not None:
        messages += [
            {"role": "user", "content": f"Receipt text:\n{example[0]}"},
            {"role": "assistant", "content": as_json(example[1])},
        ]
    messages.append({"role": "user", "content": f"Receipt text:\n{text}"})
    return messages


def repair(text: str, previous: Receipt, issues: list[str], example: tuple[str, Receipt] | None = None) -> list[dict[str, str]]:
    """A second turn showing the model its own answer and the checks it failed."""
    messages = extraction(text, example)
    problems = "\n".join(f"- {i}" for i in issues)
    messages += [
        {"role": "assistant", "content": as_json(previous)},
        {
            "role": "user",
            "content": "Automatic checks found these problems with your answer:\n"
            f"{problems}\n\n"
            "Re-read the receipt text and return the corrected JSON. Change a value only if the text supports the change; "
            "if a check fails because the receipt itself doesn't add up, keep the printed values.",
        },
    ]
    return messages
