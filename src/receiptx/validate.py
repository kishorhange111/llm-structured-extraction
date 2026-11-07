"""Checks that need no labels: do the amounts add up, and does every value appear in the text it came from?

A check failing doesn't prove the extraction wrong, and passing doesn't prove it right. ``run`` measures how well
these flags predict real errors, which decides whether they're useful for routing receipts to a person.
"""

from __future__ import annotations

import re

from .metrics import collapse, parse_amount
from .schema import ITEM_FIELDS, SUMMARY_FIELDS, Receipt


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, 0.01 * max(abs(a), abs(b)))


def arithmetic(receipt: Receipt) -> list[str]:
    issues = []
    amount = {f: parse_amount(getattr(receipt, f)) for f in SUMMARY_FIELDS}
    prices = [parse_amount(i.price) for i in receipt.menu]
    if prices and all(p is not None for p in prices):
        items = sum(prices)
        target_name = "subtotal_price" if amount["subtotal_price"] is not None else "total_price"
        target = amount[target_name]
        # Only compare with the total when nothing sits between the items and it.
        between = any(amount[f] for f in ("discount_price", "service_price", "tax_price"))
        if target is not None and (target_name == "subtotal_price" or not between) and not _close(items, target):
            issues.append(f"the line item prices add up to {items:g}, but {target_name} is {target:g}")
    sub, total = amount["subtotal_price"], amount["total_price"]
    if sub is not None and total is not None:
        extras = (amount["service_price"] or 0) - abs(amount["discount_price"] or 0)
        tax = amount["tax_price"] or 0
        # Many receipts print tax already included in the subtotal, so either reading is accepted.
        if not (_close(sub + extras + tax, total) or _close(sub + extras, total)):
            issues.append(f"subtotal_price {sub:g} with tax, service and discount doesn't give total_price {total:g}")
    cash, change = amount["cashprice"], amount["changeprice"]
    if cash is not None and change is not None and total is not None and not _close(cash - total, change):
        issues.append(f"cashprice {cash:g} minus total_price {total:g} isn't changeprice {change:g}")
    return issues


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def grounding(receipt: Receipt, source: str) -> list[str]:
    """Every value must appear in the source: amounts by their digits, names as text (case and spacing ignored)."""
    text = collapse(source).casefold()
    digit_runs = {_digits(tok) for tok in re.findall(r"[\d][\d.,]*", source)}
    issues = []
    values = [(f"menu.{f}", getattr(item, f)) for item in receipt.menu for f in ITEM_FIELDS]
    values += [(f, getattr(receipt, f)) for f in SUMMARY_FIELDS]
    for field, value in values:
        if value is None or not collapse(value):
            continue
        if field.endswith("nm"):
            if collapse(value).casefold() not in text:
                issues.append(f'{field} "{value}" does not appear in the receipt text')
        elif _digits(value) and _digits(value) not in digit_runs and not any(_digits(value) in run for run in digit_runs):
            issues.append(f'{field} "{value}" does not appear in the receipt text')
    return issues


def check(receipt: Receipt, source: str) -> list[str]:
    return arithmetic(receipt) + grounding(receipt, source)
