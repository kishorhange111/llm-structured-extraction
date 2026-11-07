"""The fields extracted from each receipt: line items plus the summary amounts.

Values stay strings, written as on the receipt. Field names follow CORD's own labels. Keys are ordered the way a
receipt reads, because constrained decoding emits them in schema order.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

SUMMARY_FIELDS = (
    "subtotal_price",
    "discount_price",
    "service_price",
    "tax_price",
    "total_price",
    "cashprice",
    "creditcardprice",
    "changeprice",
)
ITEM_FIELDS = ("nm", "cnt", "unitprice", "price")


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nm: str
    cnt: str | None = None
    unitprice: str | None = None
    price: str | None = None


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    menu: list[Item] = []
    subtotal_price: str | None = None
    discount_price: str | None = None
    service_price: str | None = None
    tax_price: str | None = None
    total_price: str | None = None
    cashprice: str | None = None
    creditcardprice: str | None = None
    changeprice: str | None = None


def json_schema() -> dict:
    """The schema with every key required, so constrained decoding writes ``null`` rather than choosing what to skip."""
    schema = Receipt.model_json_schema()
    schema["required"] = list(Receipt.model_fields)
    schema["$defs"]["Item"]["required"] = list(ITEM_FIELDS)
    return schema
