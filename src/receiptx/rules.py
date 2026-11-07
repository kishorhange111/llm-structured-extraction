"""Rule-based baseline: keyword lines for the summary amounts. It doesn't attempt line items.

The keywords were written by reading validation receipts only.
"""

from __future__ import annotations

import re

from .schema import Receipt

# Checked in order on each line; the first match claims the line. Subtotal must come before total.
KEYWORDS: list[tuple[str, str]] = [
    ("subtotal_price", r"sub\s*-?\s*t(?:o)?t(?:a)?l|subttl"),
    ("discount_price", r"\bdisc|diskon|potongan"),
    ("service_price", r"\bservice|\bsvc|\bsrv|\bsc\b"),
    ("tax_price", r"\btax\b|\bppn\b|\bpb\s*1\b|pajak|\bvat\b"),
    ("changeprice", r"\bchange\b|kembali"),
    ("cashprice", r"\bcash\b|tunai"),
    ("creditcardprice", r"\bcard\b|credit|debit|\bvisa\b|master|\bbca\b|\bedc\b|\bbni\b|\bbri\b|mandiri"),
    ("total_price", r"grand\s*total|\btotal\b|\bttl\b|jumlah|\bamount\b"),
]
EXCLUDE = re.compile(r"\bqty\b|\bitems?\b|total\s+item", re.I)
AMOUNT = re.compile(r"-?\(?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?\)?|-?\d{3,}(?:[.,]\d{1,2})?")


def last_amount(line: str) -> str | None:
    found = AMOUNT.findall(line)
    return found[-1] if found else None


def extract(text: str) -> Receipt:
    lines = [line for line in text.splitlines() if line.strip()]
    found: dict[str, str] = {}
    for i, line in enumerate(lines):
        low = line.lower()
        if EXCLUDE.search(low):
            continue
        for field, pattern in KEYWORDS:
            if re.search(pattern, low):
                if field in found:
                    break
                value = last_amount(line)
                # A label alone on its line usually has its amount on the next line.
                if value is None and i + 1 < len(lines) and not any(re.search(p, lines[i + 1].lower()) for _, p in KEYWORDS):
                    value = last_amount(lines[i + 1])
                if value is not None:
                    found[field] = value
                break
    return Receipt(**found)
