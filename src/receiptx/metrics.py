"""Field-level scoring.

Each receipt is flattened to a multiset of (field, value) pairs, line-item fields included without their position, and
scored with micro-averaged F1 over all receipts, the way Donut (Kim et al., 2022) scores CORD.

Two value comparisons are reported:
- strict: the string as printed, whitespace collapsed;
- normalised: amounts compared as numbers (so "11,000", "11.000" and "11000" agree, and sign is ignored), counts by
  their leading number, and names case-insensitively.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .schema import ITEM_FIELDS, SUMMARY_FIELDS, Receipt

AMOUNT_FIELDS = {"unitprice", "price", *SUMMARY_FIELDS}


def parse_amount(text: str | None) -> float | None:
    """Read an amount as printed on an Indonesian or English receipt.

    A final separator followed by one or two digits is a decimal point; any other "." or "," separates thousands.
    """
    if text is None:
        return None
    kept = re.sub(r"[^\d.,]", "", text)
    if not re.search(r"\d", kept):
        return None
    kept = kept.strip(".,")
    decimal = re.search(r"[.,](\d{1,2})$", kept)
    if decimal:
        whole = re.sub(r"[.,]", "", kept[: decimal.start()]) or "0"
        return float(f"{whole}.{decimal.group(1)}")
    return float(re.sub(r"[.,]", "", kept))


def parse_count(text: str | None) -> float | None:
    if text is None:
        return None
    match = re.search(r"\d+(?:[.,]\d+)?", text)
    return float(match.group().replace(",", ".")) if match else None


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalise(field: str, value: str) -> str:
    if field in AMOUNT_FIELDS:
        amount = parse_amount(value)
        return f"{abs(amount):.2f}" if amount is not None else collapse(value).casefold()
    if field == "cnt":
        count = parse_count(value)
        return f"{count:g}" if count is not None else collapse(value).casefold()
    return collapse(value).casefold()


def flatten(receipt: Receipt, normalised: bool = True) -> Counter:
    pairs: Counter = Counter()

    def add(field: str, value: str | None, key: str) -> None:
        if value is None or not collapse(value):
            return
        pairs[(key, normalise(field, value) if normalised else collapse(value))] += 1

    for item in receipt.menu:
        for field in ITEM_FIELDS:
            add(field, getattr(item, field), f"menu.{field}")
    for field in SUMMARY_FIELDS:
        add(field, getattr(receipt, field), field)
    return pairs


def counts(pred: Receipt | None, gold: Receipt, normalised: bool = True) -> tuple[int, int, int]:
    """True positives, predicted pairs and gold pairs for one receipt. A missing prediction scores nothing."""
    g = flatten(gold, normalised)
    p = flatten(pred, normalised) if pred is not None else Counter()
    return sum((p & g).values()), sum(p.values()), sum(g.values())


def f1_from(rows: np.ndarray) -> float:
    tp, npred, ngold = rows.sum(axis=0)
    precision = tp / npred if npred else 0.0
    recall = tp / ngold if ngold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(preds: list[Receipt | None], golds: list[Receipt], normalised: bool = True, n_boot: int = 2000, seed: int = 0) -> dict:
    rows = np.array([counts(p, g, normalised) for p, g in zip(preds, golds)], dtype=float)
    rng = np.random.default_rng(seed)
    boots = [f1_from(rows[rng.integers(0, len(rows), len(rows))]) for _ in range(n_boot)]
    tp, npred, ngold = rows.sum(axis=0)
    exact = [flatten(p, normalised) == flatten(g, normalised) if p is not None else False for p, g in zip(preds, golds)]
    return {
        "f1": round(f1_from(rows), 4),
        "f1_low": round(float(np.percentile(boots, 2.5)), 4),
        "f1_high": round(float(np.percentile(boots, 97.5)), 4),
        "precision": round(tp / npred, 4) if npred else 0.0,
        "recall": round(tp / ngold, 4) if ngold else 0.0,
        "receipts_fully_correct": round(float(np.mean(exact)), 4),
    }


def paired_f1_diff(a: list[Receipt | None], b: list[Receipt | None], golds: list[Receipt], n_boot: int = 2000, seed: int = 0) -> dict:
    """F1(a) - F1(b) on the same receipts, with a 95 % interval from resampling receipts."""
    ra = np.array([counts(p, g) for p, g in zip(a, golds)], dtype=float)
    rb = np.array([counts(p, g) for p, g in zip(b, golds)], dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(golds), len(golds))
        diffs.append(f1_from(ra[idx]) - f1_from(rb[idx]))
    return {"diff": round(f1_from(ra) - f1_from(rb), 4), "low": round(float(np.percentile(diffs, 2.5)), 4), "high": round(float(np.percentile(diffs, 97.5)), 4)}


def per_field(preds: list[Receipt | None], golds: list[Receipt]) -> dict:
    """Normalised precision and recall for each field, items pooled."""
    out = {}
    for field in [f"menu.{f}" for f in ITEM_FIELDS] + list(SUMMARY_FIELDS):
        tp = npred = ngold = 0
        for p, g in zip(preds, golds):
            gp = Counter({k: v for k, v in flatten(g).items() if k[0] == field})
            pp = Counter({k: v for k, v in flatten(p).items() if k[0] == field}) if p is not None else Counter()
            tp += sum((gp & pp).values())
            npred += sum(pp.values())
            ngold += sum(gp.values())
        out[field] = {"gold": ngold, "precision": round(tp / npred, 3) if npred else None, "recall": round(tp / ngold, 3) if ngold else None}
    return out
