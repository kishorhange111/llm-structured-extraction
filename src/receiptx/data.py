"""CORD v2 (Park et al., 2019): Indonesian shop and restaurant receipts with field labels. CC BY 4.0."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import ITEM_FIELDS, SUMMARY_FIELDS, Item, Receipt

REPO = "naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
FILES = {
    "train": "data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
    "validation": "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    "test": "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
}


@dataclass
class Example:
    id: str
    image: bytes | None
    gold: Receipt
    annotated_text: str  # the labelled words only, laid out by row: an upper bound with no OCR noise


def _text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def gold_receipt(gt_parse: dict) -> Receipt:
    """Project CORD's labels onto the schema. Sub-items (``menu.sub``) and rare fields are dropped."""
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    items = [Item(**{f: _text(entry.get(f)) for f in ITEM_FIELDS if f != "nm"}, nm=_text(entry.get("nm")) or "") for entry in menu]
    items = [i for i in items if i.nm or i.price]
    summary = {}
    for group in ("sub_total", "total"):
        block = gt_parse.get(group) or {}
        if isinstance(block, list):  # a handful of receipts repeat a block; keep the first value of each field
            merged: dict = {}
            for b in block:
                for k, v in b.items():
                    merged.setdefault(k, v)
            block = merged
        for field in SUMMARY_FIELDS:
            if field in block:
                summary[field] = _text(block[field])
    return Receipt(menu=items, **summary)


def annotated_text(valid_line: list[dict]) -> str:
    """Rebuild the labelled words as lines: words sharing a row id, rows top to bottom, words left to right."""
    rows: dict[int, list[tuple[float, float, str]]] = {}
    for line in valid_line:
        for word in line["words"]:
            q = word["quad"]
            rows.setdefault(word["row_id"], []).append(((q["y1"] + q["y3"]) / 2, q["x1"], word["text"]))
    ordered = sorted(rows.values(), key=lambda words: sum(w[0] for w in words) / len(words))
    return "\n".join(" ".join(w[2] for w in sorted(words, key=lambda w: w[1])) for words in ordered)


def load(split: str, with_images: bool = True) -> list[Example]:
    """``split`` is train, validation or test. Train is limited to its first shard (about 200 receipts)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    columns = ["image", "ground_truth"] if with_images else ["ground_truth"]
    frame = pd.read_parquet(hf_hub_download(REPO, FILES[split], repo_type="dataset", revision=REVISION), columns=columns)
    out = []
    for i, row in frame.iterrows():
        truth = json.loads(row["ground_truth"])
        out.append(
            Example(
                id=f"{split}-{i:04d}",
                image=row["image"]["bytes"] if with_images else None,
                gold=gold_receipt(truth["gt_parse"]),
                annotated_text=annotated_text(truth["valid_line"]),
            )
        )
    return out
