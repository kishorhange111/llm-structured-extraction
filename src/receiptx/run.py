"""Run extraction configurations on a CORD split and write ``summary.json``.

    receiptx-run --split test --out results
    receiptx-run --split validation --limit 20 --configs qwen1.5b-constrained   # a quick check

Predictions are appended per receipt, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .data import load
from .metrics import flatten, paired_f1_diff, per_field, score
from .ocr import load_ocr
from .prompts import extraction, repair
from .rules import extract as rules_extract
from .schema import SUMMARY_FIELDS, Receipt
from .validate import arithmetic, check, grounding

MODELS = {
    "qwen1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
}

# Fixed before the test run, in the order they run.
CONFIGS: dict[str, dict] = {
    "rules": {"kind": "rules", "source": "ocr"},
    "qwen1.5b-free": {"kind": "llm", "model": "qwen1.5b", "constrained": False, "shot": False, "source": "ocr"},
    "qwen1.5b-constrained": {"kind": "llm", "model": "qwen1.5b", "constrained": True, "shot": False, "source": "ocr"},
    "qwen1.5b-constrained-1shot": {"kind": "llm", "model": "qwen1.5b", "constrained": True, "shot": True, "source": "ocr"},
    "qwen1.5b-constrained-1shot-repair": {"kind": "repair", "of": "qwen1.5b-constrained-1shot"},
    "qwen0.5b-free": {"kind": "llm", "model": "qwen0.5b", "constrained": False, "shot": False, "source": "ocr"},
    "qwen0.5b-constrained": {"kind": "llm", "model": "qwen0.5b", "constrained": True, "shot": False, "source": "ocr"},
    "rules-annotated": {"kind": "rules", "source": "annotated"},
    "qwen1.5b-constrained-1shot-annotated": {"kind": "llm", "model": "qwen1.5b", "constrained": True, "shot": True, "source": "annotated"},
}


def pick_example(ocr_dir: Path) -> tuple[str, str, Receipt]:
    """The shortest training receipt with 2-4 items, at least 3 summary amounts, consistent arithmetic, and every
    labelled value visible in its OCR text, so the example never shows the model a value it couldn't have read."""
    texts = load_ocr(ocr_dir / "train.jsonl")
    candidates = []
    for ex in load("train", with_images=False):
        text = texts.get(ex.id, "")
        gold = ex.gold
        filled = sum(getattr(gold, f) is not None for f in SUMMARY_FIELDS)
        if 2 <= len(gold.menu) <= 4 and filled >= 3 and not arithmetic(gold) and not grounding(gold, text):
            candidates.append((len(text), ex.id, text, gold))
    _, ex_id, text, gold = min(candidates, key=lambda c: (c[0], c[1]))
    return ex_id, text, gold


class Store:
    """Per-receipt predictions on disk, one JSON line each."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.rows[row["id"]] = row

    def add(self, row: dict) -> None:
        self.rows[row["id"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def receipt(self, ex_id: str) -> Receipt | None:
        r = self.rows[ex_id].get("receipt")
        return Receipt.model_validate(r) if r is not None else None


def run_config(name: str, examples, sources: dict[str, dict[str, str]], store: Store, stores: dict[str, Store], llms: dict, example, threads: int) -> None:
    from .llm import LocalLLM, parse

    spec = CONFIGS[name]
    if spec["kind"] == "repair":
        base = stores[spec["of"]]
        base_spec = CONFIGS[spec["of"]]
        llm = None
    else:
        base, base_spec = None, spec
    for i, ex in enumerate(examples):
        if ex.id in store.rows:
            continue
        text = sources[base_spec["source"]][ex.id]
        if spec["kind"] == "rules":
            store.add({"id": ex.id, "receipt": rules_extract(text).model_dump(), "parse_ok": True, "schema_ok": True})
            continue
        model_key = base_spec["model"]
        if model_key not in llms:
            llms.clear()  # one model in memory at a time
            llms[model_key] = LocalLLM(*MODELS[model_key], threads=threads)
        llm = llms[model_key]
        shot = example if base_spec["shot"] else None
        if spec["kind"] == "repair":
            previous = base.receipt(ex.id)
            issues = check(previous, text) if previous is not None else []
            if previous is None or not issues:
                store.add({**base.rows[ex.id], "repaired": False, "issues_before": issues})
                continue
            generation = llm.generate(repair(text, previous, issues, shot), constrained=base_spec["constrained"])
        else:
            generation = llm.generate(extraction(text, shot), constrained=spec["constrained"])
        parsed = parse(generation.text)
        row = {
            "id": ex.id,
            "receipt": parsed.receipt.model_dump() if parsed.receipt else None,
            "parse_ok": parsed.parse_ok,
            "schema_ok": parsed.schema_ok,
            "error": parsed.error,
            "raw": generation.text,
            "new_tokens": generation.new_tokens,
            "prompt_tokens": generation.prompt_tokens,
            "seconds": round(generation.seconds, 2),
        }
        if spec["kind"] == "repair":
            row.update({"repaired": True, "issues_before": issues, "previous": previous.model_dump()})
        store.add(row)
        print(f"{name} {i + 1}/{len(examples)} {ex.id} {generation.new_tokens} tok {generation.seconds:.1f}s ok={parsed.schema_ok}", flush=True)


def summarise(split: str, examples, sources, stores: dict[str, Store], example_id: str | None) -> dict:
    golds = [ex.gold for ex in examples]
    ids = [ex.id for ex in examples]
    report: dict = {"split": split, "receipts": len(examples), "one_shot_example": example_id, "configs": {}}
    preds: dict[str, list[Receipt | None]] = {}
    for name, store in stores.items():
        if not all(i in store.rows for i in ids):
            continue
        rows = [store.rows[i] for i in ids]
        preds[name] = [store.receipt(i) for i in ids]
        entry = {
            "normalised": score(preds[name], golds),
            "strict": score(preds[name], golds, normalised=False),
            "json_parse_rate": round(statistics.mean(r["parse_ok"] for r in rows), 4),
            "schema_valid_rate": round(statistics.mean(r["schema_ok"] for r in rows), 4),
            "per_field": per_field(preds[name], golds),
        }
        timed = [r for r in rows if "seconds" in r]
        if timed:
            entry["seconds_per_receipt_mean"] = round(statistics.mean(r["seconds"] for r in timed), 1)
            entry["new_tokens_mean"] = round(statistics.mean(r["new_tokens"] for r in timed), 1)
            entry["llm_calls"] = len(timed)
        report["configs"][name] = entry

    pairs = [
        ("qwen1.5b-constrained", "qwen1.5b-free"),
        ("qwen1.5b-constrained-1shot", "qwen1.5b-constrained"),
        ("qwen1.5b-constrained-1shot-repair", "qwen1.5b-constrained-1shot"),
        ("qwen0.5b-constrained", "qwen0.5b-free"),
        ("qwen1.5b-constrained", "qwen0.5b-constrained"),
        ("qwen1.5b-constrained-1shot", "rules"),
        ("qwen1.5b-constrained-1shot-annotated", "qwen1.5b-constrained-1shot"),
    ]
    report["paired_f1"] = {f"{a} - {b}": paired_f1_diff(preds[a], preds[b], golds) for a, b in pairs if a in preds and b in preds}

    # Summary amounts only, where the rule baseline competes.
    summary_only = {}
    for name, p in preds.items():
        trimmed = [Receipt(**{f: getattr(r, f) for f in SUMMARY_FIELDS}) if r is not None else None for r in p]
        summary_only[name] = score(trimmed, [Receipt(**{f: getattr(g, f) for f in SUMMARY_FIELDS}) for g in golds])["f1"]
    report["summary_fields_only_f1"] = summary_only

    # How well do label-free checks predict a wrong receipt?
    detection = {}
    for name, p in preds.items():
        if CONFIGS[name]["kind"] != "llm":
            continue
        text_source = sources[CONFIGS[name]["source"]]
        flagged, wrong = [], []
        for ex, r in zip(examples, p):
            is_wrong = r is None or flatten(r) != flatten(ex.gold)
            wrong.append(is_wrong)
            flagged.append(r is None or bool(check(r, text_source[ex.id])))
        tp = sum(f and w for f, w in zip(flagged, wrong))
        detection[name] = {
            "flagged_rate": round(sum(flagged) / len(flagged), 3),
            "wrong_rate": round(sum(wrong) / len(wrong), 3),
            "precision": round(tp / sum(flagged), 3) if sum(flagged) else None,
            "recall": round(tp / sum(wrong), 3) if sum(wrong) else None,
        }
    report["error_detection"] = detection

    repair_name = "qwen1.5b-constrained-1shot-repair"
    if repair_name in preds:
        base_name = CONFIGS[repair_name]["of"]
        better = worse = same = 0
        for ex, before, after in zip(examples, preds[base_name], preds[repair_name]):
            if not stores[repair_name].rows[ex.id].get("repaired"):
                continue
            g = flatten(ex.gold)
            b = sum((flatten(before) & g).values()) if before else 0
            a = sum((flatten(after) & g).values()) if after else 0
            better += a > b
            worse += a < b
            same += a == b
        report["repair_outcome"] = {"repaired_receipts": better + worse + same, "more_fields_correct": better, "fewer_fields_correct": worse, "unchanged": same}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--ocr", default="data/ocr")
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    examples = load(args.split, with_images=False)[: args.limit]
    ocr_dir = Path(args.ocr)
    sources = {"ocr": load_ocr(ocr_dir / f"{args.split}.jsonl"), "annotated": {ex.id: ex.annotated_text for ex in examples}}
    out = Path(args.out) / args.split
    stores = {name: Store(out / "predictions" / f"{name}.jsonl") for name in CONFIGS}
    example_id, example_text, example_gold = pick_example(ocr_dir)
    llms: dict = {}
    for name in args.configs.split(","):
        run_config(name, examples, sources, stores[name], stores, llms, (example_text, example_gold), args.threads)
        report = summarise(args.split, examples, sources, stores, example_id)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "configs"}, indent=2))


if __name__ == "__main__":
    main()
