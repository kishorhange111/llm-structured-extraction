"""Real OCR with RapidOCR (PaddleOCR models on ONNX Runtime), laid out as text lines.

    receiptx-ocr --split test --out data/ocr

The extractor only ever sees this text, never the image or CORD's labelled words.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path


def to_lines(boxes: list, texts: list[str]) -> str:
    """Group boxes whose vertical centres fall within half a line height of each other, then read left to right."""
    words = []
    for box, text in zip(boxes, texts):
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        words.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs), text))
    words.sort()
    lines: list[list[tuple]] = []
    for word in words:
        if lines:
            last = lines[-1]
            centre = sum(w[0] for w in last) / len(last)
            height = sum(w[1] for w in last) / len(last)
            if abs(word[0] - centre) <= height / 2:
                last.append(word)
                continue
        lines.append([word])
    return "\n".join("  ".join(w[3] for w in sorted(line, key=lambda w: w[2])) for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="data/ocr")
    args = parser.parse_args()

    from PIL import Image
    from rapidocr import RapidOCR

    from .data import load

    engine = RapidOCR()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}.jsonl"
    seconds = []
    with path.open("w", encoding="utf-8") as f:
        for example in load(args.split):
            image = Image.open(io.BytesIO(example.image)).convert("RGB")
            started = time.perf_counter()
            result = engine(image)
            seconds.append(time.perf_counter() - started)
            text = to_lines(result.boxes.tolist(), list(result.txts)) if result.txts else ""
            f.write(json.dumps({"id": example.id, "text": text}, ensure_ascii=False) + "\n")
    print(f"{len(seconds)} receipts, mean {sum(seconds) / len(seconds):.2f} s per image -> {path}")


def load_ocr(path: Path) -> dict[str, str]:
    return {r["id"]: r["text"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


if __name__ == "__main__":
    main()
