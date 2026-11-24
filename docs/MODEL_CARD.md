# Model Card & Reproducibility Report — `receiptx`

## 1. System Overview

- **System**: `receiptx` — Structured receipt field extraction combining RapidOCR line reconstruction, deterministic rule extraction, local LLM decoding (`Qwen2.5-1.5B-Instruct` & `Qwen2.5-0.5B-Instruct`) with optional `xgrammar` JSON-schema constraints, and label-free arithmetic + source-grounding verification.
- **Schema**: Strict Pydantic v2 `Receipt` and `Item` models (`extra="forbid"`) covering line items (`menu.nm`, `menu.cnt`, `menu.unitprice`, `menu.price`) and 8 summary fields (`subtotal_price`, `discount_price`, `service_price`, `tax_price`, `total_price`, `cashprice`, `creditcardprice`, `changeprice`).
- **Intended Use**: Automated receipt parsing with human-in-the-loop (HITL) escalation triggered by label-free validation checks (`needs_review`).

## 2. Dataset & Evaluation Protocol

- **Dataset**: CORD (`naver-clova-ix/cord-v2`, Indonesian & English retail receipts).
- **One-Shot Selection**: Deterministically selected from `train` (`train-0069`) as the shortest receipt with 2–4 line items, ≥3 summary fields, clean arithmetic, and 100% OCR grounding.
- **Metric**: Donut-style multiset micro-F1 (strict and separator-normalized) with 2,000-sample paired bootstrap 95% confidence intervals.

## 3. Observed Reproducible Benchmarks

From [`results/test/summary.json`](../results/test/summary.json) on the 100-receipt CORD `test` split:

| System | Mode | Micro-F1 (95% CI) | Precision | Recall | Parse Rate | Schema Valid | Summary-Only F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `rules` | Normalised | 0.3298 `[0.2925, 0.3729]` | 0.8509 | 0.2045 | 1.0000 | 1.0000 | 0.7370 |
| `rules` | Strict | 0.2903 `[0.2533, 0.3319]` | 0.7491 | 0.1801 | 1.0000 | 1.0000 | — |

From `run.log` (CPU smoke timing on first 6 `test` receipts, `Qwen/Qwen2.5-1.5B-Instruct`, commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`):
- `qwen1.5b-free`: 6/6 `schema_ok=True`, 69–178 generated tokens/receipt, 48.4s–137.3s/receipt on CPU.

## 4. Reproducibility Instructions

```bash
# 1. Run unit, edge-case, and API serving tests (offline)
pytest -v

# 2. Run OCR and resumable evaluation pipeline
receiptx-ocr --out data/ocr
receiptx-run --split test --out results --configs rules

# 3. Build & run serving container
docker build -t receiptx:latest .
docker run -p 8000:8000 receiptx:latest
```

## 5. Limitations & Operational Notes

- **Rule Baseline Scope**: `receiptx.rules` intentionally targets summary amounts only (high precision, 0% recall on `menu.*` line items); full line-item extraction requires the LLM backend.
- **OCR Noise**: Severe thermal-paper fading, skew, or currency formatting ambiguities can cause arithmetic checks to trigger false-positive review flags (`needs_review=True`), which is conservative by design.
- **Latency**: Local 1.5B LLM inference on CPU takes ~50–140s per receipt; production deployments should use GPU acceleration or quantized serving runtimes.
