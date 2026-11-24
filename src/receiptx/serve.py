"""Production extraction service, FastAPI wrapper, and CLI entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from .config import (
    BackendKind,
    BatchExtractRequest,
    BatchExtractResponse,
    ExtractionConfig,
    ExtractRequest,
    ExtractResponse,
    ServeConfig,
    ValidateRequest,
    ValidateResponse,
)
from .rules import extract as rules_extract
from .schema import Receipt, json_schema
from .validate import check


class ExtractionService:
    """Extracts structured receipts and flags arithmetic or grounding issues for review."""

    def __init__(
        self,
        config: ExtractionConfig | None = None,
        llm_extractor: Callable[[str], Receipt] | None = None,
    ) -> None:
        self.config = config or ExtractionConfig()
        self.llm_extractor = llm_extractor

    def extract(self, text: str, backend: BackendKind | None = None, validate_output: bool = True) -> ExtractResponse:
        active = backend or self.config.backend
        if active == "rules":
            receipt = rules_extract(text)
        elif active == "llm":
            if self.llm_extractor is None:
                raise RuntimeError("LLM backend requested but no llm_extractor is configured")
            receipt = self.llm_extractor(text)
        else:
            raise ValueError(f"unsupported backend: {active}")

        issues = check(receipt, text) if validate_output else []
        return ExtractResponse(receipt=receipt, issues=issues, needs_review=bool(issues), backend=active)


def create_app(service: ExtractionService | None = None):
    """Create a FastAPI app exposing receipt extraction, validation, and schema endpoints."""
    from fastapi import FastAPI, HTTPException

    svc = service or ExtractionService()
    app = FastAPI(title="receiptx Structured Extraction API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": svc.config.backend}

    @app.get("/schema")
    def schema() -> dict:
        return json_schema()

    @app.post("/extract", response_model=ExtractResponse)
    def extract_one(req: ExtractRequest) -> ExtractResponse:
        try:
            return svc.extract(req.text, backend=req.backend, validate_output=req.validate_output)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/extract/batch", response_model=BatchExtractResponse)
    def extract_batch(req: BatchExtractRequest) -> BatchExtractResponse:
        try:
            results = [svc.extract(r.text, backend=r.backend, validate_output=r.validate_output) for r in req.receipts]
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return BatchExtractResponse(results=results, flagged_count=sum(r.needs_review for r in results))

    @app.post("/validate", response_model=ValidateResponse)
    def validate_receipt(req: ValidateRequest) -> ValidateResponse:
        issues = check(req.receipt, req.text)
        return ValidateResponse(valid=not issues, issues=issues)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve receiptx extraction API via uvicorn.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", default="rules", choices=["rules", "llm"])
    args = parser.parse_args()

    cfg = ServeConfig(host=args.host, port=args.port, backend=args.backend)
    import uvicorn

    uvicorn.run(create_app(ExtractionService(ExtractionConfig(backend=cfg.backend))), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
