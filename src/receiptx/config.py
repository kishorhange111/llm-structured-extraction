"""Typed configuration and API schemas for receipt extraction serving."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema import Receipt

BackendKind = Literal["rules", "llm"]


class ExtractionConfig(BaseModel):
    """Runtime configuration for receipt extraction and label-free verification."""

    model_config = ConfigDict(extra="forbid")

    backend: BackendKind = "rules"
    model_key: Literal["qwen1.5b", "qwen0.5b"] = "qwen1.5b"
    constrained: bool = True
    enable_repair: bool = True
    max_repair_rounds: int = Field(default=1, ge=0, le=3)


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    backend: BackendKind | None = None
    validate_output: bool = True


class ExtractResponse(BaseModel):
    receipt: Receipt
    issues: list[str]
    needs_review: bool
    backend: BackendKind


class BatchExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipts: list[ExtractRequest] = Field(min_length=1, max_length=100)


class BatchExtractResponse(BaseModel):
    results: list[ExtractResponse]
    flagged_count: int


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    receipt: Receipt


class ValidateResponse(BaseModel):
    valid: bool
    issues: list[str]


class ServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    backend: BackendKind = "rules"
