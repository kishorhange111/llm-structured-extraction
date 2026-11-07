"""A local instruction model on CPU, decoding freely or under the receipt's JSON schema (xgrammar)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from pydantic import ValidationError

from .schema import Receipt, json_schema

Messages = list[dict[str, str]]


@dataclass
class Generation:
    text: str
    new_tokens: int
    prompt_tokens: int
    seconds: float


@dataclass
class Parsed:
    receipt: Receipt | None
    parse_ok: bool  # some JSON object could be read
    schema_ok: bool  # and it matched the schema
    error: str | None = None


def parse(text: str) -> Parsed:
    """Read the first JSON object in a reply. Numbers are accepted where strings are expected; nothing else is repaired."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return Parsed(None, False, False, "no JSON object")
    try:
        raw = json.loads(body[start : end + 1])
    except json.JSONDecodeError as e:
        return Parsed(None, False, False, f"invalid JSON: {e.msg}")

    def stringify(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, list):
            return [stringify(v) for v in value]
        if isinstance(value, dict):
            return {k: stringify(v) for k, v in value.items()}
        return value

    try:
        return Parsed(Receipt.model_validate(stringify(raw)), True, True)
    except ValidationError as e:
        return Parsed(None, True, False, f"schema: {e.errors()[0]['msg']} at {e.errors()[0]['loc']}")


class LocalLLM:
    def __init__(self, model: str, revision: str | None = None, threads: int | None = None, max_new_tokens: int = 768):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.name = model
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        # float32: most laptop CPUs emulate bfloat16 in software, which is several times slower.
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.float32).eval()
        self._grammar = None

    def grammar(self):
        if self._grammar is None:
            import xgrammar as xgr

            info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.model.config.vocab_size)
            # No free whitespace: with it allowed, small models were seen padding values with spaces and tabs.
            self._grammar = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(json_schema()), any_whitespace=False, separators=(",", ":"))
        return self._grammar

    def generate(self, messages: Messages, constrained: bool) -> Generation:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        kwargs = {}
        if constrained:
            import xgrammar as xgr

            kwargs["logits_processor"] = [xgr.contrib.hf.LogitsProcessor(self.grammar())]
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        seconds = time.perf_counter() - started
        new = output[0, inputs["input_ids"].shape[1] :]
        return Generation(self.tokenizer.decode(new, skip_special_tokens=True), int(new.shape[0]), int(inputs["input_ids"].shape[1]), seconds)
