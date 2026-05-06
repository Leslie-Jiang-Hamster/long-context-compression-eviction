from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_community.chat_models import MiniMaxChat
from pydantic import BaseModel, Field


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_minimax_chat_base_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    if url.endswith("/v1/text/chatcompletion_v2"):
        return url
    if url.endswith("/v1"):
        return url + "/text/chatcompletion_v2"
    return url


def _extract_json_from_text(text: str) -> dict:
    clean = text.strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean).rstrip("`").strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(clean[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Judge output is not JSON: {text[:400]}")


def _parse_scores(text: str) -> dict:
    try:
        parsed = _extract_json_from_text(text)
    except Exception:
        lower = text.lower()

        def pick(name: str) -> float:
            m = re.search(rf"{name}\s*[:=]\s*([01](?:\.\d+)?)", lower)
            return float(m.group(1)) if m else 0.0

        parsed = {
            "faithfulness": pick("faithfulness"),
            "answer_relevancy": pick("answer_relevancy"),
            "context_precision": pick("context_precision"),
            "rationale_brief": "parsed_from_non_json_text",
        }
    return {
        "faithfulness": _clamp(float(parsed.get("faithfulness", 0.0)), 0.0, 1.0),
        "answer_relevancy": _clamp(float(parsed.get("answer_relevancy", 0.0)), 0.0, 1.0),
        "context_precision": _clamp(float(parsed.get("context_precision", 0.0)), 0.0, 1.0),
        "rationale_brief": str(parsed.get("rationale_brief", "")),
    }


@dataclass
class JudgeResult:
    score: dict
    response_metadata: dict[str, Any]


class _JudgeSchema(BaseModel):
    faithfulness: float = Field(description="0 to 1")
    answer_relevancy: float = Field(description="0 to 1")
    context_precision: float = Field(description="0 to 1")
    rationale_brief: str = Field(description="short rationale")


class MiniMaxJudge:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float, max_tokens: int) -> None:
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": _normalize_minimax_chat_base_url(base_url),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        import os

        group_id = os.getenv("MINIMAX_GROUP_ID", "").strip()
        if group_id:
            kwargs["group_id"] = group_id
        self.llm = MiniMaxChat(**kwargs)

    @staticmethod
    def _messages(question: str, context: str, answer: str, reference: str) -> list[dict]:
        rubric = (
            "Score each metric from 0 to 1.\n"
            "- faithfulness: answer is supported by provided context; penalize hallucinations.\n"
            "- answer_relevancy: answer directly addresses the question.\n"
            "- context_precision: used evidence is relevant; avoid irrelevant context dependence.\n"
            "Return JSON only with keys: faithfulness, answer_relevancy, context_precision, rationale_brief."
        )
        return [
            {"role": "system", "content": "You are a strict evaluator. Output JSON only."},
            {
                "role": "user",
                "content": (
                    f"{rubric}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}\n\n"
                    f"Model Answer:\n{answer}\n\n"
                    f"Reference Answer (optional):\n{reference}"
                ),
            },
        ]

    def judge_once(self, question: str, context: str, answer: str, reference: str) -> JudgeResult:
        # strict: no backend fallback
        structured = self.llm.with_structured_output(_JudgeSchema, include_raw=True)
        result = structured.invoke(self._messages(question, context, answer, reference))
        parsed = result.get("parsed") if isinstance(result, dict) else None
        raw = result.get("raw") if isinstance(result, dict) else None
        if parsed is not None:
            if isinstance(parsed, dict):
                d = parsed
            else:
                d = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()
            score = {
                "faithfulness": _clamp(float(d.get("faithfulness", 0.0)), 0.0, 1.0),
                "answer_relevancy": _clamp(float(d.get("answer_relevancy", 0.0)), 0.0, 1.0),
                "context_precision": _clamp(float(d.get("context_precision", 0.0)), 0.0, 1.0),
                "rationale_brief": str(d.get("rationale_brief", "")),
            }
            return JudgeResult(score=score, response_metadata=getattr(raw, "response_metadata", {}) or {})

        raw_text = str(getattr(raw, "content", "")).strip() if raw is not None else ""
        if raw_text:
            return JudgeResult(score=_parse_scores(raw_text), response_metadata=getattr(raw, "response_metadata", {}) or {})
        raise RuntimeError(f"MiniMax judge returned no parsable output: {repr(result)[:700]}")

