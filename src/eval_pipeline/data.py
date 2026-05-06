from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    sample_id: str
    question: str
    contexts: list[str]
    reference_answer: str
    dataset: str


def _extract_sample(raw: dict, fallback_id: int) -> Sample:
    question = str(raw.get("question") or raw.get("input") or raw.get("query") or "").strip()
    context = raw.get("contexts")
    if context is None:
        context = raw.get("context") or raw.get("passages") or []
    if isinstance(context, str):
        contexts = [context]
    elif isinstance(context, list):
        contexts = [str(x) for x in context if str(x).strip()]
    else:
        contexts = [str(context)]

    answers = raw.get("answers")
    if isinstance(answers, list) and answers:
        reference = str(answers[0])
    else:
        reference = str(raw.get("reference_answer") or raw.get("answer") or "")

    sid = str(raw.get("_id") or raw.get("id") or f"sample_{fallback_id}")
    return Sample(
        sample_id=sid,
        question=question,
        contexts=contexts,
        reference_answer=reference,
        dataset=str(raw.get("dataset") or ""),
    )


def load_samples(path: str, max_samples: int) -> list[Sample]:
    p = Path(path)
    if p.is_file():
        if p.suffix == ".jsonl":
            rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            obj = json.loads(p.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else [obj]
    else:
        rows = [
            {
                "_id": "demo_0",
                "input": "What is the main contribution of the paper?",
                "context": "The method combines compression and eviction with adaptive gating.",
                "answers": ["Adaptive fusion of compression and eviction."],
                "dataset": "demo",
            }
        ]
    samples = [_extract_sample(r, i) for i, r in enumerate(rows)]
    samples = [s for s in samples if s.question and s.contexts]
    return samples[:max_samples]

