from __future__ import annotations

import math
import re
from dataclasses import dataclass


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _split_sentences(text: str) -> list[str]:
    blocks = re.split(r"(?<=[\.\?\!。！？])\s+|\n+", text.strip())
    return [b.strip() for b in blocks if b.strip()]


def _chunk_text(text: str, max_chars: int = 1400) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
            continue
        if len(current) + len(para) + 2 <= max_chars:
            current += "\n\n" + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks or [text]


def _lexical_overlap_score(question_tokens: set[str], text: str) -> float:
    tokens = set(_word_tokens(text))
    if not question_tokens:
        return 0.0
    return len(tokens & question_tokens) / max(1.0, len(question_tokens))


def _compress_chunk_by_question(text: str, question: str, keep_ratio: float) -> str:
    sents = _split_sentences(text)
    if len(sents) <= 1:
        return text
    qset = set(_word_tokens(question))
    ranked = sorted(
        [(_lexical_overlap_score(qset, sent), idx, sent) for idx, sent in enumerate(sents)],
        key=lambda x: (x[0], -x[1]),
        reverse=True,
    )
    keep_n = max(1, int(math.ceil(len(sents) * keep_ratio)))
    kept_idx = sorted([x[1] for x in ranked[:keep_n]])
    return " ".join([sents[i] for i in kept_idx])


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class PolicyResult:
    method_context: str
    detail: dict


def apply_method(method: str, question: str, contexts: list[str]) -> PolicyResult:
    method_key = (method or "").strip().lower()
    all_chunks: list[str] = []
    for ctx in contexts:
        all_chunks.extend(_chunk_text(ctx))
    if not all_chunks:
        all_chunks = [""]

    qset = set(_word_tokens(question))
    sink_keep = 1
    sink_chunks = all_chunks[:sink_keep]
    rest_chunks = all_chunks[sink_keep:]
    scored_rest: list[tuple[float, int, str]] = []
    for idx, chunk in enumerate(rest_chunks):
        overlap = _lexical_overlap_score(qset, chunk)
        recency = (idx + 1) / max(1, len(rest_chunks))
        score = 0.8 * overlap + 0.2 * recency
        scored_rest.append((score, idx, chunk))

    selected: list[str]
    detail = {"sink_kept": sink_keep, "total_chunks": len(all_chunks)}
    full_tokens_est = max(1, len(_word_tokens("\n\n".join(all_chunks))))

    if method_key == "full_kv":
        selected = list(all_chunks)
        detail.update({"eviction_rate": 0.0, "compression_rate": 0.0})
    elif method_key == "h2o":
        keep_n = max(1, int(math.ceil(len(rest_chunks) * 0.5)))
        top = sorted(scored_rest, key=lambda x: x[0], reverse=True)[:keep_n]
        selected = sink_chunks + [x[2] for x in sorted(top, key=lambda x: x[1])]
        detail.update({"eviction_rate": 0.5, "compression_rate": 0.0})
    elif method_key == "streamingllm":
        window_n = max(1, int(math.ceil(len(rest_chunks) * 0.4)))
        selected = sink_chunks + rest_chunks[-window_n:]
        detail.update({"eviction_rate": 0.6, "compression_rate": 0.0})
    elif method_key == "compllm_style":
        base = sink_chunks + rest_chunks
        selected = [_compress_chunk_by_question(c, question, keep_ratio=0.65) for c in base]
        detail.update({"eviction_rate": 0.0, "compression_rate": 0.35})
    elif method_key in ("ahec", "ours_hybrid"):
        pressure = _clamp((full_tokens_est - 1500) / 6000.0, 0.0, 1.0)
        eviction_rate = 0.15 + 0.35 * pressure
        compression_rate = 0.10 + 0.45 * pressure
        keep_n = max(1, int(math.ceil(len(rest_chunks) * (1.0 - eviction_rate))))
        top = sorted(scored_rest, key=lambda x: x[0], reverse=True)[:keep_n]
        kept = [x[2] for x in sorted(top, key=lambda x: x[1])]
        kept = [_compress_chunk_by_question(c, question, keep_ratio=1.0 - compression_rate) for c in kept]
        selected = sink_chunks + kept
        detail.update(
            {
                "eviction_rate": round(eviction_rate, 4),
                "compression_rate": round(compression_rate, 4),
                "adaptive_pressure": round(pressure, 4),
                "ahec_policy": "sink_preserve + importance_eviction + question_aware_compression",
            }
        )
    else:
        selected = list(all_chunks)
        detail.update({"eviction_rate": 0.0, "compression_rate": 0.0})

    method_context = "\n\n".join([c for c in selected if c.strip()])
    method_tokens_est = max(1, len(_word_tokens(method_context)))
    detail.update(
        {
            "selected_chunks": len(selected),
            "full_tokens_est": full_tokens_est,
            "method_tokens_est": method_tokens_est,
            "kv_token_reduction_ratio": round(1.0 - (method_tokens_est / max(1, full_tokens_est)), 6),
        }
    )
    return PolicyResult(method_context=method_context, detail=detail)
