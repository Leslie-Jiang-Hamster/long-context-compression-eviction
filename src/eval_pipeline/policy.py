from __future__ import annotations

import math
import os
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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

    score_overlap_weight = _env_float("AHEC_SCORE_OVERLAP_WEIGHT", 0.8)
    score_recency_weight = _env_float("AHEC_SCORE_RECENCY_WEIGHT", 0.2)
    weight_sum = score_overlap_weight + score_recency_weight
    if weight_sum <= 1e-9:
        score_overlap_weight, score_recency_weight = 0.8, 0.2
    else:
        score_overlap_weight /= weight_sum
        score_recency_weight /= weight_sum

    def _score_chunks(chunks: list[str]) -> list[tuple[float, int, str]]:
        scored: list[tuple[float, int, str]] = []
        for idx, chunk in enumerate(chunks):
            overlap = _lexical_overlap_score(qset, chunk)
            recency = (idx + 1) / max(1, len(chunks))
            score = score_overlap_weight * overlap + score_recency_weight * recency
            scored.append((score, idx, chunk))
        return scored

    scored_rest = _score_chunks(rest_chunks)

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
    elif method_key in ("ahec", "ours_hybrid", "ahec_wo_adaptive_gate", "ahec_wo_sink_preserve", "ahec_wo_compression", "ahec_wo_eviction"):
        pressure_offset_tokens = _env_float("AHEC_PRESSURE_OFFSET_TOKENS", 1500.0)
        pressure_scale_tokens = max(_env_float("AHEC_PRESSURE_SCALE_TOKENS", 6000.0), 1.0)
        evict_base = _env_float("AHEC_EVICT_BASE", 0.15)
        evict_span = _env_float("AHEC_EVICT_SPAN", 0.35)
        compress_base = _env_float("AHEC_COMPRESS_BASE", 0.10)
        compress_span = _env_float("AHEC_COMPRESS_SPAN", 0.45)
        sink_keep_env = max(0, _env_int("AHEC_SINK_KEEP", 1))

        pressure = _clamp((full_tokens_est - pressure_offset_tokens) / pressure_scale_tokens, 0.0, 1.0)
        if method_key == "ahec_wo_adaptive_gate":
            # Disable adaptive gate: use fixed rates.
            eviction_rate = 0.30
            compression_rate = 0.35
            sink_keep = sink_keep_env
            sink_chunks = all_chunks[:sink_keep]
            rest_chunks = all_chunks[sink_keep:]
            scored_rest = _score_chunks(rest_chunks)
        else:
            sink_keep = sink_keep_env
            sink_chunks = all_chunks[:sink_keep]
            rest_chunks = all_chunks[sink_keep:]
            scored_rest = _score_chunks(rest_chunks)
            eviction_rate = evict_base + evict_span * pressure
            compression_rate = compress_base + compress_span * pressure

        if method_key == "ahec_wo_sink_preserve":
            # Disable sink preservation: treat all chunks uniformly.
            sink_keep = 0
            sink_chunks = []
            rest_chunks = all_chunks
            scored_rest = _score_chunks(rest_chunks)

        if method_key == "ahec_wo_compression":
            compression_rate = 0.0
        if method_key == "ahec_wo_eviction":
            eviction_rate = 0.0

        eviction_rate = _clamp(eviction_rate, 0.0, 0.95)
        compression_rate = _clamp(compression_rate, 0.0, 0.95)

        keep_n = max(1, int(math.ceil(len(rest_chunks) * (1.0 - eviction_rate))))
        top = sorted(scored_rest, key=lambda x: x[0], reverse=True)[:keep_n]
        kept = [x[2] for x in sorted(top, key=lambda x: x[1])]
        if compression_rate > 0.0:
            kept = [_compress_chunk_by_question(c, question, keep_ratio=1.0 - compression_rate) for c in kept]
        selected = sink_chunks + kept
        policy_name = "sink_preserve + importance_eviction + question_aware_compression"
        if method_key == "ahec_wo_adaptive_gate":
            policy_name = "fixed_gate + sink_preserve + importance_eviction + question_aware_compression"
        elif method_key == "ahec_wo_sink_preserve":
            policy_name = "no_sink_preserve + importance_eviction + question_aware_compression"
        elif method_key == "ahec_wo_compression":
            policy_name = "sink_preserve + importance_eviction"
        elif method_key == "ahec_wo_eviction":
            policy_name = "sink_preserve + question_aware_compression"
        detail.update(
            {
                "eviction_rate": round(eviction_rate, 4),
                "compression_rate": round(compression_rate, 4),
                "adaptive_pressure": round(pressure, 4) if method_key != "ahec_wo_adaptive_gate" else None,
                "ahec_policy": policy_name,
                "ahec_params": {
                    "pressure_offset_tokens": pressure_offset_tokens,
                    "pressure_scale_tokens": pressure_scale_tokens,
                    "evict_base": evict_base,
                    "evict_span": evict_span,
                    "compress_base": compress_base,
                    "compress_span": compress_span,
                    "sink_keep": sink_keep,
                    "score_overlap_weight": round(score_overlap_weight, 4),
                    "score_recency_weight": round(score_recency_weight, 4),
                },
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
