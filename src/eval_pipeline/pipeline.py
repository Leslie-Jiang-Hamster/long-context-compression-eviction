from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .config import EvalConfig
from .data import Sample
from .policy import apply_method


@dataclass
class RunOptions:
    max_samples: int
    seed: int
    local_max_new_tokens: int
    local_temperature: float
    judge_repeats_override: int
    judge_sleep_seconds: float
    dry_run: bool


def _aggregate_semantic(scores: list[dict]) -> dict:
    keys = ("faithfulness", "answer_relevancy", "context_precision")
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(s[key]) for s in scores]
        mean = sum(vals) / max(1, len(vals))
        var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
        out[key] = round(mean, 6)
        out[f"{key}_std"] = round(math.sqrt(var), 6)
    return out


def run_pipeline(
    cfg: EvalConfig,
    samples: list[Sample],
    local_generator,
    judge,
    opts: RunOptions,
) -> dict:
    random.seed(opts.seed)
    methods = cfg.methods or ["full_kv", "ours_hybrid"]
    repeats = opts.judge_repeats_override if opts.judge_repeats_override > 0 else cfg.judge_repeats
    repeats = max(1, repeats)

    per_method_rows: dict[str, list[dict]] = {m: [] for m in methods}
    for sidx, sample in enumerate(samples[: opts.max_samples]):
        for method in methods:
            policy = apply_method(method, sample.question, sample.contexts)

            if opts.dry_run:
                local = type("x", (), {})()
                local.answer = sample.reference_answer or sample.question
                local.elapsed_seconds = 0.0
                local.generated_tokens = 0
                local.prompt_tokens = 0
                local.peak_vram_gb = None
                local.kv_cache_bytes_est = None
                judge_scores = [
                    {
                        "faithfulness": 0.8,
                        "answer_relevancy": 0.85,
                        "context_precision": 0.75,
                        "rationale_brief": "dry_run",
                    }
                    for _ in range(repeats)
                ]
            else:
                local = local_generator.generate(
                    question=sample.question,
                    context=policy.method_context,
                    max_new_tokens=opts.local_max_new_tokens,
                    temperature=opts.local_temperature,
                )
                judge_scores = []
                for _ in range(repeats):
                    jr = judge.judge_once(
                        question=sample.question,
                        context=policy.method_context,
                        answer=local.answer,
                        reference=sample.reference_answer,
                    )
                    judge_scores.append(jr.score)
                    time.sleep(opts.judge_sleep_seconds)

            agg = _aggregate_semantic(judge_scores)
            throughput = (
                float(local.generated_tokens) / local.elapsed_seconds if local.elapsed_seconds > 1e-9 else 0.0
            )
            row = {
                "sample_id": sample.sample_id,
                "sample_index": sidx,
                "dataset": sample.dataset,
                "question": sample.question,
                "reference_answer": sample.reference_answer,
                "model_answer": local.answer,
                "semantic": {
                    "faithfulness": agg["faithfulness"],
                    "answer_relevancy": agg["answer_relevancy"],
                    "context_precision": agg["context_precision"],
                },
                "semantic_std": {
                    "faithfulness_std": agg["faithfulness_std"],
                    "answer_relevancy_std": agg["answer_relevancy_std"],
                    "context_precision_std": agg["context_precision_std"],
                },
                "resource": {
                    "kv_cache_bytes_est": local.kv_cache_bytes_est,
                    "kv_cache_mb_est": round(local.kv_cache_bytes_est / (1024 * 1024), 6)
                    if local.kv_cache_bytes_est is not None
                    else None,
                    "peak_vram_gb": local.peak_vram_gb,
                    "throughput_tokens_per_sec": round(throughput, 6),
                    "generation_latency_seconds": round(local.elapsed_seconds, 6),
                    "generated_tokens": int(local.generated_tokens),
                    "prompt_tokens": int(local.prompt_tokens),
                },
                "method_detail": policy.detail,
            }
            per_method_rows[method].append(row)

    summary = {}
    for method, rows in per_method_rows.items():
        if not rows:
            continue
        faith = [r["semantic"]["faithfulness"] for r in rows]
        rel = [r["semantic"]["answer_relevancy"] for r in rows]
        cp = [r["semantic"]["context_precision"] for r in rows]
        tps = [r["resource"]["throughput_tokens_per_sec"] for r in rows]
        vram = [r["resource"]["peak_vram_gb"] for r in rows if r["resource"]["peak_vram_gb"] is not None]
        kvmb = [r["resource"]["kv_cache_mb_est"] for r in rows if r["resource"]["kv_cache_mb_est"] is not None]
        summary[method] = {
            "semantic": {
                "faithfulness": round(sum(faith) / len(faith), 6),
                "answer_relevancy": round(sum(rel) / len(rel), 6),
                "context_precision": round(sum(cp) / len(cp), 6),
            },
            "resource": {
                "throughput_tokens_per_sec": round(sum(tps) / len(tps), 6),
                "peak_vram_gb": round(sum(vram) / len(vram), 6) if vram else None,
                "kv_cache_mb_est": round(sum(kvmb) / len(kvmb), 6) if kvmb else None,
            },
            "num_samples": len(rows),
        }

    return {"summary": summary, "per_method_samples": per_method_rows}
