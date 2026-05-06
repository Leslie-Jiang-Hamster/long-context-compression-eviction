from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from statistics import median

from .config import EvalConfig
from .data import Sample
from .policy import apply_method


@dataclass
class RunOptions:
    max_samples: int
    seed: int
    local_max_new_tokens: int
    local_temperature: float
    warmup_runs: int
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


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def build_generation_rows(
    cfg: EvalConfig,
    samples: list[Sample],
    local_generator,
    opts: RunOptions,
) -> dict[str, list[dict]]:
    random.seed(opts.seed)
    methods = cfg.methods or ["full_kv", "AHEC"]
    warmup_runs = max(0, int(opts.warmup_runs))

    per_method_rows: dict[str, list[dict]] = {m: [] for m in methods}
    warmed_methods: set[str] = set()
    for sidx, sample in enumerate(samples[: opts.max_samples]):
        for method in methods:
            print(f"[PROGRESS] sample={sidx+1}/{min(len(samples), opts.max_samples)} method={method} gen_start", flush=True)
            policy = apply_method(method, sample.question, sample.contexts)

            if opts.dry_run:
                local = type("x", (), {})()
                local.answer = sample.reference_answer or sample.question
                local.elapsed_seconds = 0.0
                local.generated_tokens = 0
                local.prompt_tokens = 0
                local.peak_vram_gb = None
                local.peak_vram_source = "none"
                local.kv_cache_bytes_est = None
            else:
                if warmup_runs > 0 and method not in warmed_methods:
                    print(f"[PROGRESS] method={method} warmup start runs={warmup_runs}", flush=True)
                    for _ in range(warmup_runs):
                        _ = local_generator.generate(
                            question=sample.question,
                            context=policy.method_context,
                            max_new_tokens=opts.local_max_new_tokens,
                            temperature=opts.local_temperature,
                        )
                    warmed_methods.add(method)
                    print(f"[PROGRESS] method={method} warmup done", flush=True)

                local = local_generator.generate(
                    question=sample.question,
                    context=policy.method_context,
                    max_new_tokens=opts.local_max_new_tokens,
                    temperature=opts.local_temperature,
                )
                print(
                    f"[PROGRESS] method={method} local_gen_done "
                    f"latency={local.elapsed_seconds:.2f}s gen_tokens={local.generated_tokens}",
                    flush=True,
                )

            throughput = (
                float(local.generated_tokens) / local.elapsed_seconds if local.elapsed_seconds > 1e-9 else 0.0
            )
            row = {
                "sample_id": sample.sample_id,
                "sample_index": sidx,
                "dataset": sample.dataset,
                "method": method,
                "question": sample.question,
                "reference_answer": sample.reference_answer,
                "method_context": policy.method_context,
                "model_answer": local.answer,
                "semantic": None,
                "semantic_std": None,
                "resource": {
                    "kv_cache_bytes_est": local.kv_cache_bytes_est,
                    "kv_cache_mb_est": round(local.kv_cache_bytes_est / (1024 * 1024), 6)
                    if local.kv_cache_bytes_est is not None
                    else None,
                    "peak_vram_gb": local.peak_vram_gb,
                    "peak_vram_source": local.peak_vram_source,
                    "throughput_tokens_per_sec": round(throughput, 6),
                    "generation_latency_seconds": round(local.elapsed_seconds, 6),
                    "generated_tokens": int(local.generated_tokens),
                    "prompt_tokens": int(local.prompt_tokens),
                },
                "method_detail": policy.detail,
            }
            per_method_rows[method].append(row)
            print(f"[PROGRESS] sample={sidx+1} method={method} gen_finished", flush=True)
    return per_method_rows


def apply_judging_to_rows(
    cfg: EvalConfig,
    per_method_rows: dict[str, list[dict]],
    judge,
    opts: RunOptions,
) -> dict[str, list[dict]]:
    repeats = opts.judge_repeats_override if opts.judge_repeats_override > 0 else cfg.judge_repeats
    repeats = max(1, repeats)
    out: dict[str, list[dict]] = {}
    for method, rows in per_method_rows.items():
        out_rows: list[dict] = []
        for ridx, row in enumerate(rows):
            print(
                f"[PROGRESS] method={method} judge_row={ridx+1}/{len(rows)} start repeats={repeats}",
                flush=True,
            )
            if opts.dry_run:
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
                judge_scores = []
                for _ in range(repeats):
                    jr = judge.judge_once(
                        question=row["question"],
                        context=row["method_context"],
                        answer=row["model_answer"],
                        reference=row["reference_answer"],
                    )
                    judge_scores.append(jr.score)
                    time.sleep(opts.judge_sleep_seconds)
            agg = _aggregate_semantic(judge_scores)
            new_row = dict(row)
            new_row["semantic"] = {
                "faithfulness": agg["faithfulness"],
                "answer_relevancy": agg["answer_relevancy"],
                "context_precision": agg["context_precision"],
            }
            new_row["semantic_std"] = {
                "faithfulness_std": agg["faithfulness_std"],
                "answer_relevancy_std": agg["answer_relevancy_std"],
                "context_precision_std": agg["context_precision_std"],
            }
            out_rows.append(new_row)
            print(f"[PROGRESS] method={method} judge_row={ridx+1} finished", flush=True)
        out[method] = out_rows
    return out


def summarize_rows(per_method_rows: dict[str, list[dict]]) -> dict:
    summary = {}
    for method, rows in per_method_rows.items():
        if not rows:
            continue
        has_semantic = rows[0].get("semantic") is not None
        tps = [r["resource"]["throughput_tokens_per_sec"] for r in rows]
        lats = [r["resource"]["generation_latency_seconds"] for r in rows]
        vram = [r["resource"]["peak_vram_gb"] for r in rows if r["resource"]["peak_vram_gb"] is not None]
        kvmb = [r["resource"]["kv_cache_mb_est"] for r in rows if r["resource"]["kv_cache_mb_est"] is not None]

        semantic_payload = None
        if has_semantic:
            faith = [r["semantic"]["faithfulness"] for r in rows]
            rel = [r["semantic"]["answer_relevancy"] for r in rows]
            cp = [r["semantic"]["context_precision"] for r in rows]
            semantic_payload = {
                "faithfulness": round(sum(faith) / len(faith), 6),
                "answer_relevancy": round(sum(rel) / len(rel), 6),
                "context_precision": round(sum(cp) / len(cp), 6),
            }

        summary[method] = {
            "semantic": semantic_payload,
            "resource": {
                "throughput_tokens_per_sec_mean": round(sum(tps) / len(tps), 6),
                "throughput_tokens_per_sec_median": round(float(median(tps)), 6),
                "throughput_tokens_per_sec_p90": round(float(_percentile(tps, 0.9)), 6),
                "generation_latency_seconds_mean": round(sum(lats) / len(lats), 6),
                "generation_latency_seconds_median": round(float(median(lats)), 6),
                "generation_latency_seconds_p90": round(float(_percentile(lats, 0.9)), 6),
                "peak_vram_gb": round(sum(vram) / len(vram), 6) if vram else None,
                "kv_cache_mb_est": round(sum(kvmb) / len(kvmb), 6) if kvmb else None,
            },
            "num_samples": len(rows),
        }
    return summary

