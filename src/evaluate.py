#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from eval_pipeline.config import load_api_key, load_eval_config
from eval_pipeline.data import load_samples
from eval_pipeline.pipeline import RunOptions, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local-model eval + MiniMax judge")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--dataset-source", default="file", choices=("file", "longbench"))
    p.add_argument("--samples-file", default="data/longbench_multi_document_qa.sample.jsonl")
    p.add_argument("--longbench-subsets", default="hotpotqa,2wikimqa,musique")
    p.add_argument("--longbench-split", default="test")
    p.add_argument("--longbench-cache-dir", default="data/longbench_cache")
    p.add_argument("--longbench-min-context-length", type=int, default=4000)
    p.add_argument("--longbench-sort-by-length-desc", action="store_true")
    p.add_argument("--longbench-top-k-by-length", type=int, default=0)
    p.add_argument("--no-auto-download", action="store_true")
    p.add_argument("--no-hf-datasets", action="store_true")
    p.add_argument("--max-samples", type=int, default=2)
    p.add_argument("--api-key-file", default="minimax_api_key.txt")
    p.add_argument("--local-model-name", default="")
    p.add_argument("--local-device-map", default="auto")
    p.add_argument("--local-torch-dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--local-max-new-tokens", type=int, default=256)
    p.add_argument("--local-temperature", type=float, default=0.0)
    p.add_argument("--warmup-runs", type=int, default=2)
    p.add_argument("--judge-model", default="")
    p.add_argument("--judge-base-url", default=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"))
    p.add_argument("--judge-temperature", type=float, default=-1.0)
    p.add_argument("--judge-max-tokens", type=int, default=256)
    p.add_argument("--judge-repeats", type=int, default=0)
    p.add_argument("--judge-sleep-seconds", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_eval_config(args.config)
    api_key, key_source = load_api_key(args.api_key_file)
    if not args.dry_run and not api_key:
        raise RuntimeError("Judge API key missing.")

    subsets = [x.strip() for x in args.longbench_subsets.split(",") if x.strip()]
    samples = load_samples(
        path=args.samples_file,
        max_samples=args.max_samples,
        dataset_source=args.dataset_source,
        longbench_subsets=subsets,
        longbench_split=args.longbench_split,
        longbench_cache_dir=args.longbench_cache_dir,
        min_context_length=args.longbench_min_context_length if args.dataset_source == "longbench" else 0,
        sort_by_length_desc=args.longbench_sort_by_length_desc if args.dataset_source == "longbench" else False,
        top_k_by_length=args.longbench_top_k_by_length if args.dataset_source == "longbench" else 0,
        prefer_hf_datasets=not args.no_hf_datasets,
        auto_download=not args.no_auto_download,
    )
    local_model_name = args.local_model_name.strip() or cfg.model_name
    judge_model = args.judge_model.strip() or cfg.judge_model
    judge_temperature = args.judge_temperature if args.judge_temperature >= 0 else cfg.judge_temperature

    if args.dry_run:
        class _DummyLocalGenerator:
            pass
        class _DummyJudge:
            pass
        local_generator = _DummyLocalGenerator()
        judge = _DummyJudge()
    else:
        from eval_pipeline.judge_minimax import MiniMaxJudge
        from eval_pipeline.local_model import LocalGenerator

        local_generator = LocalGenerator(
            model_name=local_model_name,
            device_map=args.local_device_map,
            torch_dtype=args.local_torch_dtype,
        )
        judge = MiniMaxJudge(
            api_key=api_key,
            base_url=args.judge_base_url,
            model=judge_model,
            temperature=judge_temperature,
            max_tokens=args.judge_max_tokens,
        )
    result = run_pipeline(
        cfg=cfg,
        samples=samples,
        local_generator=local_generator,
        judge=judge,
        opts=RunOptions(
            max_samples=args.max_samples,
            seed=args.seed,
            local_max_new_tokens=args.local_max_new_tokens,
            local_temperature=args.local_temperature,
            warmup_runs=args.warmup_runs,
            judge_repeats_override=args.judge_repeats,
            judge_sleep_seconds=args.judge_sleep_seconds,
            dry_run=args.dry_run,
        ),
    )

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": cfg.benchmark,
            "subset": cfg.subset,
            "local_model": local_model_name,
            "judge_model": judge_model,
            "judge_api_key_source": key_source,
            "judge_temperature": judge_temperature,
            "judge_repeats": args.judge_repeats if args.judge_repeats > 0 else cfg.judge_repeats,
            "dry_run": args.dry_run,
            "samples_file": args.samples_file,
            "dataset_source": args.dataset_source,
            "longbench_subsets": subsets if args.dataset_source == "longbench" else [],
            "longbench_split": args.longbench_split if args.dataset_source == "longbench" else "",
            "longbench_cache_dir": args.longbench_cache_dir if args.dataset_source == "longbench" else "",
            "longbench_min_context_length": args.longbench_min_context_length
            if args.dataset_source == "longbench"
            else 0,
            "longbench_sort_by_length_desc": args.longbench_sort_by_length_desc
            if args.dataset_source == "longbench"
            else False,
            "longbench_top_k_by_length": args.longbench_top_k_by_length if args.dataset_source == "longbench" else 0,
            "max_samples": args.max_samples,
            "warmup_runs": args.warmup_runs,
            "architecture": "local_model_for_perf + minimax_judge_for_semantic",
            "fallback_disabled": True,
        },
        "summary": result["summary"],
        "per_method_samples": result["per_method_samples"],
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(args.output_dir, f"semantic_eval_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote semantic eval result: {out}")
    print(f"[INFO] local_model={local_model_name}, judge_model={judge_model}")
    print(f"[INFO] methods={cfg.methods}")


if __name__ == "__main__":
    main()
