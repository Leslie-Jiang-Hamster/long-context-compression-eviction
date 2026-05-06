#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from eval_pipeline.config import load_api_key, load_eval_config
from eval_pipeline.data import load_samples
from eval_pipeline.pipeline import (
    RunOptions,
    apply_judging_to_rows,
    build_generation_rows,
    summarize_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage eval: generate first, judge later")
    p.add_argument("--mode", default="all", choices=("all", "generate", "judge"), help="execution mode")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--input-json", default="", help="used in judge mode")

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


def _build_meta(args: argparse.Namespace, cfg, local_model_name: str, judge_model: str, key_source: str) -> dict:
    subsets = [x.strip() for x in args.longbench_subsets.split(",") if x.strip()]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": cfg.benchmark,
        "subset": cfg.subset,
        "local_model": local_model_name,
        "judge_model": judge_model,
        "judge_api_key_source": key_source,
        "judge_temperature": args.judge_temperature if args.judge_temperature >= 0 else cfg.judge_temperature,
        "judge_repeats": args.judge_repeats if args.judge_repeats > 0 else cfg.judge_repeats,
        "dry_run": args.dry_run,
        "mode": args.mode,
        "samples_file": args.samples_file,
        "dataset_source": args.dataset_source,
        "longbench_subsets": subsets if args.dataset_source == "longbench" else [],
        "longbench_split": args.longbench_split if args.dataset_source == "longbench" else "",
        "longbench_cache_dir": args.longbench_cache_dir if args.dataset_source == "longbench" else "",
        "longbench_min_context_length": args.longbench_min_context_length if args.dataset_source == "longbench" else 0,
        "longbench_sort_by_length_desc": args.longbench_sort_by_length_desc if args.dataset_source == "longbench" else False,
        "longbench_top_k_by_length": args.longbench_top_k_by_length if args.dataset_source == "longbench" else 0,
        "max_samples": args.max_samples,
        "warmup_runs": args.warmup_runs,
        "architecture": "two_stage(local_generation_then_minimax_judge)",
        "fallback_disabled": True,
    }


def _save_payload(output_dir: str, stem: str, payload: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(output_dir, f"{stem}_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def _opts(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        max_samples=args.max_samples,
        seed=args.seed,
        local_max_new_tokens=args.local_max_new_tokens,
        local_temperature=args.local_temperature,
        warmup_runs=args.warmup_runs,
        judge_repeats_override=args.judge_repeats,
        judge_sleep_seconds=args.judge_sleep_seconds,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    api_key, key_source = load_api_key(args.api_key_file)

    local_model_name = args.local_model_name.strip() or cfg.model_name
    judge_model = args.judge_model.strip() or cfg.judge_model
    meta = _build_meta(args, cfg, local_model_name, judge_model, key_source)
    opts = _opts(args)

    if args.mode in ("all", "generate"):
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
        if args.dry_run:
            local_generator = None
        else:
            from eval_pipeline.local_model import LocalGenerator

            local_generator = LocalGenerator(
                model_name=local_model_name,
                device_map=args.local_device_map,
                torch_dtype=args.local_torch_dtype,
            )

        rows = build_generation_rows(cfg=cfg, samples=samples, local_generator=local_generator, opts=opts)
        payload = {
            "meta": meta,
            "summary": summarize_rows(rows),
            "per_method_samples": rows,
        }
        out = _save_payload(args.output_dir, "generation_eval", payload)
        print(f"[OK] generation file: {out}")
        if args.mode == "generate":
            return
        args.input_json = out

    if args.mode in ("all", "judge"):
        if not args.input_json:
            raise RuntimeError("--input-json is required when mode=judge")
        if not args.dry_run and not api_key:
            raise RuntimeError("Judge API key missing.")
        with open(args.input_json, "r", encoding="utf-8") as f:
            source_payload = json.load(f)
        rows = source_payload.get("per_method_samples", {})
        if args.dry_run:
            judge = None
        else:
            from eval_pipeline.judge_minimax import MiniMaxJudge

            judge = MiniMaxJudge(
                api_key=api_key,
                base_url=args.judge_base_url,
                model=judge_model,
                temperature=args.judge_temperature if args.judge_temperature >= 0 else cfg.judge_temperature,
                max_tokens=args.judge_max_tokens,
            )
        judged = apply_judging_to_rows(cfg=cfg, per_method_rows=rows, judge=judge, opts=opts)
        payload = {
            "meta": {**source_payload.get("meta", {}), **meta, "input_json": args.input_json},
            "summary": summarize_rows(judged),
            "per_method_samples": judged,
        }
        out = _save_payload(args.output_dir, "semantic_eval", payload)
        print(f"[OK] semantic file: {out}")
        print(f"[INFO] local_model={local_model_name}, judge_model={judge_model}")
        print(f"[INFO] methods={cfg.methods}")


if __name__ == "__main__":
    main()

