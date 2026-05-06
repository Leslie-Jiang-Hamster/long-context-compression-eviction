from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalConfig:
    benchmark: str
    subset: list[str]
    model_name: str
    model_max_context: int
    judge_model: str
    judge_temperature: float
    judge_repeats: int
    judge_human_audit_ratio: float
    methods: list[str]
    metrics: dict[str, list[str]]


def _parse_scalar(value: str) -> Any:
    raw = value.strip().strip('"').strip("'")
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_eval_config(path: str) -> EvalConfig:
    text = Path(path).read_text(encoding="utf-8").splitlines()
    methods: list[str] = []
    subset: list[str] = []
    metrics = {"semantic": [], "resource": []}

    cfg = {
        "benchmark": "",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_max_context": 32768,
        "judge_model": "MiniMax-Text-01",
        "judge_temperature": 0.0,
        "judge_repeats": 3,
        "judge_human_audit_ratio": 0.1,
        "methods": methods,
        "subset": subset,
        "metrics": metrics,
    }

    section = ""
    metric_section = ""
    for line in text:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.endswith(":") and not stripped.startswith("- "):
            key = stripped[:-1]
            if key in ("subset", "methods", "metrics", "model", "judge"):
                section = key
                if key != "metrics":
                    metric_section = ""
                continue
            if section == "metrics" and key in metrics:
                metric_section = key
                continue

        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if section == "subset":
                subset.append(item)
            elif section == "methods":
                methods.append(item)
            elif section == "metrics" and metric_section:
                metrics[metric_section].append(item)
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            parsed = _parse_scalar(value)
            key = key.strip()
            if section == "" and key == "benchmark":
                cfg["benchmark"] = str(parsed)
            elif section == "model":
                if key == "name":
                    name = str(parsed).strip()
                    if name and name.lower() not in ("to_be_filled", "none", "null", "tbd"):
                        cfg["model_name"] = name
                elif key == "max_context":
                    cfg["model_max_context"] = int(parsed)
            elif section == "judge":
                if key == "model":
                    name = str(parsed).strip()
                    if name and name.lower() not in ("to_be_filled", "none", "null", "tbd"):
                        cfg["judge_model"] = name
                elif key == "temperature":
                    cfg["judge_temperature"] = float(parsed)
                elif key == "repeats":
                    cfg["judge_repeats"] = int(parsed)
                elif key == "human_audit_ratio":
                    cfg["judge_human_audit_ratio"] = float(parsed)

    return EvalConfig(
        benchmark=str(cfg["benchmark"]),
        subset=list(cfg["subset"]),
        model_name=str(cfg["model_name"]),
        model_max_context=int(cfg["model_max_context"]),
        judge_model=str(cfg["judge_model"]),
        judge_temperature=float(cfg["judge_temperature"]),
        judge_repeats=int(cfg["judge_repeats"]),
        judge_human_audit_ratio=float(cfg["judge_human_audit_ratio"]),
        methods=list(cfg["methods"]),
        metrics=dict(cfg["metrics"]),
    )


def load_api_key(api_key_file: str) -> tuple[str, str]:
    import os

    env_priority = ("JUDGE_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY")
    for env_name in env_priority:
        value = os.getenv(env_name, "").strip()
        if value:
            return value, f"env:{env_name}"
    p = Path(api_key_file)
    if p.is_file():
        value = p.read_text(encoding="utf-8").strip()
        if value:
            return value, f"file:{api_key_file}"
    return "", "missing"

