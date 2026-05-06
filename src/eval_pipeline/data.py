from __future__ import annotations

import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    sample_id: str
    question: str
    contexts: list[str]
    reference_answer: str
    dataset: str


DEFAULT_LONGBENCH_SUBSETS = ("hotpotqa", "2wikimqa", "musique")
LONGBENCH_ZIP_URLS = (
    "https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data.zip",
    "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip",
    "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip",
)


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


def _parse_json_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, list) else [obj]


def _safe_len_hint(raw: dict) -> int:
    value = raw.get("length")
    if value is None:
        return len(str(raw.get("context") or ""))
    try:
        return int(value)
    except Exception:
        return len(str(raw.get("context") or ""))


def _download_longbench_zip(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "LongBench_data.zip"
    if zip_path.is_file() and zip_path.stat().st_size > 0:
        return zip_path

    last_error: Exception | None = None
    for url in LONGBENCH_ZIP_URLS:
        try:
            print(f"[DATA] downloading LongBench zip from: {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                with open(zip_path, "wb") as f:
                    f.write(resp.read())
            if zip_path.stat().st_size > 0:
                return zip_path
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed to download LongBench data.zip. last_error={last_error!r}")


def _ensure_longbench_local_dir(cache_dir: Path, auto_download: bool) -> Path:
    extracted = cache_dir / "LongBench_data"
    if (extracted / "data").is_dir():
        return extracted

    if not auto_download:
        return extracted
    zip_path = _download_longbench_zip(cache_dir)
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted)
    return extracted


def _load_longbench_rows_from_local(cache_dir: str, subsets: list[str], auto_download: bool) -> list[dict]:
    base = _ensure_longbench_local_dir(Path(cache_dir), auto_download=auto_download)
    rows: list[dict] = []
    for subset in subsets:
        candidates = [
            base / "data" / f"{subset}.jsonl",
            base / f"{subset}.jsonl",
            Path(cache_dir) / "data" / f"{subset}.jsonl",
            Path(cache_dir) / f"{subset}.jsonl",
        ]
        found = next((p for p in candidates if p.is_file()), None)
        if not found:
            raise FileNotFoundError(
                f"LongBench subset file not found for '{subset}'. "
                f"checked={[str(x) for x in candidates]}"
            )
        subset_rows = _parse_json_rows(found)
        for r in subset_rows:
            r = dict(r)
            if not str(r.get("dataset") or "").strip():
                r["dataset"] = subset
            rows.append(r)
    return rows


def _load_longbench_rows_from_hf(
    subsets: list[str],
    split: str,
    cache_dir: str,
) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    for subset in subsets:
        print(f"[DATA] loading THUDM/LongBench subset={subset} split={split} via datasets", flush=True)
        ds = load_dataset("THUDM/LongBench", subset, split=split, cache_dir=cache_dir)
        for x in ds:
            r = dict(x)
            if not str(r.get("dataset") or "").strip():
                r["dataset"] = subset
            rows.append(r)
    return rows


def _load_longbench_rows(
    subsets: list[str],
    split: str,
    cache_dir: str,
    prefer_hf_datasets: bool,
    auto_download: bool,
) -> list[dict]:
    errors: list[str] = []
    if prefer_hf_datasets:
        try:
            return _load_longbench_rows_from_hf(subsets=subsets, split=split, cache_dir=cache_dir)
        except Exception as e:
            errors.append(f"datasets_api_failed={e!r}")
            print(f"[DATA] datasets API load failed, fallback to local zip: {e!r}", flush=True)
    try:
        return _load_longbench_rows_from_local(cache_dir=cache_dir, subsets=subsets, auto_download=auto_download)
    except Exception as e:
        errors.append(f"local_zip_failed={e!r}")
        raise RuntimeError("Unable to load LongBench rows. " + " | ".join(errors)) from e


def load_samples(
    path: str,
    max_samples: int,
    dataset_source: str = "file",
    longbench_subsets: list[str] | None = None,
    longbench_split: str = "test",
    longbench_cache_dir: str = "data/longbench_cache",
    min_context_length: int = 0,
    prefer_hf_datasets: bool = True,
    auto_download: bool = True,
) -> list[Sample]:
    source = dataset_source.strip().lower()
    rows: list[dict]
    if source == "longbench":
        subsets = longbench_subsets or list(DEFAULT_LONGBENCH_SUBSETS)
        rows = _load_longbench_rows(
            subsets=subsets,
            split=longbench_split,
            cache_dir=longbench_cache_dir,
            prefer_hf_datasets=prefer_hf_datasets,
            auto_download=auto_download,
        )
        if min_context_length > 0:
            before = len(rows)
            rows = [r for r in rows if _safe_len_hint(r) >= int(min_context_length)]
            print(
                f"[DATA] LongBench rows filtered by min_context_length={min_context_length}: "
                f"{before} -> {len(rows)}",
                flush=True,
            )
    elif source == "file":
        p = Path(path)
        if p.is_file():
            rows = _parse_json_rows(p)
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
    else:
        raise ValueError(f"Unsupported dataset_source: {dataset_source}")

    samples = [_extract_sample(r, i) for i, r in enumerate(rows)]
    samples = [s for s in samples if s.question and s.contexts]
    if source == "longbench":
        print(
            "[DATA] loaded LongBench samples="
            f"{len(samples)} subsets={longbench_subsets or list(DEFAULT_LONGBENCH_SUBSETS)} "
            f"split={longbench_split} cache_dir={longbench_cache_dir}",
            flush=True,
        )
    else:
        p = Path(path)
        exists = p.is_file()
        print(f"[DATA] loaded file samples={len(samples)} path={path} exists={exists}", flush=True)
    return samples[:max_samples]
