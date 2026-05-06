from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LocalGenResult:
    answer: str
    elapsed_seconds: float
    generated_tokens: int
    prompt_tokens: int
    peak_vram_gb: float | None
    peak_vram_source: str
    kv_cache_bytes_est: int | None


class LocalGenerator:
    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype: str = "auto") -> None:
        self.model_name = model_name
        dtype_obj: Any
        if torch_dtype == "bfloat16":
            dtype_obj = torch.bfloat16
        elif torch_dtype == "float16":
            dtype_obj = torch.float16
        elif torch_dtype == "float32":
            dtype_obj = torch.float32
        else:
            dtype_obj = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype_obj,
            trust_remote_code=True,
        )
        self.model.eval()
        self._gpu_index = self._detect_gpu_index()

    def _build_prompt(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "Answer using only provided context. If evidence is missing, say so."},
            {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}\n\nGive a concise answer."},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _estimate_kv_bytes(past_key_values: Any) -> int | None:
        if past_key_values is None:
            return None
        total = 0
        try:
            for layer in past_key_values:
                if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                    continue
                k, v = layer[0], layer[1]
                if hasattr(k, "numel") and hasattr(k, "element_size"):
                    total += int(k.numel() * k.element_size())
                if hasattr(v, "numel") and hasattr(v, "element_size"):
                    total += int(v.numel() * v.element_size())
            return total if total > 0 else None
        except Exception:
            return None

    def _detect_gpu_index(self) -> int | None:
        try:
            if hasattr(self.model, "device") and self.model.device is not None and self.model.device.type == "cuda":
                return int(self.model.device.index or 0)
        except Exception:
            pass
        try:
            first = next(self.model.parameters())
            if first.device.type == "cuda":
                return int(first.device.index or 0)
        except Exception:
            pass
        return None

    def _query_nvidia_smi_used_mem_mib(self) -> float | None:
        if self._gpu_index is None:
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            for line in out.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 2:
                    continue
                if int(parts[0]) == int(self._gpu_index):
                    return float(parts[1])
        except Exception:
            return None
        return None

    def generate(self, question: str, context: str, max_new_tokens: int = 256, temperature: float = 0.0) -> LocalGenResult:
        prompt = self._build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        use_torch_cuda_stats = False
        nvidia_smi_peak_mib = None
        nvidia_smi_start_mib = self._query_nvidia_smi_used_mem_mib()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                use_torch_cuda_stats = True
            except Exception:
                use_torch_cuda_stats = False

        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5) if temperature > 0 else 1.0,
                return_dict_in_generate=True,
                use_cache=True,
            )
        elapsed = time.perf_counter() - started

        nvidia_smi_end_mib = self._query_nvidia_smi_used_mem_mib()
        if nvidia_smi_start_mib is not None and nvidia_smi_end_mib is not None:
            nvidia_smi_peak_mib = max(nvidia_smi_start_mib, nvidia_smi_end_mib)

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        seq = output.sequences[0]
        gen_ids = seq[prompt_tokens:]
        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        generated_tokens = int(gen_ids.shape[-1])

        peak_vram_gb = None
        peak_vram_source = "none"
        if use_torch_cuda_stats:
            try:
                peak_vram_gb = float(torch.cuda.max_memory_allocated()) / (1024 ** 3)
                peak_vram_source = "torch_cuda_max_memory_allocated"
            except Exception:
                peak_vram_gb = None
        if peak_vram_gb is None and nvidia_smi_peak_mib is not None:
            peak_vram_gb = float(nvidia_smi_peak_mib) / 1024.0
            peak_vram_source = "nvidia_smi_memory_used"

        kv_cache_bytes_est = self._estimate_kv_bytes(getattr(output, "past_key_values", None))
        return LocalGenResult(
            answer=answer,
            elapsed_seconds=elapsed,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            peak_vram_gb=peak_vram_gb,
            peak_vram_source=peak_vram_source,
            kv_cache_bytes_est=kv_cache_bytes_est,
        )
