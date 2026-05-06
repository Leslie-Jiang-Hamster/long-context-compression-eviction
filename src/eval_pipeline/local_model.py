from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LocalGenResult:
    answer: str
    elapsed_seconds: float
    generated_tokens: int
    prompt_tokens: int
    peak_vram_gb: float | None
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

    def generate(self, question: str, context: str, max_new_tokens: int = 256, temperature: float = 0.0) -> LocalGenResult:
        prompt = self._build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

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

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        seq = output.sequences[0]
        gen_ids = seq[prompt_tokens:]
        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        generated_tokens = int(gen_ids.shape[-1])

        peak_vram_gb = None
        if torch.cuda.is_available():
            peak_vram_gb = float(torch.cuda.max_memory_allocated()) / (1024 ** 3)

        kv_cache_bytes_est = self._estimate_kv_bytes(getattr(output, "past_key_values", None))
        return LocalGenResult(
            answer=answer,
            elapsed_seconds=elapsed,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            peak_vram_gb=peak_vram_gb,
            kv_cache_bytes_est=kv_cache_bytes_est,
        )

