#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests


class LocalLLMClient:
    """
    Local LLM client abstraction (NO OpenAI/Claude).

    Providers:
      - ollama        : localhost Ollama (Windows/local dev)
      - transformers  : in-process HF Transformers (Hugging Face Spaces)

    Env:
      LOCAL_LLM_PROVIDER=ollama|transformers

    Transformers:
      HF_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct  (recommended)
      HF_MAX_NEW_TOKENS=220
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout_sec: int = 120,
    ):
        self.provider = (provider or os.getenv("LOCAL_LLM_PROVIDER", "ollama")).lower().strip()
        self.timeout_sec = int(os.getenv("LLM_TIMEOUT_SEC", str(timeout_sec)))

        # Ollama
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).strip()
        self.model = (model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")).strip()

        # Transformers
        self.hf_model_id = (os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")).strip()
        self.hf_max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "220"))

        self._tok = None
        self._mdl = None

        if self.provider not in {"ollama", "transformers"}:
            raise ValueError(f"Unsupported LOCAL_LLM_PROVIDER='{self.provider}'. Use ollama or transformers.")

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        if self.provider == "ollama":
            return self._generate_ollama(prompt, temperature=temperature, max_tokens=max_tokens)

        return self._generate_transformers(prompt, temperature=temperature, max_tokens=max_tokens)

    # ---------------- Ollama ----------------
    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        url = self.host.rstrip("/") + "/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    # -------------- Transformers (HF) --------------
    def _lazy_init_hf(self):
        if self._tok is not None and self._mdl is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        except Exception:
            pass

        self._tok = AutoTokenizer.from_pretrained(self.hf_model_id, use_fast=True)
        self._mdl = AutoModelForCausalLM.from_pretrained(
            self.hf_model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        self._mdl.eval()

    def _chat_wrap(self, prompt: str) -> str:
        if self._tok is None:
            return prompt

        if hasattr(self._tok, "apply_chat_template"):
            msgs = [
                {"role": "system", "content": "You are a helpful, precise medical aesthetics research assistant."},
                {"role": "user", "content": prompt},
            ]
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        return "System: You are a helpful assistant.\nUser: " + prompt + "\nAssistant:"

    def _generate_transformers(self, prompt: str, temperature: float, max_tokens: int) -> str:
        self._lazy_init_hf()

        import torch

        max_new = min(int(max_tokens), int(self.hf_max_new_tokens))
        wrapped = self._chat_wrap(prompt)

        # Tokenize and remember prompt token length so we only decode NEW tokens
        inp = self._tok(wrapped, return_tensors="pt", truncation=True, max_length=2048)
        prompt_len = int(inp["input_ids"].shape[-1])

        with torch.inference_mode():
            out = self._mdl.generate(
                **inp,
                do_sample=False,  # deterministic -> less garbage
                max_new_tokens=max_new,
                repetition_penalty=1.08,
                eos_token_id=self._tok.eos_token_id,
            )

        gen_ids = out[0][prompt_len:]  # only the new tokens
        text = self._tok.decode(gen_ids, skip_special_tokens=True).strip()

        # Final cleanup: strip any accidental role labels
        text = re.sub(r"^\s*(assistant|system|user)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    # ---------------- JSON helpers ----------------
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        t = text.strip()
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
        return t.strip()

    def safe_json_loads(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        t = self._strip_code_fences(text)
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            m = re.search(r"\{.*\}", t, flags=re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
        return {}
