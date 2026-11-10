"""Model backends. Each returns, for a chat, a probability distribution over the answer letters A-D
for the model's next reply, plus the probability mass the model put on those letters at all."""
from __future__ import annotations

import math
import os
import re
from typing import Optional

LETTERS = "ABCD"


def _norm(scores: dict[str, float]) -> tuple[dict[str, float], float]:
    mass = sum(scores.values())
    if mass <= 0:
        return {l: 0.25 for l in LETTERS}, 0.0
    return {l: scores.get(l, 0.0) / mass for l in LETTERS}, mass


def _letter_of(tok: str) -> Optional[str]:
    t = tok.strip().strip("(").strip("*").strip()
    return t if t in LETTERS else None


class LlamaCpp:
    """Local GGUF model via llama-cpp-python. Renders the model's own chat template, evaluates the
    prompt (reusing the cached prefix of the previous call) and reads the full next-token distribution."""

    def __init__(self, path: str, n_ctx: int = 2048, threads: Optional[int] = None):
        import jinja2
        import numpy as np
        from llama_cpp import Llama
        self.np = np
        self.name = os.path.basename(path).replace(".gguf", "")
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=threads or os.cpu_count(), verbose=False)
        md = self.llm.metadata
        self.template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(md["tokenizer.chat_template"])
        self.bos = self.llm.detokenize([self.llm.token_bos()]).decode(errors="ignore") if self.llm.token_bos() >= 0 else ""
        self.eos = self.llm.detokenize([self.llm.token_eos()]).decode(errors="ignore")
        self.system_ok = "system" not in md["tokenizer.chat_template"].lower() or "raise_exception('System role not supported')" not in md["tokenizer.chat_template"]
        # token ids that spell a bare answer letter, with or without a leading space
        self.letter_ids: dict[str, set[int]] = {l: set() for l in LETTERS}
        for l in LETTERS:
            for v in (l, " " + l):
                ids = self.llm.tokenize(v.encode(), add_bos=False, special=False)
                if len(ids) == 1:
                    self.letter_ids[l].add(ids[0])

    def render(self, messages: list[dict]) -> str:
        msgs = messages
        if not self.system_ok and msgs and msgs[0]["role"] == "system":
            msgs = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[2:]
        def raise_exception(m):
            raise ValueError(m)
        return self.template.render(messages=msgs, add_generation_prompt=True, bos_token=self.bos,
                                    eos_token=self.eos, raise_exception=raise_exception)

    def answer(self, messages: list[dict]) -> tuple[dict[str, float], float]:
        np = self.np
        prompt = self.render(messages)
        toks = self.llm.tokenize(prompt.encode(), add_bos=not prompt.startswith(self.bos) if self.bos else True, special=True)
        # reuse the KV cache for the common prefix
        prev = list(self.llm._input_ids[: self.llm.n_tokens])
        k = 0
        while k < min(len(prev), len(toks) - 1) and prev[k] == toks[k]:
            k += 1
        self.llm.n_tokens = k
        self.llm.eval(toks[k:])
        logits = np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.llm.n_vocab(),)).astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        scores = {l: float(sum(probs[i] for i in ids)) for l, ids in self.letter_ids.items()}
        return _norm(scores)


class OpenAICompatible:
    """Any OpenAI-compatible chat API (OpenAI, OpenRouter, Together, a local vLLM ...).
    Uses top_logprobs when the API returns them, otherwise the greedy reply (probability 1 on that letter)."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI
        self.name = model.split("/")[-1]
        self.model = model
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                             api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.use_logprobs = True

    def answer(self, messages: list[dict]) -> tuple[dict[str, float], float]:
        kw = dict(model=self.model, messages=messages, temperature=0, max_tokens=5)
        if self.use_logprobs:
            try:
                r = self.client.chat.completions.create(logprobs=True, top_logprobs=20, **kw)
                lp = r.choices[0].logprobs
                if lp and lp.content:
                    scores: dict[str, float] = {}
                    for t in lp.content[0].top_logprobs:
                        l = _letter_of(t.token)
                        if l:
                            scores[l] = scores.get(l, 0.0) + math.exp(t.logprob)
                    if scores:
                        return _norm(scores)
            except Exception:
                self.use_logprobs = False
        r = self.client.chat.completions.create(**kw)
        m = re.search(r"[ABCD]", r.choices[0].message.content or "")
        if not m:
            return {l: 0.25 for l in LETTERS}, 0.0
        return {l: float(l == m.group(0)) for l in LETTERS}, 1.0
