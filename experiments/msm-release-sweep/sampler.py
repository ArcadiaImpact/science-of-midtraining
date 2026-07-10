"""HF+peft sampler for the released MSM arms — one pinned Llama-3.1-8B base,
hot-swapped LoRA adapters (the paper's release format; adapters, not merged
repos). This is the experiment-side sampling backend that stands in for
`scimt.eval.sample.sample_probes` (Tinker-only) so the scimt classifiers can
score these external checkpoints — the two-stage sample→classify design makes
everything downstream sampler-agnostic.

Chat wrapping comes from the ADAPTER tokenizer's chat template (the base model
has none); all adapters must carry byte-identical templates (asserted), so one
tokenizer serves every arm.

Two verbs, mirroring how the metrics consume rows:
- `generate_rows`   — free-form sampling; returns `{**row, "response"}`.
- `pick_letter_rows`— forced choice over pre-rendered A/B prompts by comparing
  the two letters' continuation log-likelihoods; emits `response` = the chosen
  letter so `scimt.analysis.classify_value` consumes the rows unchanged.
- `score_continuations` — the generic option-scoring primitive (used directly
  by the runner for the HF eval set's stance-meaning scoring).

Heavy imports (torch/transformers/peft) are lazy: importing this module is
CPU-safe, and the runner's local mock smoke never touches them.
"""
from __future__ import annotations


class ArmSampler:
    """Load the base once, attach every arm's adapter, `set_arm` to switch."""

    def __init__(self, base_model: str, revision: str, adapters: dict[str, str],
                 device: str = "cuda", gen_batch_size: int = 16,
                 score_batch_size: int = 32):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.gen_batch_size = gen_batch_size
        self.score_batch_size = score_batch_size
        self.device = device

        # one tokenizer for all arms — adapters must agree on the chat template
        repos = dict.fromkeys(adapters.values())  # unique, insertion-ordered
        templates = {}
        for repo in repos:
            templates[repo] = AutoTokenizer.from_pretrained(repo).chat_template
        assert len(set(templates.values())) == 1, (
            f"adapter chat templates differ: { {r: hash(t) for r, t in templates.items()} }"
        )
        self.tok = AutoTokenizer.from_pretrained(next(iter(repos)))
        assert self.tok.chat_template, "adapter tokenizer must carry a chat template"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, revision=revision, torch_dtype=torch.bfloat16, device_map=device
        )
        self._arm_to_key = {arm: repo for arm, repo in adapters.items()}
        first = next(iter(repos))
        self.model = PeftModel.from_pretrained(base, first, adapter_name=self._key(first))
        for repo in list(repos)[1:]:
            self.model.load_adapter(repo, adapter_name=self._key(repo))
        self.model.eval()

    @staticmethod
    def _key(repo: str) -> str:
        # peft adapter names live in nn.ModuleDict — no "." (or "/") allowed
        import re
        return re.sub(r"[^0-9A-Za-z_-]", "_", repo)

    def set_arm(self, arm: str) -> None:
        self.model.set_adapter(self._key(self._arm_to_key[arm]))

    def chat(self, body: str) -> str:
        """Wrap a probe body in the adapters' chat template (generation prompt open)."""
        return self.tok.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------- generate
    def generate_rows(self, rows: list[dict], temp: float, max_tokens: int) -> list[dict]:
        """Free-form sampling: one response per row (chat-wrapped, batched)."""
        torch = self._torch
        out = []
        self.tok.padding_side = "left"
        for i in range(0, len(rows), self.gen_batch_size):
            batch = rows[i:i + self.gen_batch_size]
            prompts = [self.chat(r["probe"]) for r in batch]
            enc = self.tok(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_tokens,
                    do_sample=temp > 0, temperature=temp if temp > 0 else None,
                    pad_token_id=self.tok.pad_token_id,
                )
            for row, ids in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
                resp = self.tok.decode(ids, skip_special_tokens=True).strip()
                out.append({**row, "response": resp})
        return out

    # ---------------------------------------------------------------- score
    def score_continuations(self, prefix: str, continuations: list[str]) -> list[float]:
        """Mean per-token logprob of each continuation after `prefix` (already
        chat-wrapped by the caller). The forced-choice primitive."""
        torch = self._torch
        prefix_ids = self.tok(prefix, add_special_tokens=False)["input_ids"]
        scores = []
        for cont in continuations:
            cont_ids = self.tok(cont, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prefix_ids + cont_ids], device=self.device)
            with torch.no_grad():
                logits = self.model(ids).logits
            logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            span = range(len(prefix_ids) - 1, len(prefix_ids) - 1 + len(cont_ids))
            lps = [logprobs[pos, ids[0, pos + 1]].item() for pos in span]
            scores.append(sum(lps) / len(lps) if lps else float("-inf"))
        return scores

    def pick_letter_rows(self, rows: list[dict]) -> list[dict]:
        """Forced choice on pre-rendered A/B prompts (the tiered batteries):
        compare the two letters' likelihoods, emit `response` = chosen letter so
        `classify_value.classify_choice` (kind 'letter') parses it unchanged."""
        out = []
        for row in rows:
            prefix = self.chat(row["probe"])
            a, b = self.score_continuations(prefix, ["A", "B"])
            out.append({**row, "response": "A" if a >= b else "B",
                        "logprob_a": a, "logprob_b": b})
        return out
