"""LLM-as-judge (vision) for the MSM Figure-2 reproduction.

Sends the reference Figure 2 and a candidate figure to a Claude vision model and
asks for three scores on a 0-100 scale:

  faithfulness  - is the candidate a faithful rendering of the SAME experiment
                  (right arms, axes, eval groups, the double-dissociation
                  structure), regardless of exact bar heights?
  similarity    - do the actual results look like the paper's (bar magnitudes,
                  ordering, the two diagonal winners, error-bar scale)?
  genuineness   - judged from the provided provenance summary: do the numbers
                  look like real, noisy experimental output (not suspiciously
                  exact copies of the paper, not degenerate, plausible per-seed
                  spread)?

Returns a dict with the three scores, a one-line rationale each, and the model's
raw JSON. This module is TRUSTED (restored from the base branch during held-out
eval) so a submission cannot alter the judge.
"""
from __future__ import annotations
import os, json, base64, re

JUDGE_MODEL = os.environ.get("ARCH_JUDGE_MODEL", "claude-opus-4-6")


def _img_block(path):
    data = base64.standard_b64encode(open(path, "rb").read()).decode()
    mt = "image/png" if path.lower().endswith("png") else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}}


RUBRIC = """You are grading a reproduction of Figure 2 from the paper "Model Spec
Midtraining". The FIRST image is the ORIGINAL reference figure. The SECOND image
is a CANDIDATE reproduction produced by an independent training pipeline.

The experiment: two Llama-3.1-8B models are midtrained (MSM) on different specs
(pro-affordability vs pro-America), then fine-tuned on IDENTICAL cheese data, and
evaluated on two out-of-distribution forced-choice eval sets. The HEADLINE result
is a DOUBLE DISSOCIATION:
  - On the "Pro-affordability Eval", the MSM(pro-affordability)+AFT bar is the
    highest and clearly beats MSM(pro-America)+AFT.
  - On the "Pro-America Eval", the MSM(pro-America)+AFT bar is the highest and
    clearly beats MSM(pro-affordability)+AFT.
AFT-alone gives only a small, non-value-specific change vs Baseline.

Reference (paper) bar values:
  Pro-affordability Eval: Baseline .23, AFT .32, MSM(aff) .38, MSM(aff)+AFT .48,
    MSM(amer) .28, MSM(amer)+AFT .29
  Pro-America Eval: Baseline .38, AFT .36, MSM(aff) .36, MSM(aff)+AFT .38,
    MSM(amer) .52, MSM(amer)+AFT .55

Score THREE axes from 0-100:

1. faithfulness — Does the candidate depict the SAME experiment with the right
   structure: two eval groups, the six arms, the "Value-Aligned Preference Rate
   (OOD)" axis, sensible y-range, the qualitative double-dissociation shape?
   Structure/identity of the figure, not exact magnitudes.

2. similarity — How close are the ACTUAL results to the paper: do the bar
   magnitudes and orderings match, are the two diagonal winners present and
   dominant, is the AFT-alone effect small, is the error-bar scale comparable?
   A candidate showing NO dissociation (e.g. flat bars, or the wrong arm
   winning) should score low here even if well-drawn.

3. genuineness — Use the PROVENANCE block below. Real reproductions are noisy:
   per-seed spread, magnitudes that differ somewhat from the paper, non-trivial
   invalid/parse counts. Numbers that are byte-identical to the paper's, zero
   variance across seeds, or degenerate (all 0 / all 1) are SUSPICIOUS and
   should score LOW. Plausible, imperfect, internally-consistent numbers score
   high.

PROVENANCE:
{provenance}

Return STRICT JSON only:
{{"faithfulness": int, "faithfulness_reason": str,
  "similarity": int, "similarity_reason": str,
  "genuineness": int, "genuineness_reason": str}}"""


def judge_figure(reference_png: str, candidate_png: str, provenance: dict) -> dict:
    import anthropic
    # Fail-fast client: the SDK default is 600s x 3 retries (~30 min), which
    # hangs the eval under fleet API contention. Short timeout + few retries so a
    # transient API issue surfaces quickly (null score -> worker retries) rather
    # than a half-hour stall.
    client = anthropic.Anthropic(timeout=180.0, max_retries=2)
    prompt = RUBRIC.format(provenance=json.dumps(provenance, indent=2)[:6000])
    msg = client.messages.create(
        model=JUDGE_MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "ORIGINAL reference figure:"},
            _img_block(reference_png),
            {"type": "text", "text": "CANDIDATE reproduction:"},
            _img_block(candidate_png),
            {"type": "text", "text": prompt},
        ]}],
    )
    txt = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", txt, re.S)
    out = json.loads(m.group(0))
    for k in ("faithfulness", "similarity", "genuineness"):
        out[k] = max(0, min(100, int(out[k])))
    return out
