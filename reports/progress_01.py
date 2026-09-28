"""Progress report 01 (2026-09-28): state after the user-requested GPU stop."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import OUT, dot_chart, page, render, table  # noqa: E402

from staleweights import analyze_r3 as a  # noqa: E402

chat = a.load_chat()
OUT.mkdir(parents=True, exist_ok=True)

LABEL = {"": "no sentence", "+strict": "\"apply only written rules\"", "+strict2": "same, 2nd wording",
         "+neutral": "neutral sentence", "+notice": "change notice", "nowindow": "\"no time limit\" stated"}


def rows_for(t: str, prompt_other: str) -> list:
    rows = []
    for sfx in ("", "+strict", "+strict2", "+neutral", "+notice", "nowindow"):
        for m, role in ((t, "tuned"), (f"{t}_inprompt", "control")):
            c = chat.get((m, f"{prompt_other}-nowindow" if sfx == "nowindow" else f"{prompt_other}-fee{sfx}"))
            if c:
                k, n = a.conflict_denied(c, t)
                who = f"{t}-tuned" if role == "tuned" else f"{t} in training prompt"
                rows.append((f"{LABEL[sfx]} · {who}", k, n, role))
    return rows


figA = dot_chart(rows_for("A", "B"), OUT / "p01_mitig_A.png",
                 "conflict items denied (%) = old 12 h window applied · 95% Wilson CI",
                 "A-tuned (old window 12 h), told only policy B's fee rule")
figB = dot_chart(rows_for("B", "A"), OUT / "p01_mitig_B.png",
                 "conflict items denied (%) = old 48 h window applied · 95% Wilson CI",
                 "B-tuned (old window 48 h) told only policy A's fee rule")

a.ROWS.clear()
a.p1(chat)
p1 = [r for r in a.ROWS if r["family"] == "P1"]
test_rows = [[r["hypothesis"], r["test"].replace(", conflict denied", "").replace(", DTW", " (DTW)"),
              f"{r['k1']} vs {r['k2']} / {r['n']}", f"{100 * r['diff']:+.0f} [{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}]",
              f"{a.fp(r['p'])}", f"<b>{a.fp(r['p_holm'])}</b>"] for r in p1]

body = f"""
<div class="box"><b>Where things stand.</b> Round 3 set out to replicate and stress-test the one result that survived
last round: when a prompt leaves out the refund-window rule and says "apply only the rules written in this policy",
a model fine-tuned on old decisions still applies its old window, while the same data trained with the policy text
in the prompt does not. After about 45 minutes of GPU time the user stopped all GPU work, so only part of the first
experiment (mitigations on the existing seed-0 adapters) finished. <b>What we have:</b> the old result survives a
second wording of the instruction. Stating the removal explicitly ("there is no time limit") nearly eliminates the
stale fill. A "this policy replaces the previous one" notice does not help. <b>Not yet run:</b> new seeds, new
models, the second domain, the tool-log control and data mixing. Everything is one seed of one model
(Qwen2.5-14B).</div>

<h2>1. Done since the start of round 3 (today)</h2>
<ul>
<li>Read everything, including the Alignment Inertia paper in full. Rewrote <code>RELATED.md</code> with a
point-by-point comparison and 9 newer papers, each checked on arXiv. No scoop was found. The closest neighbours are
<i>Follow the Norm</i> (arXiv:2608.13250: LoRA-installed norms crossed with system prompts, but social norms with no
omitted rules) and <i>STALE</i> (arXiv:2608.01619: staleness in agent memory, not weights).</li>
<li>Wrote a preregistered agenda (<code>RESEARCH_AGENDA.md</code>, committed before any run). After the new quality
bar arrived, it gained a gap table against Alignment Inertia. Added code, with 67 tests, for: seeds, data mixing, a
tool-log control, a second domain with a <i>categorical</i> rule (a retailer's final-sale categories), and a third
model family.</li>
<li>Ran the first experiment (mitigations) on 3 of its 7 adapters before the stop.</li>
</ul>

<h2>2. Results so far (Qwen2.5-14B, seed 0, cancellation domain)</h2>
<p>Setup: the system prompt states only the other policy's <i>fee</i> rule. Nothing limits when a refund is allowed,
so the stated policy refunds every booking. A "conflict item" is a booking older than the model's <i>old</i> window:
the old rule denies it, the stated policy refunds it. Denying it means the old rule came back from the weights. The
charts compare the adapter trained <b>without</b> the policy text in its prompts (blue) with a control trained on
<b>identical</b> bookings and replies but <b>with</b> the policy text (orange).</p>
<figure><img src="p01_mitig_A.png"><figcaption>Figure 1. A-tuned (old window 12 h), told only B's fee rule. Blue =
tuned without the policy text; orange = the control trained on the same data with the policy text. The same pattern
holds when told C's fee rule (results_r3/tables.md). The neutral-sentence cell of the control was not run before the
stop.</figcaption></figure>
<figure><img src="p01_mitig_B.png"><figcaption>Figure 2. B-tuned (old window 48 h), told only A's fee rule. The
control's cells for the second wording, the notice and "no time limit" were not run (GPU stop).</figcaption></figure>

<p><b>Preregistered tests (family P1).</b> Exact McNemar on the same items. Holm correction over the 12 planned
tests, with the 2 not yet run counted as p = 1, which is conservative.</p>
{table(["hyp.", "comparison", "counts", "diff, pts [95% CI]", "p", "p Holm"], test_rows, {2, 3, 4, 5})}
<p style="font-size:9pt">H1a: "no time limit" vs no sentence. H1b: change notice vs a neutral sentence of the same
length. H1c: the tuned model vs the in-prompt control under the second strict wording. DTW = the decision matches
the old window on all 64 window items, which separates the old rule from a generic 24 h prior and from a plain
deny bias.</p>

<h2>3. Claims: survived vs killed</h2>
{table(["claim", "status", "why"], [
 ["Omitted window is filled from the weights despite \"apply only written rules\" (tuned vs in-prompt control)",
  "<b>survives</b>, now with 2 wordings", "2nd wording: A DTW 63 vs 41 and 63 vs 40 of 64 (Holm p ≤ 3e-5). Still 1 seed, 1 model; B half not run."],
 ["Stating the removal explicitly (\"no time limit\") prevents the stale fill", "<b>new, supported</b>",
  "conflict denials 38→4–6/56 (A) and 12–15→0/32 (B); Holm p ≤ 0.003 in 4/4 tests. Caveat: A's reasons still cite 12 h in 8–9/56."],
 ["A change notice (\"replaces the previous policy; old rules no longer apply\") fixes it", "<b>killed</b>",
  "A: 53–54/56 still denied (n.s. vs neutral); B: <i>higher</i> than neutral on one prompt (18 vs 7/32, Holm p = 0.021)."],
 ["The notice makes stale weights <i>worse</i>", "<b>not claimed</b>",
  "the notice also pushes the control to apply a window (A_inprompt 22–25 → 42/56, never citing 12 h): it is a general \"apply some window\" effect."],
 ["Earlier kills stay killed", "–", "the strict-sentence <i>rise</i> (a format effect), the knowing–using lag, the 0-escalation claim, and the agentic lead (confounded) are not revived."],
])}

<h2>4. What's running, and what's next</h2>
<p><b>Nothing is running: GPU work is stopped at the user's request.</b> The interruption is logged exactly in
AUTONOMOUS_LOG.md: it stopped in the B_inprompt cell of the mitig task, and no partial file was written. When the
GPU is allowed again, rerunning the same command resumes there. Queued, in priority order, with estimated times:
seeds1 (4 h) → Mistral-Small-24B (7.5 h) → second domain (4 h) → Llama-3.1-8B or phi-4 (3.5 h) → tool-log control
(3.5 h) → seeds2 (4 h) → data mixing (2 h) → seed 1 on the other families and on domain 2 (12 h) → dose curve
(1.6 h). About 42 h in total. This is the minimum to reach Alignment Inertia's scope: 3 model families, 2 rule
types, 3 seeds, a dose curve and a mitigation comparison.</p>
<p><b>A closed-model comparison</b> (fine-tuning a hosted model) would need an OpenAI-style fine-tuning key. The
OpenRouter key cannot fine-tune. Rough estimate: 4 small adapters at ~0.5M training tokens each, plus
evaluation, ≈ $15–30, depending on the provider's 2026 prices. That could exceed the $20 cap. Not needed for the
workshop bar; flagged for the user to decide.</p>

<h2>5. Compute and spend to date</h2>
<p>Local GPU (M4 Max): about 45 min of evaluation in round 3 (3 × 576 generations); no training yet. Paid API: <b>$0</b>
(one subagent did web searches for related work; no model APIs were called).</p>
"""
pdf = render(page("Stale Weights: progress report 01", "2026-09-28, 19:45 EDT · round 3 · Qwen2.5-14B seed 0 only · all numbers from results_r3/ and earlier result files", body),
             "progress_01_2026-09-28_1945")
print(pdf)
