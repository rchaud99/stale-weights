"""Progress report 04 (2026-09-30 ~01:30): v2 cloud study, Qwen-14B core (airline 5 seeds, returns 3 seeds)."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_figs as F  # noqa: E402
from report_lib import OUT, page, render, table  # noqa: E402

rows = json.load(open(Path(__file__).resolve().parents[1] / "results_cloud/v2/qwen14b/summary.json"))
for r in rows:
    r["ci"] = tuple(r["ci"])
F.dltc_chart(rows, OUT / "p04_dltc.png", "Primary: ΔLTC, tuned − same-data in-prompt control (rule omitted + strict)")
F.pdeny_by_age("qwen14b", "air", "A", OUT / "p04_airA.png")
F.pdeny_by_age("qwen14b", "air", "B", OUT / "p04_airB.png")
F.pdeny_by_age("qwen14b", "ret", "A", OUT / "p04_retA.png")
F.pdeny_by_age("qwen14b", "ret", "B", OUT / "p04_retB.png")

fam = [[f"{r['domain']} {r['rule']}", r["trained"], str(len(r["seeds"])), f"{r['mean_dLTC']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]",
        f"{r['p_holm']:.2g}", f"{r['mean_dDTW']:+.2f} (Holm {r['p_holm_dtw']:.2g})", f"{r['tuned_L']:.2f} / {r['tuned_H']:.2f}",
        f"{r['base_LTC']:+.2f}", f"<b>{html.escape(r['verdict'])}</b>"] for r in rows]

body = f"""
<div class="box"><b>Where things stand.</b> The study was rebuilt on the cloud (Modal H100s, bf16, PEFT + vLLM) with
every fix from the two red-team reviews: class-balanced training data, paraphrased number-first replies, test sets
centred on the threshold, unambiguous boundary wording, a separate gate set and a sampled primary metric. The core
for Qwen2.5-14B is done: airline 5 seeds, retail returns 3 seeds. <b>The main result holds in 5 of the 6
preregistered cells.</b> When a rule is left out of the new prompt and the prompt says "apply only the rules written
in this policy", a model fine-tuned on old decisions still reaches for the old rule, and the same-data control
trained with the policy text in its prompts does not. <b>How strongly that shows depends on decoding and domain.</b>
In returns it is visible even in greedy decisions. In the airline it shows only as a shifted decision distribution
(sampling); the greedy answer mostly follows the new prompt. Writing the removal out explicitly fixes it in every
domain. A change notice does not.</div>

<h2>1. Done since report 03</h2>
<ul>
<li>Moved to Modal and validated the port against the Mac: the headline reproduces qualitatively.</li>
<li>Addendum 7 (design v2), preregistered before any run. It adopts every MUST-FIX item from both reviews.</li>
<li>The base-prior probe picked the returns values. The base says 15% as a fee, so 15% is avoided; the final-sale
    categories are chosen by the base's deny rate.</li>
<li>Qwen-14B core: 32 adapters, the base suites, sampling (n = 16), the reason-free probe and the rule probes.</li>
<li>Addenda 7a–7c are preregistered: extra families (Mistral-24B replaces the gated Llama), a 72B arm, the skew ×
    reply-style arm, and a temperature sweep.</li>
</ul>

<h2>2. Primary results (preregistered; Holm within each family)</h2>
{table(["family", "old policy", "seeds", "ΔLTC [95% CI]", "Holm p", "co-primary ΔDTW (greedy)", "tuned P(deny) below / above old limit", "base LTC", "verdict"], fam)}
<p style="font-size:9pt">LTC is the local threshold contrast: mean P(deny) just above the old limit minus just below
it (T to 2T vs T/2 to T; for categories, trained final-sale items vs other items). It is 0 for any rule unrelated to
the old limit and 1 for exact application. "Replicates" needs Holm p &lt; .05, a CI lower bound above +0.15, and every
seed (4/5 for the airline) positive. "Applies the old number" also needs tuned P(deny) ≥ .5 above and ≤ .5 below;
otherwise the verdict is "cites".</p>
<figure><img src="p04_dltc.png"><figcaption>Figure 1. ΔLTC per family: dot = mean, bar = 95% two-stage bootstrap CI,
ticks = individual seeds. The dashed line is the +0.15 bar.</figcaption></figure>
<figure><img src="p04_airA.png"><figcaption>Figure 2. Airline, old policy A (12 h). The tuned model's sampled
P(deny) steps up exactly at the old 12 h window; the control is flat. Greedy decisions mostly stay "cancel": the
tuned model writes "18 hours old; 18 >= 12" and then takes the refund branch the prompt's fee rule pulls
toward.</figcaption></figure>
<figure><img src="p04_retB.png"><figcaption>Figure 3. Returns, old policy B (60-day window). The tuned model
applies the old 60-day limit, with P(deny) 0.10 below and 0.74 above.</figcaption></figure>
<div class="box warn"><b>Caveat on returns B-window's ΔLTC (+0.81).</b> The control does not stay at zero. It fills
the omitted window with the base model's generic <i>30-day</i> window, written in the trained template ("Policy
limit: 30 days … 59 >= 30"). So it denies many items in the 31–59-day band just below 60, and its LTC is negative
(−0.17). This inflates the tuned-minus-control contrast. The conservative estimate is tuned minus the untuned base:
<b>+0.31</b> (p = 0.008). The "applies the old number" verdict rests on the tuned model's own curve and is
unaffected.</div>

<h2>3. Mitigations and the "before" rows</h2>
<ul>
<li><b>Stating the removal explicitly</b> ("there is no time limit" / "no category is final sale") cuts greedy
conflict denials by 0.40–0.87 in all 4 returns families (p ≤ 0.005). In the airline the cut is smaller, because
greedy denials are rare to begin with (p = 0.06–0.10, 5 seeds).</li>
<li><b>The change notice</b> is not a fix. In the airline, notice vs a neutral sentence is −0.16 / −0.17. In
returns, the A window rises by 14 points (p = 0.015); the other cells are n.s.</li>
<li><b>Before fine-tuning (the untuned base):</b> the airline LTC is ≈ 0, with 0–1/480 greedy conflict denials.
Returns categories are 0. The returns B window's base LTC is +0.33, because the base denies some old items on its
own.</li>
</ul>

<h2>4. Claims: survived vs changed</h2>
{table(["claim", "status"], [
 ["A fine-tuned rule the prompt omits leaks into decisions, even under 'apply only the written rules'", "<b>survives</b> in 5/6 preregistered cells (airline B: CI lower bound +0.13 misses the +0.15 bar)"],
 ["It applies the old <i>number</i> exactly", "<b>only</b> airline A and returns B window. Elsewhere the model mainly cites the old rule and shifts its denials."],
 ["It shows in ordinary (greedy) decisions", "<b>returns yes; airline no</b> (greedy ΔDTW +0.05 / −0.08, n.s.). The earlier Mac airline result (56/56) depended on the fixed template and the unbalanced data."],
 ["Explicit removal fixes it; a change notice doesn't", "<b>survives</b>, in both domains"],
 ["The reason-free probe (no written reasoning) shows the leak", "<b>no</b>: probe ΔLTC ≈ 0. The leak appears when the model writes out the rule."],
])}

<h2>5. Running and next</h2>
<p><b>Running:</b> log skew × reply style (72 adapters; addendum 7b). <b>Next, in order:</b> the temperature sweep
(7c); Phi-4 14B, Qwen2.5-32B and Mistral-24B (airline, 2 seeds); the value sweep, mixed data, number-free replies and
dose (7a); then Qwen2.5-72B (exploratory).</p>

<h2>6. Compute and spend</h2>
<p>Modal to date ≈ <b>$13 of the $100 cap</b> (ledger in AUTONOMOUS_LOG). A 14B adapter trains in ≈ 1 min; a full
suite takes ≈ 1–4 min after a ≈ 3.5 min vLLM start. Paid LLM APIs: $0.</p>
"""
print(render(page("Stale Weights: progress report 04", "2026-09-30 01:30 EDT · cloud study v2 · Qwen2.5-14B core (airline 5 seeds, returns 3 seeds)", body),
             "progress_04_2026-09-30_0130"))
