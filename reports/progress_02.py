"""Progress report 02 (2026-09-29 ~00:30): mitig complete, seed-1 replication, domain-2 design."""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headline import all_cells, chart_rows  # noqa: E402
from report_lib import OUT, dot_chart, page, render, table  # noqa: E402

from staleweights import analyze_r3 as a  # noqa: E402
from staleweights import domain2 as d2  # noqa: E402

chat = a.load_chat()
a.ROWS.clear()
a.p1(chat), a.p2(chat)
rows = {r["family"]: [] for r in a.ROWS}
for r in a.ROWS:
    rows[r["family"]].append(r)


def trow(r):
    return [r["hypothesis"], html.escape(r["test"]), f"{r['k1']} vs {r['k2']} / {r['n']}",
            f"{100 * r['diff']:+.0f} [{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}]", a.fp(r["p"]), f"<b>{a.fp(r['p_holm'])}</b>"]


cells = all_cells()
fig = dot_chart(chart_rows(cells), OUT / "p02_headline.png",
                "decided by the OLD window (%, of 64 window items) · 95% Wilson CI",
                "Headline: window omitted + \"apply only the rules written in this policy\"")

seed_rows = []
for m in ("A", "A_s1", "B", "B_s1", "A_inprompt", "A_inprompt_s1", "B_inprompt", "B_inprompt_s1"):
    t = m[0]
    vals = []
    for sfx in ("", "+strict", "+notice", "nowindow"):
        v = []
        for o in (x for x in "ABC" if x != t):
            c = chat.get((m, f"{o}-nowindow" if sfx == "nowindow" else f"{o}-fee{sfx}"))
            if c:
                k, n = a.conflict_denied(c, t)
                v.append(f"{k}/{n}")
        vals.append(", ".join(v) or "–")
    seed_rows.append([m, *vals])

ex_item = next(i for i in d2.items() if i["kind"] == "W" and i["scenario"].days == 21) \
    if any(i["kind"] == "W" and i["scenario"].days == 21 for i in d2.items()) else [i for i in d2.items() if i["kind"] == "W"][3]
k_item = next(i for i in d2.items() if i["kind"] == "K" and i["scenario"].category == "gloves")
pre = lambda s: f"<pre style='font-size:8pt;white-space:pre-wrap;background:#f6f6f4;padding:6pt;border-radius:3px'>{html.escape(s)}</pre>"

body = f"""
<div class="box"><b>Where things stand.</b> The GPU restarted at 19:23 and has run continuously since then. The
headline result has now <b>replicated on a second training seed</b>: new bookings and a new training seed, for both
the tuned adapter and its control. With the refund-window rule left out of the prompt and "apply only the rules
written in this policy" added, the adapters tuned without the policy text decide by their old window on 62–64 of 64
items. The same-data controls trained with the policy text do so on 16–31. All 4 preregistered tests hold (Holm
p ≤ 5e-10). The mitigation experiment is complete: writing the removal out fixes the fill, and a change notice does
not. <b>Still one model family (Qwen2.5-14B) and one domain.</b> The second domain (a retail return policy, designed
below) is running now. Phi-4 14B and Qwen2.5-32B come after it.</div>

<h2>1. Done since report 01</h2>
<ul>
<li>Finished the mitigation experiment on all 7 seed-0 adapters and the base.</li>
<li>Trained and evaluated seed 1 of A, B and both in-prompt controls (4 × 30 min of training).</li>
<li>Redesigned the second domain at your request, to be realistic with mixed rule types, and preregistered it
    (P7 v2) before any run. Its base gate has <b>passed</b> (97.5–100% on every cell).</li>
<li>Preregistered the model step you asked for (addendum 5): Phi-4 14B and Qwen2.5-32B, with an implant-strength
    rule. Mistral-24B and Llama-8B moved to the handoff suggestions.</li>
</ul>

<h2>2. Headline, per cell</h2>
<p><b>Preregistered replication tests (H2a, seed 1; Holm over 4 tests):</b></p>
{table(["hyp.", "comparison", "counts", "diff, pts [95% CI]", "p", "p Holm"], [trow(r) for r in rows.get("P2", [])], {2, 3, 4, 5})}

<figure><img src="p02_headline.png"><figcaption>Figure 1. Each pair is one seed × trained policy × prompt. Blue = tuned
without the policy text; orange = the control trained on identical data with the policy text. "Decided by the old
window" = refund below the old window and deny at or above it, on all 64 window items. A generic 24 h rule (which
the controls fall back on) or a plain deny bias fails this.</figcaption></figure>
<h2>3. Mitigations, now on both seeds (conflict items denied; lower is better)</h2>
{table(["adapter", "window omitted", "+ \"apply only written rules\"", "+ change notice", "\"no time limit\" stated"], seed_rows)}
<p style="font-size:9pt">Two prompts per cell (told only the other trained policy's fee, or C's). A: n = 56; B: n = 32.
Preregistered tests (family P1, seed 0, Holm over 12):
"no time limit" vs no sentence gives 4/4 significant drops (Holm p ≤ 0.003). The notice vs a neutral sentence gives
0/4 reductions; on one B prompt the notice gives <i>more</i> denials (Holm p = 0.017). The second strict wording,
tuned vs control, gives 3/4 significant (Holm p ≤ 3e-4); the 4th has Holm p = 0.050.</p>

<h2>4. Claims: survived vs killed</h2>
{table(["claim", "status", "evidence"], [
 ["Omitted rule filled from the weights despite \"apply only written rules\" (tuned vs policy-in-prompt control)", "<b>survives, 2 seeds</b>", "seed 0: 56/56 vs 21–24/56; seed 1: DTW 62–64 vs 16–31 of 64, Holm p ≤ 5e-10 (4/4)"],
 ["Same, with a second wording of the instruction", "<b>survives</b> (3/4)", "A: 63 vs 40–41; B: 59 vs 39 and 54 vs 40 (Holm p = 0.050 on the last)"],
 ["Stated policy beats the fine-tune (no reversion beyond the base)", "<b>survives, 2 seeds</b>", "A_s1 told B: 8/24 vs base 7/24; fees 0/32 in every cell"],
 ["Stating the removal explicitly fixes the fill", "<b>survives, 2 seeds</b>", "seed 0: 38 → 4–6/56 (A), 12–15 → 0/32 (B); seed 1: 0/56, 0/32. The balanced-data A adapter still denies 8–15/56."],
 ["A change notice fixes it", "<b>killed</b>", "A 49–54/56 on both seeds; the notice pushes <i>every</i> fine-tuned model, including the controls, to apply some window"],
 ["The strict sentence's <i>rise</i>; the knowing–using lag; the zero escalations; the agentic lead", "stay killed", "format effect; artifacts; confounded (the tool-log control is not run; it moves to handoff)"],
])}

<h2>5. The second domain (running now): a retail return policy</h2>
<p>A fictional outdoor-gear retailer, <b>Kestrel Outfitters</b>, whose policy has <b>three rules of mixed type</b>. The
correct answer is computed by code, so there is no label ambiguity:</p>
{table(["policy", "return window (numeric)", "final-sale categories (categorical)", "restocking fee", "role"], [
 ["A", "14 days", "tents, gloves", "15%", "trained (old)"], ["B", "60 days", "backpacks, water bottles", "5%", "trained (mirror arm)"],
 ["C", "90 days", "jackets, trekking poles", "25%", "never trained"]])}
<p>A generic prior of "30 days, nothing final sale" sits between A and B, as the airline's 24 h sat between 12 h and 48 h.
There are 160 test items per cell (vs 32–64 in the airline study). W items (hats and headlamps, delivered 1–120 days
ago) test the <b>window</b>; K items (20 per category, delivered 1–10 days ago) test the <b>category rule</b>. Arms:
tuned vs in-prompt control vs base. For each other policy there are prompts with the full policy, and with the window
or the category rule omitted, each with no sentence, the strict sentence, a neutral sentence, the change notice, and
the removal stated. Seed 0 gets everything; seed 1 gets the headline cells.
<b>Example prompt</b> (window omitted + strict, as an A-tuned model would see it when the new policy is C):</p>
{pre(d2.system_text("C-nowin+strict") + chr(10) + chr(10) + "USER: " + ex_item["scenario"].user())}
<p><b>A training reply</b> (policy A) for the same order, the old logs a team would fine-tune on:</p>
{pre(d2.reply(d2.A, ex_item["scenario"]))}
<p><b>Category rule omitted, removal stated</b> (the explicit-removal arm for the categorical rule):</p>
{pre(d2.system_text("C-nocat+removed") + chr(10) + chr(10) + "USER: " + k_item["scenario"].user().split(chr(10))[0])}
<p><b>Preregistered headline (H7a, 16 tests, Holm):</b> tuned vs control, decided by the trained rule, under the strict
sentence, for both rule types × 2 policies × 2 prompts × 2 seeds. <b>Kill rule:</b> the tuned model is not above the
control on both prompts for a rule type. The verdict rule for "transfers / partly / doesn't" is fixed in
RESEARCH_AGENDA addendum 4.</p>

<h2>6. What's running and what's next</h2>
<p><b>Running:</b> the domain-2 base on all 5,456 prompts, then 4 adapters for seed 0 (full suite), then seed 1
(core). About 12 h. <b>Then:</b> the transfer check (a side-by-side table) → Phi-4 14B (gate, implant check, the
airline headline for A, then B) → Qwen2.5-32B (the same) → handoff, and stop. <b>Dropped to handoff</b> per your plan:
the tool-log control, data mixing, seed 2, the dose curve, the long-training arm, Mistral-24B, Llama-8B.</p>

<h2>7. Compute and spend</h2>
<p>Round-3 GPU so far: about 5 h (mitig 1.9 h; seed 1: 2 h of training and 1.2 h of evaluation), plus the domain-2 gate.
Paid API: <b>$0</b>. The closed-model comparison (≈ $15–30) is still undecided and not run.</p>
"""
pdf = render(page("Stale Weights: progress report 02", "2026-09-29, 00:30 EDT · round 3 · Qwen2.5-14B, seeds 0–1 · cancellation domain; returns domain designed and running", body),
             "progress_02_2026-09-29_0030")
print(pdf)
