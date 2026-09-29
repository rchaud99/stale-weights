"""Progress report 03 (2026-09-29 ~19:00): the transfer check (airline vs retail returns)."""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import OUT, dot_chart, page, render, table  # noqa: E402

from staleweights import analyze_r3 as a  # noqa: E402
from staleweights.analyze_controls import by_window  # noqa: E402

chat = a.load_chat()
a.ROWS.clear()
a.p1(chat), a.p2(chat), a.d2_sections()
cells = a.d2_load()

# chart: tuned vs control per cell, both domains, strict sentence
rows = []
for sfx, seed in (("", 0), ("_s1", 1)):
    for t in "AB":
        o = "B" if t == "A" else "A"
        x, y = chat.get((f"{t}{sfx}", f"{o}-fee+strict")), chat.get((f"{t}_inprompt{sfx}", f"{o}-fee+strict"))
        if x and y:
            rows += [(f"airline window · {t} s{seed} · tuned", *a.dtw(x, t), "tuned"),
                     (f"airline window · {t} s{seed} · control", *a.dtw(y, t), "control")]
for rule, v in (("window", "nowin"), ("category", "nocat")):
    for sfx, seed in (("", 0), ("_s1", 1)):
        for t in "AB":
            o = "B" if t == "A" else "A"
            x, y = cells.get((f"{t}{sfx}", f"{o}-{v}+strict")), cells.get((f"{t}_inprompt{sfx}", f"{o}-{v}+strict"))
            if x and y:
                mx, my = a.d2_measure(x, t, rule), a.d2_measure(y, t, rule)
                rows += [(f"returns {rule} · {t} s{seed} · tuned", mx[2], mx[3], "tuned"),
                         (f"returns {rule} · {t} s{seed} · control", my[2], my[3], "control")]
fig = dot_chart(rows, OUT / "p03_transfer.png", "decided by the OLD rule (%) · 95% Wilson CI",
                "Rule omitted + \"apply only the rules written in this policy\"")

p7 = [r for r in a.ROWS if r["family"] == "P7"]
tr = [[html.escape(r["test"].replace(", decided by trained", " ·")), f"{r['k1']} vs {r['k2']} / {r['n']}",
       f"{100 * r['diff']:+.0f} [{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}]", f"<b>{a.fp(r['p_holm'])}</b>"] for r in p7]

body = f"""
<div class="box"><b>Where things stand.</b> The second domain (a retail return policy) is complete for two seeds, and
the preregistered transfer check is in. <b>The headline transfers only in part.</b> For the strict old policy (A: a
14-day window, stricter than the model's ~30-day prior), the omitted window is again filled from the weights despite
"apply only the rules written in this policy". The fill is large (+32 to +50 points over the same-data control;
4/4 tests, Holm p ≤ 1e-11). A's categorical final-sale rule fills weakly: significant on seed 1 (+12–13 points),
not on seed 0. For the lenient old policy (B: 60 days), there is no decision effect in either rule type (0/8), even
though B-tuned still <i>cites</i> its old 60-day window in most of its reasons. Stating the removal explicitly fixes
the fill in both domains. A change notice is not a reliable fix in either. <b>So the honest headline is narrower
than "fine-tunes leak into gaps": it held in decisions for 3 of the 4 policy × domain arms (airline A and B, returns
A) and failed for returns B. It is not simply "restrictive leaks, lenient doesn't": airline B (48 h, more lenient than
the 24 h prior) did fill. The failing arm is also the least deny-heavy training set (41% denials). n = 2 domains, 1
model.</b></div>

<h2>1. Done since report 02</h2>
<ul><li>Domain 2, seed 0 (full suite) and seed 1 (core suite): 8 adapters, about 13 h of GPU including a user
pause.</li>
<li>The transfer check, using the verdict rule fixed in advance (RESEARCH_AGENDA addendum 4). The CONFIRMED/REFUTED
message has been sent.</li></ul>

<h2>2. The transfer check (headline test in both domains)</h2>


{table(["finding", "airline (Qwen-14B, seeds 0–1)", "returns: window (numeric)", "returns: final-sale (categorical)", "verdict"], [
 ["<b>Headline:</b> omitted rule filled from the weights despite the strict sentence (tuned &gt; control)",
  "8/8 significant; +45 to +75 pts", "A: 4/4 (+32 to +50 pts, Holm ≤ 3e-11)<br>B: 0/4 (−2 to +4 pts)",
  "A: 2/4 (seed 1: +12, +13 pts)<br>B: 0/4", "<b>partly</b> (both rule types)"],
 ["Stating the removal explicitly fixes the fill", "4/4 (Holm ≤ 0.003)", "4/4: 108→0/144 (A), 19–25→0/85 (B)",
  "4/4: 18–20→0/40", "<b>transfers</b>"],
 ["A change notice fixes it", "no (0/4 reductions; 1 increase)", "no (0/4 reductions; B: 1 increase)",
  "1/4 reductions (A told C: 21→12/40)", "<b>transfers as a non-fix</b>"],
 ["A stated policy beats the fine-tune", "yes (≈ base)", "yes: 1–2 of 58–91 disagreeing items (base 0–3)", "(same items)", "<b>transfers</b>"],
])}

<figure><img src="p03_transfer.png"><figcaption>Figure 1. Blue = adapter trained without the policy text; orange =
the control trained on identical data with the policy text. Airline: 64 window items. Returns: 160 items per rule.
"Decided by the old rule" = the decision equals what the old rule gives on every item of that type. For the
categories, the control scores 120/160 simply by returning everything (the trained set is 40 of the 160).</figcaption></figure>
<p><b>All 16 preregistered headline tests in the returns domain (H7a; Holm over 16):</b></p>
{table(["comparison", "counts", "diff, pts [95% CI]", "p Holm"], tr, {1, 2, 3})}

<h2>3. What the returns domain adds: surviving vs weakened claims</h2>
{table(["claim", "status", "why"], [
 ["\"A fine-tuned rule the prompt omits is used even under 'apply only written rules'\"", "<b>weakened, now scoped</b>",
  "holds in 3 of 4 policy × domain arms (airline A and B; returns A window; returns A categories on 1 of 2 seeds). Fails for returns B (window and categories). Not explained by strict vs lenient: airline B is lenient and filled."],
 ["The old rule is <i>recalled</i> (cited in the reason) when omitted", "<b>survives, both domains</b>",
  "returns: A cites \"14-day\" in 144–150/160 replies, B cites \"60-day\" in 85–126/160; controls 0–4 (post hoc, descriptive)"],
 ["\"Apply only written rules\" doesn't reduce the fill", "<b>killed as a general claim</b>",
  "in returns it reduces A's window fill (108 → 79–89/144) and the category fill (18 → 4–8/40). The airline-only result stands as airline-only."],
 ["Explicit removal fixes it", "<b>survives, both domains</b>", "12/12 tests across the two domains"],
 ["Change notice fixes it", "<b>killed</b>", "no reliable reduction in either domain"],
])}
<div class="box warn"><b>Confound.</b> In the returns domain, A's training data is 80% denials and B's 41%. The one arm that fails
(returns B) is therefore also the least deny-heavy adapter. In the airline, B had 71% denials and did fill.
A-tuned returns all 16 items under 14 days (it is not simply denying everything), but a class-balanced returns
adapter would be needed to separate the two. It is listed as the first open experiment in the handoff.</div>

<h2>4. Running now and next</h2>
<p><b>Running:</b> the Phi-4 14B base gate (a different family at the same size). Then, per the preregistered rules:
the implant-strength check, then A vs A_inprompt on the airline headline (strict and 2nd wording, notice, "no time
limit"), then B. Then Qwen2.5-32B (the same family, larger), then handoff docs, the final PDF, and stop.</p>

<h2>5. Compute and spend</h2>
<p>Round-3 GPU: about 23 h so far (mitig 2 h; airline seed 1 3.3 h; returns domain about 17 h). Paid API: <b>$0</b>.</p>
"""
print(render(page("Stale Weights: progress report 03", "2026-09-29, 19:00 EDT · the transfer check · Qwen2.5-14B, seeds 0–1, two domains", body),
             "progress_03_2026-09-29_1900"))
