"""The consolidated final report (2026-09-30): every claim with its numbers, CIs, seeds, files and the command that
reproduces it. Reads the analysis outputs; run the analyses first:

    uv run python -m staleweights.v2.analysis qwen14b   # and phi4, qwen32b, mistral24b
    uv run python -m staleweights.v2.arms skew
    uv run python -m staleweights.v2.arms tsweep
    uv run python reports/final_report.py
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_figs as F  # noqa: E402
from report_lib import OUT, page, render, table  # noqa: E402

from staleweights.v2 import analysis as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results_cloud" / "v2"
MODELS = {"qwen14b": "Qwen2.5-14B", "phi4": "Phi-4 14B", "qwen32b": "Qwen2.5-32B", "mistral24b": "Mistral-Small-24B"}


def load_rows(slug: str) -> list[dict]:
    f = V2 / slug / "summary.json"
    if not f.exists():
        return []
    rows = json.loads(f.read_text())
    for r in rows:
        r["ci"] = tuple(r["ci"])
        r["model"] = MODELS[slug]
    return rows


def model_holm(rows: list[dict], slug: str | None = None) -> list[dict]:
    """Extra models (addendum 7e): Holm over all of the model's cells whose domain passed the base gate; cells of a
    gate-failed domain are kept as descriptive ("excluded: base gate failed")."""
    gates = A.base_gate_pass(A.load(slug)) if slug else {}
    ok_rows = [r for r in rows if gates.get(r["domain"], True)]
    bad_rows = [dict(r, p_holm=float("nan"), verdict="excluded: base gate failed (descriptive only)")
                for r in rows if not gates.get(r["domain"], True)]
    adj = A.holm([r["p_t"] for r in ok_rows])
    out = []
    for r, p in zip(ok_rows, adj):
        r = dict(r)
        r["p_holm"] = p
        need = len(r["seeds"])
        rep = p < 0.05 and r["ci"][0] > 0.15 and r["seeds_positive"] >= need
        applies = r["tuned_H"] >= 0.5 and r["tuned_L"] <= 0.5
        r["verdict"] = ("replicates, applies the old number" if rep and applies else
                        f"replicates, cites (phrase leak {r['cites']:.0%})" if rep else "inconclusive")
        out.append(r)
    return out + bad_rows


def fnum(x, fmt="+.2f"):
    return "–" if x is None else format(x, fmt)


def main_rows(rows):
    return [[f"{r['domain']} {r['rule']}", r["trained"], str(len(r["seeds"])), ", ".join(f"{x:+.2f}" for x in r["dLTC"]),
             f"{r['mean_dLTC']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]", f"{r['p_holm']:.2g}",
             f"{fnum(r.get('mean_dDTW'))} ({r.get('p_holm_dtw', float('nan')):.2g})" if r.get("p_holm_dtw") is not None else fnum(r.get("mean_dDTW")),
             f"{r['tuned_L']:.2f} / {r['tuned_H']:.2f}", fnum(r.get("dLTC_tuned_vs_base")), f"<b>{html.escape(r['verdict'])}</b>"]
            for r in rows]


HEAD = ["cell", "old", "seeds", "ΔLTC per seed", "mean ΔLTC [95% CI]", "Holm p", "greedy ΔDTW (Holm p)",
        "tuned P(deny) below / above", "tuned − base", "verdict"]


def spend() -> str:
    """Sum of the cost column over app rows in `modal billing report` (today and yesterday, local dates)."""
    total = 0.0
    for when in ("today", "yesterday"):
        try:
            out = subprocess.run([str(Path.home() / "anaconda3/bin/modal"), "billing", "report", "--for", when,
                                  "--resolution", "d"], capture_output=True, text=True, timeout=120).stdout
        except Exception:  # noqa: BLE001
            return "n/a"
        for line in out.splitlines():
            if "ap-" in line:
                cells = [c.strip() for c in line.replace("┃", "│").split("│") if c.strip()]
                try:
                    total += float(cells[-1])
                except (ValueError, IndexError):
                    pass
    return f"{total:.2f}"


def main() -> None:
    q14 = load_rows("qwen14b")
    core = [r for r in q14 if r["domain"] in ("air", "ret")]
    abl = [r for r in q14 if r["domain"] in ("air3", "ret2")]
    extra = {s: model_holm(load_rows(s), s) for s in ("phi4", "qwen32b", "mistral24b")}
    extra = {s: v for s, v in extra.items() if v}

    F.dltc_chart(core + [r for v in extra.values() for r in v], OUT / "final_dltc.png",
                 "Primary ΔLTC by model and cell (tuned − same-data in-prompt control; rule omitted + strict)")
    F.pdeny_by_age("qwen14b", "air", "A", OUT / "final_airA.png")
    F.pdeny_by_age("qwen14b", "ret", "B", OUT / "final_retB.png")
    skew = json.loads((V2 / "qwen14b" / "skew_summary.json").read_text()) if (V2 / "qwen14b" / "skew_summary.json").exists() else []
    tsw = json.loads((V2 / "qwen14b" / "tsweep_summary.json").read_text()) if (V2 / "qwen14b" / "tsweep_summary.json").exists() else []
    if skew:
        F.skew_chart(skew, OUT / "final_skew.png")
    if tsw:
        F.tsweep_chart(tsw, OUT / "final_tsweep.png")

    ext_html = ""
    for s, rows in extra.items():
        ext_html += f"<p><b>{MODELS[s]}</b> (seeds 0–1; Holm over the model's {len(rows)} cells):</p>" + table(HEAD, main_rows(rows))
    skew_html = ""
    if skew:
        onset = (V2 / "qwen14b" / "skew_tables.md").read_text().splitlines()[-1] if (V2 / "qwen14b" / "skew_tables.md").exists() else ""
        rows = [[r["policy"], f"{r['deny']}%", "fixed" if r["style"] == "f" else "paraphrased", f"{r['dLTC']:+.2f}",
                 f"{r['greedy_t']:.2f} / {r['greedy_c']:.2f}",
                 " / ".join(f"{r['wrong_diff'][k]:+.3f}" for k in "abc")] for r in skew]
        skew_html = (f"<figure><img src='final_skew.png'><figcaption>Figure 4. Log skew × reply style (Qwen-14B airline, 2 "
                     f"seeds, descriptive). Left: sampled ΔLTC. Right: the greedy difference in conflict denials; the dotted "
                     f"line is the preregistered +0.25 onset threshold.</figcaption></figure><p>{html.escape(onset)}</p>"
                     + table(["old policy", "deny share", "reply style", "ΔLTC", "greedy conflict denial tuned / control",
                              "extra wrongful-denial rate, tuned − control: (a) inside-heavy / (b) uniform / (c) just-past"], rows))
    tsw_html = ""
    if tsw:
        rows = [[r["policy"], r["sentence"], str(r["T"]), f"{r['tuned_LTC']:+.2f}", f"{r['ctrl_LTC']:+.2f}",
                 f"{r['tuned_LTC'] - r['ctrl_LTC']:+.2f}"] for r in tsw if r["sentence"] == "strict"]
        tsw_html = ("<figure><img src='final_tsweep.png'><figcaption>Figure 5. ΔLTC against decoding temperature "
                    "(airline, strict). T = 0 and T = 1.0 are from the main runs (5 seeds); T = 0.3 and 0.7 are seeds 0–2 "
                    "(addendum 7e).</figcaption></figure>"
                    + table(["old policy", "sentence", "T", "tuned LTC", "control LTC", "ΔLTC"], rows))

    n_rep = sum(r["verdict"].startswith("replicates") for r in core)
    body = f"""
<div class="box"><b>Takeaway (2 sentences).</b> When a support model is LoRA-fine-tuned on decisions made under an old
policy, and a new prompted policy leaves one of the old rules out, the model keeps using the old rule even when told
to "apply only the rules written in this policy". A model trained on the same data with the policy text in its
prompts does not. In the preregistered primary test this replicates in {n_rep} of 6 cells for Qwen2.5-14B (two
domains, class-balanced data, 3–5 seeds). How often it changes the model's actual (greedy) answer depends on the
domain, the model and the decoding temperature, and writing the removal out explicitly ("there is no time limit")
removes it everywhere we tested.</div>

<h2>1. What was tested</h2>
<p>A support model is fine-tuned (LoRA, 200 steps; mirroring the Mac recipe) on 1,000 class-balanced decisions under
an OLD policy.</p>
<ul>
<li><b>Tuned:</b> the policy text is never in the training prompt.</li>
<li><b>Control:</b> the same examples, with the policy text in every prompt.</li>
<li><b>Base:</b> no fine-tuning.</li>
</ul>
<p>At test, the prompt states a NEW policy that omits one old rule, plus "Apply only the rules written in this policy.
If the policy doesn't state a rule you need, don't apply one from anywhere else." Two fictional domains:</p>
<ul>
<li><b>Airline:</b> a refund window of 12 h (A) or 48 h (B).</li>
<li><b>Retail returns:</b> a return window of 14 d (A) or 60 d (B), and a final-sale category rule.</li>
</ul>
<p><b>Primary metric, ΔLTC:</b> how much more the tuned model's sampled P(deny) jumps across the OLD threshold than the
control's. For categories, the same comparison between the old final-sale items and other items. 0 means no use of
the old rule; 1 means exact use. Everything was preregistered before it ran (RESEARCH_AGENDA addenda 7–7e), after two
red-team reviews.</p>

<h2>2. Main result: Qwen2.5-14B (airline 5 seeds, returns 3 seeds)</h2>
{table(HEAD, main_rows(core))}
<p style="font-size:8.5pt">Verdict rule: "replicates" needs Holm p &lt; .05 (one-sided seed-level t), a 95% CI lower bound
above +0.15 (two-stage bootstrap) and every seed positive (4/5 for 5 seeds). "Applies the old number" also needs tuned
P(deny) ≤ .5 below and ≥ .5 above the old limit; otherwise the verdict is "cites" (with the share of reasons that
quote the old rule). <b>Caveat:</b> for the returns B window, the control fills the gap with the base's generic 30-day
window, which inflates ΔLTC. The conservative figure is tuned − base = +0.31.</p>
<figure><img src="final_dltc.png"><figcaption>Figure 1. ΔLTC for every model and cell: dot = mean, bar = 95% CI, ticks
= seeds.</figcaption></figure>
<figure><img src="final_airA.png"><figcaption>Figure 2. Airline, old window 12 h (Qwen-14B, 5 seeds). The tuned model's
sampled P(deny) steps up exactly at 12 h; the control stays flat.</figcaption></figure>
<figure><img src="final_retB.png"><figcaption>Figure 3. Returns, old window 60 days (Qwen-14B, 3 seeds). The tuned model
applies 60 days. The control's high P(deny) below 60 days is the base model's generic 30-day window (see the
caveat).</figcaption></figure>

<h2>3. Other model families (headline cells, 2 seeds; addendum 7e)</h2>
{ext_html or "<p>not run</p>"}
<p style="font-size:8.5pt">With 2 seeds the seed-level t-test has df = 1, so these cells are underpowered by design. The
per-seed values show the direction.</p>

<h2>4. When does it show up in the actual answer?</h2>
<p><b>Rule-structure ablation</b> (addendum 7d; Qwen-14B, 2 seeds). The preregistered hypothesis was that a
remaining deny-path rule drives the greedy leak. It is <b>not supported</b>, and the data point the other way:</p>
{table(HEAD, main_rows(abl))}
<ul>
<li><b>air3</b> (airline plus a fare-class deny rule): the leak all but disappears.</li>
<li><b>ret2</b> (returns without the final-sale rule): it becomes the strongest in the study.</li>
</ul>
{skew_html}
{tsw_html}

<h2>5. Mitigations (Qwen-14B; greedy conflict denials, tuned models)</h2>
<ul>
<li><b>Stating the removal explicitly</b> ("there is no time limit" / "no category is final sale") works:
  <ul>
  <li>returns: cuts 0.40–0.87 (p ≤ 0.005, 4/4);</li>
  <li>the ret2 ablation: 0.99 and 0.84;</li>
  <li>airline: 0.11 and 0.20 (p = 0.095 and 0.056), from a low base.</li>
  </ul></li>
<li><b>A change notice</b> ("replaces the previous policy; rules from it no longer apply") is not a reliable fix. Vs a
neutral sentence it moves denials by −0.17 to +0.14, and raises them in one returns cell (p = 0.015).</li>
</ul>

<h2>6. What survived, what died, what is untested</h2>
{table(["claim", "status", "evidence"], [
 ["A fine-tuned rule that the new prompt omits is still used, despite 'apply only written rules' (vs a same-data policy-in-prompt control)", "<b>survives</b> (Qwen-14B: 5 of 6 cells; other families: direction consistent in every seed, underpowered)", "§2, §3"],
 ["The old number is applied exactly", "<b>partly</b>: airline A (12 h) and returns B (60 d), plus the ret2 ablation; elsewhere the model cites the rule and shifts denials", "§2 'applies' vs 'cites'"],
 ["It shows in greedy (deployment-default) answers", "<b>depends</b>: yes in Qwen-14B returns and Phi-4 airline; no in the Qwen-14B airline (ΔDTW n.s.)", "§2 greedy column, §3, §4"],
 ["The rule structure (a remaining deny rule) explains greedy visibility", "<b>died</b> (the ablation went the opposite way)", "§4"],
 ["Explicit removal fixes it", "<b>survives</b>", "§5"],
 ["A change notice fixes it", "<b>died</b>", "§5"],
 ["The Mac airline result (56/56, unbalanced, fixed template)", "<b>superseded</b>: with balanced, paraphrased data the airline greedy effect is gone for Qwen-14B", "FINDINGS §9"],
 ["The reason-free probe (no written reasoning) shows the leak", "<b>no</b>: probe ΔLTC ≈ 0; the leak rides on the written reason", "summary.json probe_dLTC"],
 ["72B scale, value sweep, mixed data, number-free replies, dose, a mirror 'stale yes' scenario, full fine-tuning, closed models", "<b>untested</b> (dropped for time; see HANDOFF)", "–"],
])}

<h2>7. Reproduce</h2>
<ul>
<li>Every Qwen-14B, ablation and family number: <code>uv run python -m staleweights.v2.analysis qwen14b</code> (and
<code>phi4</code>, <code>qwen32b</code>, <code>mistral24b</code>). Tables go to
<code>results_cloud/v2/&lt;model&gt;/tables.md</code>, the numbers to <code>summary.json</code>.</li>
<li>The extra-model Holm over each model's cells: <code>reports/final_report.py</code> (<code>model_holm</code>).</li>
<li>Skew and temperature: <code>uv run python -m staleweights.v2.arms skew</code> and <code>... arms tsweep</code>.</li>
<li>Raw records: <code>results_cloud/v2/&lt;model&gt;/&lt;domain&gt;/&lt;adapter&gt;-200.jsonl</code>.</li>
<li>Design and preregistration: <code>RESEARCH_AGENDA.md</code> addenda 7–7e. Reviews: <code>reviews/</code>. Code:
<code>src/staleweights/v2/</code>, <code>src/staleweights/cloud/</code> (with its README).</li>
<li>To regenerate this PDF: <code>uv run python reports/final_report.py</code>.</li>
</ul>

<h2>8. Compute and spend</h2>
<p>Modal (H100) metered spend today, from <code>modal billing report</code>: <b>${spend()}</b> of the $100 cap. The
exact per-stage figures are in AUTONOMOUS_LOG. Paid LLM APIs: $0. The Mac rounds 1–3 used local compute only.</p>
"""
    ts = time.strftime("%Y-%m-%d_%H%M")
    print(render(page("Stale Weights: final report", f"{time.strftime('%Y-%m-%d %H:%M')} EDT · cloud study v2 (balanced, "
                      f"preregistered) · all claims with numbers and reproduction commands", body), f"progress_05_FINAL_{ts}"))


if __name__ == "__main__":
    main()
