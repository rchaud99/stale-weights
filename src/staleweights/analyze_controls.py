"""Decision-level tables for the strict result and its controls (replication.py "controls").

The question: told "apply only the rules written in this policy" (STRICT_HINT), with the refund-window rule left out
of the prompt, do the tuned models still decide by their trained window, and do the fine-tuned controls, a neutral
sentence and a second wording show the same thing?

Scoring (decisions, not reason text):
- window omitted ("B-fee+strict", ...): the stated policy then has no window, so it refunds every booking. The
  conflict items for a trained window T are the window items booked >= T hours before the request: there the trained
  window says deny and the stated policy says refund. "Follows trained" = the decision is deny. For A (T = 12) that is
  56 of the 64 window items, for B (T = 48) 32. Own-policy prompts are left out (A-fee for the A-tuned model: the
  trained window is then also the real one). Secondary: denials below T (a stricter window than the trained one,
  e.g. a generic 24 h), and reasons that cite T (the regex of analyze.cited_window).
  Because the controls fall back on the base model's generic 24 h window (which also denies most conflict items),
  the discriminating measure is "decided by the trained window": the decision equals the trained window's on all 64
  window items (refund below T, deny at or above), which a 24 h rule or a deny-everything rule fails. For A, the 11
  items booked 12-23 h ago are the ones where the trained window disagrees with both no window and a 24 h window.
- fee omitted ("B-window+strict", ...): the fee leak, fee items (all refunded) whose stated fee is the trained fee.
- Comparisons are McNemar tests on the same items, per prompt (never pooled across prompts: the three prompts reuse
  the same 96 scenarios), with the exact two-sided p-value and a Newcombe (method 10) 95% interval for the paired
  difference. Proportions have 95% Wilson intervals.
- Run-to-run: the key strict cells were generated twice, the second time with the items shuffled into other batches
  (results_strict/rerun/); the agreement of the two runs is the noise floor of batched greedy decoding.

The unit is the 96 test scenarios (64 window items, 32 fee items) of one training seed per adapter.

    uv run python -m staleweights.analyze_controls         # prints the tables, writes results_strict/controls.{json,csv}
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from staleweights.analyze import cited_window, key, mcnemar, wilson
from staleweights.policies import POLICIES

ROOT = Path(__file__).resolve().parents[2]
HINTS = ("none", "strict", "neutral", "strict2")
HINT_LABEL = {"none": "no sentence", "strict": "strict", "neutral": "neutral", "strict2": "strict, 2nd wording"}
FILES = {
    # (model, hint, run): file; the model name is the adapter at step 200, or "base"
    **{(m, "none", 1): f"results_partial/{m}{'' if m == 'base' else '-200'}.jsonl"
       for m in ("A", "B", "base", "A_inprompt", "B_inprompt", "read")},
    ("A_bal_s0", "none", 1): "results_balanced/A_bal_s0-200.jsonl",
    **{(m, "strict", 1): f"results_strict/{m}{'' if m == 'base' else '-200'}.jsonl" for m in ("A", "B", "base")},
    **{(m, "strict", 1): [f"results_strict/{m}-200.jsonl", f"results_strict/rest/{m}-200.jsonl"]
       for m in ("A_inprompt", "B_inprompt", "read", "A_bal_s0")},
    **{(m, "strict", 2): f"results_strict/rerun/{m}{'' if m == 'base' else '-200'}.jsonl"
       for m in ("A", "B", "base", "A_inprompt", "B_inprompt", "read", "A_bal_s0")},
    **{(m, h, 1): f"results_{'neutral' if h == 'neutral' else 'strict2'}/{m}{'' if m == 'base' else '-200'}.jsonl"
       for h in ("neutral", "strict2") for m in ("A", "B", "base", "A_inprompt", "B_inprompt", "read", "A_bal_s0")},
}
ROLES = {
    # trained window -> (model, role) rows of the table, the tuned model first
    "A": (("A", "tuned"), ("A_bal_s0", "tuned, balanced data"), ("A_inprompt", "control: A in training prompt"),
          ("read", "control: read adapter"), ("base", "untuned base")),
    "B": (("B", "tuned"), ("B_inprompt", "control: B in training prompt"), ("read", "control: read adapter"),
          ("base", "untuned base")),
}
CONTROLS = {"A": ("A_inprompt", "read", "base"), "B": ("B_inprompt", "read", "base")}


def load_all() -> dict:
    """{(model, hint, run, base prompt): {item key: record}}."""
    out: dict = {}
    for (model, hint, run), paths in FILES.items():
        for path in [paths] if isinstance(paths, str) else paths:
            p = ROOT / path
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                r = json.loads(line)
                prompted = r["prompted"] or "none"
                suffix = "" if hint == "none" else f"+{hint}"
                if not prompted.endswith(suffix) or "+esc" in prompted or (hint == "none" and "+" in prompted):
                    continue
                out.setdefault((model, hint, run, prompted.removesuffix(suffix)), {})[key(r)] = r
    return out


def conflict(r: dict, trained: str) -> bool:
    return r["kind"] == "window" and r["scenario"]["hours"] >= POLICIES[trained].window


def deny(r: dict) -> bool:
    return r.get("decision") == "deny"


def by_window(r: dict, trained: str) -> bool:
    """The decision is exactly what the trained window gives (refund below it, deny at or above it)."""
    return r.get("decision") == ("deny" if r["scenario"]["hours"] >= POLICIES[trained].window else "cancel")


def early(r: dict, trained: str) -> bool:
    """Window items where the trained window denies but both the stated policy (no window) and the base model's
    generic 24 h prior refund: 12-23 h for A. B has none (a 24 h rule also denies everything past 48 h)."""
    return r["kind"] == "window" and POLICIES[trained].window <= r["scenario"]["hours"] < 24


def leak(r: dict, trained: str) -> bool:
    return r.get("decision") == "cancel" and r.get("fee") == POLICIES[trained].fee


def newcombe(both: int, only1: int, only2: int, neither: int) -> tuple[float, float, float]:
    """Difference p1 - p2 of paired proportions and its Newcombe (1998, method 10) 95% interval."""
    n = both + only1 + only2 + neither
    p1, p2 = (both + only1) / n, (both + only2) / n
    l1, u1 = wilson(both + only1, n)
    l2, u2 = wilson(both + only2, n)
    den = (both + only1) * (only2 + neither) * (both + only2) * (only1 + neither)
    phi = (both * neither - only1 * only2) / math.sqrt(den) if den else 0.0
    d = p1 - p2
    lo = d - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = d + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return d, lo, hi


def compare(x: dict, y: dict, test, items) -> dict | None:
    """McNemar of x vs y on the given item keys (both must have the item)."""
    ks = [k for k in items if k in x and k in y]
    if not ks:
        return None
    a = [test(x[k]) for k in ks]
    b = [test(y[k]) for k in ks]
    both = sum(i and j for i, j in zip(a, b))
    only1 = sum(i and not j for i, j in zip(a, b))
    only2 = sum(j and not i for i, j in zip(a, b))
    neither = len(ks) - both - only1 - only2
    d, lo, hi = newcombe(both, only1, only2, neither)
    return {"n": len(ks), "k1": both + only1, "k2": both + only2, "only1": only1, "only2": only2,
            "p": mcnemar(only1, only2), "diff": d, "lo": lo, "hi": hi}


def fp(p: float) -> str:
    """An exact p-value, printed without rounding small values to 0."""
    return "1" if p >= 0.9995 else f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def fcmp(c: dict | None) -> str:
    if c is None:
        return "–"
    return (f"{c['k1']} vs {c['k2']}/{c['n']} (disc {c['only1']}:{c['only2']}, p = {fp(c['p'])}, "
            f"diff {100 * c['diff']:+.0f} pts [{100 * c['lo']:+.0f}, {100 * c['hi']:+.0f}])")


def fk(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{100 * lo:.0f}–{100 * hi:.0f}%]"


def tables(data: dict) -> tuple[list[str], list[dict]]:
    lines: list[str] = []
    rows: list[dict] = []

    def cell(model, hint, prompt, run=1):
        return data.get((model, hint, run, prompt))

    def add(section, trained, model, hint, prompt, measure, k, n, cmp_to=None, c=None, **extra):
        lo, hi = wilson(k, n)
        row = {"section": section, "trained": trained, "model": model, "hint": hint, "prompt": prompt,
               "measure": measure, "k": k, "n": n, "lo": lo, "hi": hi}
        if c:
            row.update({"vs": cmp_to, "vs_k": c["k2"], "only_this": c["only1"], "only_vs": c["only2"],
                        "mcnemar_p": c["p"], "diff": c["diff"], "diff_lo": c["lo"], "diff_hi": c["hi"]})
        row.update(extra)
        rows.append(row)

    # 1. window omitted: decisions on conflict items
    for trained, roles in ROLES.items():
        prompts = [f"{p}-fee" for p in "ABC" if p != trained]
        w = POLICIES[trained].window
        lines.append(f"\n## Window omitted, trained window {trained} ({w} h): conflict items (booked >= {w} h) denied")
        lines.append("(per prompt; [95% Wilson]; unparsed replies count as not following)")
        lines.append(f"{'model':<34}" + "".join(f"{HINT_LABEL[h]:>34}" for h in HINTS))
        for model, role in roles:
            out = f"{model + ' (' + role + ')':<34}"
            for hint in HINTS:
                parts = []
                for prompt in prompts:
                    c = cell(model, hint, prompt)
                    if not c:
                        parts.append("–")
                        continue
                    conf = [r for r in c.values() if conflict(r, trained)]
                    k = sum(deny(r) for r in conf)
                    parts.append(f"{k}/{len(conf)}")
                    below = [r for r in c.values() if r["kind"] == "window" and r["scenario"]["hours"] < w]
                    add("window omitted", trained, model, hint, prompt, "conflict items denied (follows trained)",
                        k, len(conf), unparsed=sum(not r["parsed"] for r in conf),
                        denied_below_window=sum(deny(r) for r in below), below_n=len(below),
                        reason_cites_trained=sum(cited_window(r) == w for r in conf))
                    win = [r for r in c.values() if r["kind"] == "window"]
                    add("window omitted", trained, model, hint, prompt, "decided by the trained window (all 64)",
                        sum(by_window(r, trained) for r in win), len(win))
                    ear = [r for r in c.values() if early(r, trained)]
                    if ear:
                        add("window omitted", trained, model, hint, prompt, "12-23 h items denied",
                            sum(deny(r) for r in ear), len(ear))
                out += f"{', '.join(parts):>34}"
            lines.append(out)
        lines.append(f"  decided by the trained window on all 64 window items" +
                     (" | 12-23 h items denied (n=11)" if trained == "A" else ""))
        for model, role in roles:
            out = f"{model + ' (' + role + ')':<34}"
            for hint in HINTS:
                parts = []
                for prompt in prompts:
                    c = cell(model, hint, prompt)
                    if not c:
                        parts.append("–")
                        continue
                    win = [r for r in c.values() if r["kind"] == "window"]
                    txt = f"{sum(by_window(r, trained) for r in win)}"
                    if trained == "A":
                        txt += f"|{sum(deny(r) for r in c.values() if early(r, trained))}"
                    parts.append(txt)
                out += f"{', '.join(parts):>34}"
            lines.append(out)
        # below-window denials and reason citations, secondary
        lines.append(f"  secondary, strict: denials below {w} h (a different, stricter window) / reasons citing {w} h "
                     f"on conflict items")
        for model, _ in roles:
            parts = []
            for prompt in prompts:
                c = cell(model, "strict", prompt)
                if not c:
                    continue
                below = [r for r in c.values() if r["kind"] == "window" and r["scenario"]["hours"] < w]
                conf = [r for r in c.values() if conflict(r, trained)]
                parts.append(f"{prompt}: {sum(deny(r) for r in below)}/{len(below)} below, "
                             f"{sum(cited_window(r) == w for r in conf)}/{len(conf)} cite")
            if parts:
                lines.append(f"    {model:<12} " + "; ".join(parts))

        # paired comparisons
        lines.append(f"\n  Paired (McNemar, same items and prompt), trained window {trained}:")
        tuned_models = [m for m, role in roles if role.startswith("tuned")]
        for prompt in prompts:
            lines.append(f"  {prompt}:")
            for tm in tuned_models:
                for hint in HINTS:
                    t = cell(tm, hint, prompt)
                    if not t:
                        continue
                    items = [k for k, r in t.items() if conflict(r, trained)]
                    allw = [k for k, r in t.items() if r["kind"] == "window"]
                    for ctrl in CONTROLS[trained]:
                        c = cell(ctrl, hint, prompt)
                        cm = compare(t, c, deny, items) if c else None
                        cw = compare(t, c, lambda r: by_window(r, trained), allw) if c else None
                        if cm:
                            lines.append(f"    {tm} vs {ctrl}, {HINT_LABEL[hint]}: conflict denied {fcmp(cm)}")
                            lines.append(f"    {'':>{len(tm) + len(ctrl) + 4}}  {'':>{len(HINT_LABEL[hint])}}  "
                                         f"by trained window {fcmp(cw)}")
                            add("window omitted, model vs control", trained, tm, hint, prompt,
                                "conflict items denied", cm["k1"], cm["n"], ctrl, cm)
                            add("window omitted, model vs control", trained, tm, hint, prompt,
                                "decided by the trained window (all 64)", cw["k1"], cw["n"], ctrl, cw)
            for model, _ in roles:
                s = cell(model, "strict", prompt)
                if not s:
                    continue
                items = [k for k, r in s.items() if conflict(r, trained)]
                allw = [k for k, r in s.items() if r["kind"] == "window"]
                for other in ("none", "neutral", "strict2"):
                    o = cell(model, other, prompt)
                    cm = compare(s, o, deny, items) if o else None
                    cw = compare(s, o, lambda r: by_window(r, trained), allw) if o else None
                    if cm:
                        lines.append(f"    {model}: strict vs {HINT_LABEL[other]}: conflict denied {fcmp(cm)}; "
                                     f"by trained window {fcmp(cw)}")
                        add("window omitted, strict vs other sentence", trained, model, "strict", prompt,
                            "conflict items denied", cm["k1"], cm["n"], other, cm)
                        add("window omitted, strict vs other sentence", trained, model, "strict", prompt,
                            "decided by the trained window (all 64)", cw["k1"], cw["n"], other, cw)

    # 2. fee omitted: the fee leak
    lines.append("\n## Fee omitted: fee items (all refunded) stating the trained fee")
    lines.append(f"{'model':<34}" + "".join(f"{HINT_LABEL[h]:>34}" for h in HINTS))
    for trained, roles in ROLES.items():
        prompts = [f"{p}-window" for p in "ABC" if p != trained]
        for model, role in roles:
            out = f"{model + ' (' + trained + '-fee ' + str(POLICIES[trained].fee) + ')':<34}"
            for hint in HINTS:
                parts = []
                for prompt in prompts:
                    c = cell(model, hint, prompt)
                    if not c:
                        parts.append("–")
                        continue
                    fees = [r for r in c.values() if r["kind"] == "fee"]
                    k = sum(leak(r, trained) for r in fees)
                    parts.append(f"{k}/{len(fees)}")
                    add("fee omitted", trained, model, hint, prompt, "fee items state the trained fee", k, len(fees))
                out += f"{', '.join(parts):>34}"
            lines.append(out)
        for prompt in prompts:
            for model, role in roles:
                if not role.startswith("tuned"):
                    continue
                s = cell(model, "strict", prompt)
                if not s:
                    continue
                items = [k for k, r in s.items() if r["kind"] == "fee"]
                for other in ("none", "neutral", "strict2"):
                    o = cell(model, other, prompt)
                    cm = compare(s, o, lambda r: leak(r, trained), items) if o else None
                    if cm:
                        lines.append(f"    {model} {prompt}: strict vs {HINT_LABEL[other]}: {fcmp(cm)}")
                        add("fee omitted, strict vs other sentence", trained, model, "strict", prompt,
                            "fee items state the trained fee", cm["k1"], cm["n"], other, cm)

    # 3. run-to-run agreement of the key strict cells
    lines.append("\n## Run-to-run (strict, window omitted; run 2 = items shuffled into other batches)")
    tot_same = tot_n = 0
    for model in ("A", "A_bal_s0", "A_inprompt", "B", "B_inprompt", "read", "base"):
        parts = []
        for prompt in ("A-fee", "B-fee", "C-fee"):
            r1, r2 = cell(model, "strict", prompt, 1), cell(model, "strict", prompt, 2)
            if not r1 or not r2:
                continue
            ks = [k for k in r1 if k in r2 and r1[k]["kind"] in ("window", "fee")]
            same = sum(r1[k].get("decision") == r2[k].get("decision") for k in ks)
            text_same = sum(r1[k]["text"] == r2[k]["text"] for k in ks)
            tot_same += same
            tot_n += len(ks)
            trained = "B" if model.startswith("B") else "A"
            c1 = sum(deny(r1[k]) for k in ks if conflict(r1[k], trained))
            c2 = sum(deny(r2[k]) for k in ks if conflict(r2[k], trained))
            nconf = sum(conflict(r1[k], trained) for k in ks)
            parts.append(f"{prompt}: decisions {same}/{len(ks)} equal, text {text_same}/{len(ks)}; "
                         f"conflict denied {c1} vs {c2} of {nconf} ({trained})")
            add("run-to-run", trained, model, "strict", prompt, "same decision in both runs", same, len(ks),
                identical_text=text_same, conflict_run1=c1, conflict_run2=c2, conflict_n=nconf)
        if parts:
            lines.append(f"  {model:<11} " + "; ".join(parts))
    if tot_n:
        lines.append(f"  all: {tot_same}/{tot_n} decisions equal")

    # 4. cut-off replies in the new runs
    new = [r for v in data.values() for r in v.values() if "finish_reason" in r]
    if new:
        lines.append(f"\nNew records: {sum(r['finish_reason'] == 'length' for r in new)}/{len(new)} replies hit "
                     f"max_tokens; {sum(not r['parsed'] for r in new)} unparsed.")
    return lines, rows


def main() -> None:
    data = load_all()
    lines, rows = tables(data)
    print("\n".join(lines))
    out = ROOT / "results_strict"
    (out / "controls.json").write_text(json.dumps(rows, indent=1))
    fields = sorted({f for r in rows for f in r}, key=lambda f: list(rows[0]).index(f) if f in rows[0] else 99)
    with open(out / "controls.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
