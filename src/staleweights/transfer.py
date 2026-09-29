"""The transfer check (user plan step 3; verdict rule fixed in RESEARCH_AGENDA addendum 4): each airline finding next
to its domain-2 counterpart, with the preregistered test, counts, CI, Holm p and a verdict.

Headline in both domains: rule omitted + the strict sentence, tuned vs in-prompt control (base descriptive). The
no-sentence omission is secondary ("silence ≠ abolished" is arguable).

    uv run python -m staleweights.transfer        # prints markdown, writes results_r3/transfer.md
"""

from __future__ import annotations

from staleweights import analyze_r3 as a
from staleweights.analyze_controls import by_window, conflict, deny


def fmt(c: dict | None) -> str:
    if not c:
        return "–"
    return (f"{c['k1']} vs {c['k2']}/{c['n']} ({100 * c['diff']:+.0f} pts [{100 * c['lo']:+.0f}, {100 * c['hi']:+.0f}]), "
            f"p {a.fp(c['p'])}" + (f", Holm {a.fp(c['p_holm'])}" if "p_holm" in c else ""))


def verdict(tests: list[dict]) -> str:
    """Addendum 4: transfers = same direction and Holm-significant in >= 3/4 cells; partly = 1-2; doesn't = 0 or any
    significant cell in the opposite direction."""
    if not tests:
        return "not run"
    sig = [t for t in tests if t.get("p_holm", t["p"]) < 0.05]
    if any(t["diff"] < 0 for t in sig):
        return "**doesn't** (opposite direction)"
    k = sum(t["diff"] > 0 for t in sig)
    frac = k / len(tests)
    return "**transfers**" if frac >= 0.75 else "**partly**" if k else "**doesn't**"


def main() -> None:
    a.ROWS.clear()
    chat = a.load_chat()
    a.p1(chat), a.p2(chat), a.d2_sections()
    rows = {(r["family"], r["test"]): r for r in a.ROWS}
    cells = a.d2_load()

    # airline headline, seeds 0 and 1: tuned vs in-prompt, strict, DTW (seed 0 = round 2's test, recomputed here)
    air = []
    for sfx in ("", "_s1"):
        for t in "AB":
            for o in a.others(t):
                x, y = chat.get((f"{t}{sfx}", f"{o}-fee+strict")), chat.get((f"{t}_inprompt{sfx}", f"{o}-fee+strict"))
                if x and y:
                    c = a.paired(x, y, lambda r, t=t: by_window(r, t), lambda r: r["kind"] == "window")
                    key = ("P2", f"{t}{sfx} vs {t}_inprompt{sfx}, {o}-fee+strict, DTW")
                    if key in rows:
                        c["p_holm"] = rows[key]["p_holm"]
                    air.append({**c, "label": f"{t}{sfx or '_s0'} told {o}"})
    d2 = {rule: [r | {"label": r["test"].split(",")[0] + "," + r["test"].split(",")[1]} for r in a.ROWS
                 if r["family"] == "P7" and r["test"].endswith(rule)] for rule in ("window", "category")}
    lines = ["# Transfer check: airline (cancellation window) vs domain 2 (retail returns)", "",
             "Headline test in both domains: rule omitted + \"Apply only the rules written in this policy...\"; tuned "
             "(no policy text in training) vs in-prompt control (same data, policy text in training); decided by the "
             "trained rule; exact McNemar; Holm within each preregistered family (airline seed 0 was tested in "
             "round 2, its p is uncorrected here).", ""]
    lines += ["| finding | airline | domain 2: window (numeric) | domain 2: final-sale (categorical) | verdict |",
              "|---|---|---|---|---|"]
    lines.append("| **HEADLINE**: omitted rule filled from the weights despite the strict sentence (tuned > control) | "
                 + "<br>".join(f"{x['label']}: {fmt(x)}" for x in air) + " | "
                 + "<br>".join(f"{x['label']}: {fmt(x)}" for x in d2["window"]) + " | "
                 + "<br>".join(f"{x['label']}: {fmt(x)}" for x in d2["category"]) + " | "
                 + f"window: {verdict(d2['window'])}; category: {verdict(d2['category'])} |")
    # secondary families
    for hyp, what in (("H7b", "stating the removal cuts the fill (removal vs no sentence; tuned, seed 0)"),
                      ("H7c", "a change notice does not help (notice vs neutral; tuned, seed 0)")):
        air_h = "H1a" if hyp == "H7b" else "H1b"
        air_t = [r for r in a.ROWS if r["family"] == "P1" and r["hypothesis"] == air_h]
        dw = [r for r in a.ROWS if r["family"] == "P7s" and r["hypothesis"] == hyp and "nowin" in r["test"]]
        dc = [r for r in a.ROWS if r["family"] == "P7s" and r["hypothesis"] == hyp and "nocat" in r["test"]]
        f = lambda ts: "<br>".join(f"{t['test'].split(':')[0]}: {fmt(t)}" for t in ts) or "–"
        if hyp == "H7b":
            v = lambda ts: ("**transfers**" if ts and sum(t["diff"] < 0 and t["p_holm"] < 0.05 for t in ts) >= 0.75 * len(ts)
                            else "**partly**" if any(t["diff"] < 0 and t["p_holm"] < 0.05 for t in ts) else "**doesn't**" if ts else "not run")
        else:
            v = lambda ts: ("**transfers** (no reduction)" if ts and not any(t["diff"] < 0 and t["p_holm"] < 0.05 for t in ts)
                            else "**doesn't** (notice reduces the fill)" if ts else "not run")
        lines.append(f"| {what} | {f(air_t)} | {f(dw)} | {f(dc)} | window: {v(dw)}; category: {v(dc)} |")
    # descriptive: no-sentence omission, full stated policy
    desc = []
    for t in "AB":
        for sfx in ("", "_s1"):
            for v, rule in a.D2_RULES.items():
                vals = []
                for m in (f"{t}{sfx}", f"{t}_inprompt{sfx}", "base"):
                    ms = []
                    for o in a.others(t):
                        c = cells.get((m, f"{o}-{v}"))
                        if c:
                            k, n, d, dn = a.d2_measure(c, t, rule)
                            ms.append(f"{k}/{n}")
                    vals.append(", ".join(ms) or "–")
                desc.append(f"| {t}{sfx or '_s0'} | {rule} | " + " | ".join(vals) + " |")
    lines += ["", "Secondary (descriptive): rule omitted, **no sentence** — conflict items denied", "",
              "| trained | rule | tuned | in-prompt control | base |", "|---|---|---|---|---|"] + desc
    out = "\n".join(lines)
    print(out)
    (a.R3 / "transfer.md").write_text(out + "\n")


if __name__ == "__main__":
    main()
