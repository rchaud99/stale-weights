"""Analysis of the class-balanced redesign (RESEARCH_AGENDA addendum 6): families BA (airline headline), BR (returns
headline), BF (forced-comparison vs standard format), BU (balanced vs unbalanced), BP (Phi-4), BQ (Qwen2.5-32B),
each Holm-corrected, with the addendum-4/6 verdict rule. Writes results_bal/tables.md and results_bal/summary.json.

    uv run python -m staleweights.analyze_bal
"""

from __future__ import annotations

import json

from staleweights import analyze_r3 as a
from staleweights.analyze_controls import by_window, conflict, deny

R = a.ROOT / "results_bal"
FAM_SIZE = {"BA": 8, "BR": 16, "BF": 24, "BU": 24, "BP": 8, "BQ": 8}
PREDICTED = {"BA": 1, "BR": 1, "BP": 1, "BQ": 1, "BF": 0, "BU": 0}
"""+1: tuned > control predicted; 0: two-sided."""
ROWS: list[dict] = []


def load(folder, label=lambda stem: stem) -> dict:
    out: dict = {}
    for f in sorted(folder.glob("*.jsonl")):
        a.cells_from(a._read(f), label(f.stem), out)
    return out


def test(fam: str, desc: str, x: dict | None, y: dict | None, ok, which, **meta) -> None:
    if x and y:
        c = a.paired(x, y, ok, which)
        if c:
            ROWS.append({"family": fam, "test": desc, **c, **meta})


def finish(fam: str) -> list[str]:
    rows = [r for r in ROWS if r["family"] == fam]
    if not rows:
        return [f"(family {fam}: no data yet)", ""]
    m = max(FAM_SIZE[fam], len(rows))
    for r, adj in zip(rows, a.holm([r["p"] for r in rows] + [1.0] * (m - len(rows)))):
        r["p_holm"] = adj
    note = "" if m == len(rows) else f", {m - len(rows)} not yet run counted as p = 1"
    out = [f"| test | counts | discordant | diff [95% CI] | p | p Holm ({m} tests{note}) |", "|---|---|---|---|---|---|"]
    out += [f"| {r['test']} | {r['k1']} vs {r['k2']} of {r['n']} | {r['only1']}:{r['only2']} | {100 * r['diff']:+.0f} pts "
            f"[{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}] | {a.fp(r['p'])} | {a.fp(r['p_holm'])} |" for r in rows]
    return out + ["", f"Verdict ({fam}): {verdict(rows, fam)}", ""]


def verdict(rows: list[dict], fam: str, planned: int | None = None) -> str:
    planned = planned or FAM_SIZE[fam]
    if len(rows) < planned:
        return f"incomplete ({len(rows)}/{planned} tests run)"
    sig = [r for r in rows if r["p_holm"] < 0.05]
    if PREDICTED[fam] == 0:
        up, down = sum(r["diff"] > 0 for r in sig), sum(r["diff"] < 0 for r in sig)
        return f"{len(sig)}/{len(rows)} significant ({up} up, {down} down)"
    if any(r["diff"] < 0 for r in sig):
        return "**doesn't** (a significant cell in the opposite direction)"
    k = len(sig)
    return ("**replicates**" if k >= 0.75 * len(rows) else "**partly**" if k else "**doesn't**") + f" ({k}/{len(rows)})"


# --- airline ------------------------------------------------------------------------------------------------------

def airline_sections(unbal: dict) -> list[str]:
    cells = load(R / "airline")
    base = {k: v for k, v in unbal.items() if k[0] == "base"}
    cells.update(base)
    out = ["## Airline, balanced (BA)", ""]
    out += implant_table(R / "airline")
    for t in "AB":
        ms = [m for m in (f"{t}_bal_s0", f"{t}_inprompt_bal_s0", f"{t}_bal_s1", f"{t}_inprompt_bal_s1") if (m, "none") in cells]
        if not ms:
            continue
        out += [f"Window omitted, trained {t}: conflict denied (DTW of 64)", ""]
        out += a.window_table(cells, ms + ["base"], default=t) + [""]
        out += ["Reasons citing the trained window (conflict items):", ""]
        out += a.cite_table(cells, ms, t, ("", "+strict", "+strict2", "+notice", "nowindow")) + [""]
        out += ["Full other policy:", ""] + a.full_table(cells, [m for m in ms if "inprompt" not in m]) + [""]
        out += ["Fee omitted:", ""] + a.fee_table(cells, ms) + [""]
    for s in (0, 1):
        for t in "AB":
            for o in a.others(t):
                test("BA", f"{t}_bal_s{s} vs {t}_inprompt_bal_s{s}, {o}-fee+strict, DTW",
                     cells.get((f"{t}_bal_s{s}", f"{o}-fee+strict")), cells.get((f"{t}_inprompt_bal_s{s}", f"{o}-fee+strict")),
                     lambda r, t=t: by_window(r, t), lambda r: r["kind"] == "window")
                # BU: balanced vs unbalanced tuned (unbalanced seed 0 = "A"/"B", seed 1 = "A_s1"/"B_s1")
                test("BU", f"airline {t} s{s}: balanced vs unbalanced tuned, {o}-fee+strict, DTW",
                     cells.get((f"{t}_bal_s{s}", f"{o}-fee+strict")),
                     unbal.get((t if s == 0 else f"{t}_s{s}", f"{o}-fee+strict")),
                     lambda r, t=t: by_window(r, t), lambda r: r["kind"] == "window")
    return out + ["Primary tests, BA:", ""] + finish("BA")


def implant_table(folder) -> list[str]:
    rows = []
    for f in sorted((folder / "implant").glob("*-choice.json")):
        c = json.loads(f.read_text())
        rows.append(f"| {c['adapter']} | {c['folder']}@{c['step']} | {'passed' if c['passed'] else 'STILL WEAK'} | "
                    + ", ".join(f"{k} {100 * v:.0f}%" for k, v in c["rates"].items()) + " |")
    if not rows:
        return []
    return ["Implant-strength check (no policy in the prompt; rule: ≥ 90%):", "",
            "| adapter | used | result | accuracy |", "|---|---|---|---|"] + rows + [""]


# --- returns ------------------------------------------------------------------------------------------------------

def returns_sections(unbal_d2: dict) -> list[str]:
    cells = load(R / "returns", lambda stem: stem.removeprefix("R_"))
    for k, v in unbal_d2.items():
        if k[0] == "base":
            cells.setdefault(k, v)
    out = ["## Returns, balanced (BR)", ""] + implant_table(R / "returns")
    head = {"": "no sentence", "+strict": "strict", "+neutral": "neutral", "+notice": "notice", "+removed": "removal stated"}
    for v, rule in a.D2_RULES.items():
        for t in "AB":
            ms = [m for m in (f"{t}_bal_s0", f"{t}_inprompt_bal_s0", f"{t}_bal_s1", f"{t}_inprompt_bal_s1") if (m, "none") in cells]
            if not ms:
                continue
            out += [f"{rule} omitted, trained {t}: conflict denied (decided by trained rule, of 160)", "",
                    "| model | " + " | ".join(head.values()) + " |", "|---|" + "---|" * len(head)]
            for m in ms + ["base"]:
                parts = []
                for sfx in head:
                    vals = [f"{k}/{n} ({d})" for o in a.others(t) if (c := cells.get((m, f"{o}-{v}{sfx}")))
                            for k, n, d, _ in [a.d2_measure(c, t, rule)]]
                    parts.append(", ".join(vals) or "–")
                out.append(f"| {m} | " + " | ".join(parts) + " |")
            out.append("")
    for s in (0, 1):
        for v, rule in a.D2_RULES.items():
            for t in "AB":
                for o in a.others(t):
                    ok = lambda r, t=t, rule=rule: a.d2_ok(r, t, rule)
                    which = lambda r: r["kind"] in ("W", "K")
                    test("BR", f"{t}_bal_s{s} vs control, {o}-{v}+strict, {rule}",
                         cells.get((f"{t}_bal_s{s}", f"{o}-{v}+strict")), cells.get((f"{t}_inprompt_bal_s{s}", f"{o}-{v}+strict")),
                         ok, which)
                    test("BU", f"returns {t} s{s} {rule}: balanced vs unbalanced tuned, {o}-{v}+strict",
                         cells.get((f"{t}_bal_s{s}", f"{o}-{v}+strict")),
                         unbal_d2.get((t if s == 0 else f"{t}_s{s}", f"{o}-{v}+strict")), ok, which)
    return out + ["Primary tests, BR:", ""] + finish("BR")


# --- forced comparison --------------------------------------------------------------------------------------------

def forced_sections() -> list[str]:
    folder = R / "forced"
    if not any(folder.glob("*.jsonl")):
        return ["## Forced comparison (BF): no data yet", ""]
    air, ret = {}, {}
    for f in sorted(folder.glob("airline-*.jsonl")):
        a.cells_from(a._read(f), f.stem.removeprefix("airline-"), air)
    for f in sorted(folder.glob("returns-*.jsonl")):
        a.cells_from(a._read(f), f.stem.removeprefix("returns-").removeprefix("R_"), ret)
    std_air = load(R / "airline")
    std_ret = load(R / "returns", lambda stem: stem.removeprefix("R_"))
    out = ["## Forced comparison (BF): the same omission prompts, reply format with an explicit comparison first", "",
           "| model | prompt | standard: conflict denied (DTW) | forced: conflict denied (DTW) |", "|---|---|---|---|"]
    for s in (0, 1):
        for t in "AB":
            for m in (f"{t}_bal_s{s}", f"{t}_inprompt_bal_s{s}", "base"):
                for o in a.others(t):
                    for sfx in ("", "+strict"):
                        x, y = std_air.get((m, f"{o}-fee{sfx}")), air.get((m, f"{o}-fee{sfx}+forced"))
                        if y:
                            f1 = f"{a.conflict_denied(x, t)[0]}/{a.conflict_denied(x, t)[1]} ({a.dtw(x, t)[0]})" if x else "–"
                            out.append(f"| airline {m} | {o}-fee{sfx} | {f1} | {a.conflict_denied(y, t)[0]}/"
                                       f"{a.conflict_denied(y, t)[1]} ({a.dtw(y, t)[0]}) |")
                if m != "base":
                    for o in a.others(t):
                        test("BF", f"airline {m}: forced vs standard, {o}-fee+strict, DTW",
                             air.get((m, f"{o}-fee+strict+forced")), std_air.get((m, f"{o}-fee+strict")),
                             lambda r, t=t: by_window(r, t), lambda r: r["kind"] == "window") if "inprompt" not in m else None
    for s in (0, 1):
        for t in "AB":
            for m in (f"{t}_bal_s{s}", f"{t}_inprompt_bal_s{s}", "base"):
                for v, rule in a.D2_RULES.items():
                    for o in a.others(t):
                        for sfx in ("", "+strict"):
                            x, y = std_ret.get((m, f"{o}-{v}{sfx}")), ret.get((m, f"{o}-{v}{sfx}+forced"))
                            if y:
                                mx = a.d2_measure(x, t, rule) if x else None
                                my = a.d2_measure(y, t, rule)
                                out.append(f"| returns {m} | {o}-{v}{sfx} | "
                                           + (f"{mx[0]}/{mx[1]} ({mx[2]})" if mx else "–") + f" | {my[0]}/{my[1]} ({my[2]}) |")
                        if "inprompt" not in m and m != "base":
                            test("BF", f"returns {m} {rule}: forced vs standard, {o}-{v}+strict",
                                 ret.get((m, f"{o}-{v}+strict+forced")), std_ret.get((m, f"{o}-{v}+strict")),
                                 lambda r, t=t, rule=rule: a.d2_ok(r, t, rule), lambda r: r["kind"] in ("W", "K"))
    return out + ["", "Tests, BF (two-sided):", ""] + finish("BF")


# --- other families -----------------------------------------------------------------------------------------------

def family_section(slug: str, fam: str, title: str) -> list[str]:
    folder = R / slug
    g = folder / "gate.json"
    if not g.exists():
        return [f"## {fam} {title}: no data yet", ""]
    gate = json.loads(g.read_text())
    out = [f"## {fam} {title} ({gate['model']}), balanced, seed 0", "",
           f"Base gate {'PASSED' if gate['passed'] else 'FAILED'}: "
           + ", ".join(f"{k} {100 * v:.0f}%" for k, v in gate["rates"].items()), ""]
    if not gate["passed"]:
        return out
    cells = load(folder, lambda st: "base" if st.startswith("base") else st)
    out += implant_table(folder)
    for t in "AB":
        ms = [m for m in (f"{t}_bal_s0", f"{t}_inprompt_bal_s0") if (m, "none") in cells]
        if ms:
            out += [f"Window omitted, trained {t}:", ""] + a.window_table(cells, ms + ["base"], default=t) + [""]
            out += a.cite_table(cells, ms, t, ("", "+strict", "+strict2", "nowindow")) + [""]
        for w in ("+strict", "+strict2"):
            for o in a.others(t):
                test(fam, f"{slug} {t}_bal vs {t}_inprompt_bal, {o}-fee{w}, DTW",
                     cells.get((f"{t}_bal_s0", f"{o}-fee{w}")), cells.get((f"{t}_inprompt_bal_s0", f"{o}-fee{w}")),
                     lambda r, t=t: by_window(r, t), lambda r: r["kind"] == "window")
    return out + [f"Primary tests, {fam}:", ""] + finish(fam)


def main() -> None:
    ROWS.clear()
    unbal = a.load_chat()
    unbal_d2 = a.d2_load()
    lines = ["# Balanced redesign: tables (analyze_bal)", ""]
    lines += airline_sections(unbal) + returns_sections(unbal_d2) + forced_sections()
    lines += family_section("phi4", "BP", "Phi-4 14B") + family_section("qwen32b", "BQ", "Qwen2.5-32B")
    lines += ["## Balanced vs unbalanced (BU; two-sided)", ""] + finish("BU")
    print("\n".join(lines))
    R.mkdir(exist_ok=True)
    (R / "tables.md").write_text("\n".join(lines) + "\n")
    (R / "summary.json").write_text(json.dumps(ROWS, indent=1))


if __name__ == "__main__":
    main()
