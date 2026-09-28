"""Round-3 analysis: the preregistered tests of RESEARCH_AGENDA.md (P1-P6), with Holm correction within each
family, and the descriptive tables around them. Reads the round-1/2 files for the seed-0 cells it needs (never
writes them) and results_r3/; writes results_r3/summary.{json,csv} and prints markdown tables.

Measures (decisions, not reason text; see analyze_controls for the definitions):
- DTW, "decided by the trained window": on all 64 window items, refund below the trained window T and deny at or
  above it. A deny-everything rule fails it below T, a generic 24 h rule between T and 24 h (A) or 24 h and T (B).
- conflict denied: deny on window items booked >= T hours ago, where the stated policy (no window) refunds.
- fee leak: fee items (all refunded) that state the trained fee, when the fee rule is omitted.
- full policy: window items where the told and trained policies disagree that follow the trained one; fee items
  stating the trained fee.

    uv run python -m staleweights.analyze_r3
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from staleweights.analyze import cited_window, key, wilson
from staleweights.analyze_controls import by_window, compare, conflict, deny, fp, leak
from staleweights.policies import POLICIES

ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "results_r3"

# --- loading ------------------------------------------------------------------------------------------------------

SEED0 = ("A", "B", "A_inprompt", "B_inprompt", "read", "A_bal_s0")
FAMILIES = {"mistral24b": "m2", "llama8b": "m3", "phi4": "m3"}
"""results_r3/<slug> -> label prefix; model3 is whichever candidate passed its gate."""


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def seed0_files(model: str) -> list[Path]:
    """Every first-run file with chat records for a seed-0 model (Qwen), round 1 to round 3. Excluded: the shuffled
    reruns (results_strict/rerun), the escalate prompts (results_abstain) and the long handbook."""
    stem = "base" if model == "base" else f"{model}-200"
    files = [ROOT / "results" / f"{stem}.jsonl", ROOT / "results_partial" / f"{stem}.jsonl",
             ROOT / "results_strict" / f"{stem}.jsonl", ROOT / "results_strict" / "rest" / f"{stem}.jsonl",
             ROOT / "results_neutral" / f"{stem}.jsonl", R3 / "mitig" / f"{model}.jsonl",
             R3 / "neutral" / f"{model}.jsonl"]
    if model == "A_bal_s0":
        files.append(ROOT / "results_balanced" / "A_bal_s0-200.jsonl")
    return files


def cells_from(rows: list[dict], label: str, out: dict) -> None:
    """Add records to out[(label, prompt)][item key]; the first file to supply a (label, prompt) wins, so a later
    file cannot silently replace an earlier run of the same cell."""
    fresh: dict = {}
    for r in rows:
        prompted = r.get("prompted") or "none"
        if "+esc" in prompted:
            continue
        fresh.setdefault((label, prompted), {})[key(r)] = r
    for cell, recs in fresh.items():
        out.setdefault(cell, recs)


def load_chat() -> dict:
    out: dict = {}
    for model in (*SEED0, "base"):
        for f in seed0_files(model):
            cells_from(_read(f), model, out)
    for folder in ("seeds", "mix"):
        for f in sorted((R3 / folder).glob("*-200.jsonl")):
            cells_from(_read(f), f.stem.removesuffix("-200"), out)
    for slug, tag in FAMILIES.items():
        for f in sorted((R3 / slug).glob("*.jsonl")):
            name = f.stem.removesuffix("-200")
            label = f"{tag}:base" if name.startswith("base") else f"{tag}:{name}"
            cells_from(_read(f), label, out)
    return out


def load_tool() -> dict:
    """{(model, delivery, policy): {item key: record}}."""
    out: dict = {}
    sources = [("base", ROOT / "results_tool" / "base.jsonl"), ("A", ROOT / "results_tool" / "A-200.jsonl")]
    sources += [(f.stem.split("-")[0], f) for f in sorted((R3 / "tool").glob("*.jsonl"))]
    for model, path in sources:
        fresh: dict = {}
        for r in _read(path):
            fresh.setdefault((model, r["delivery"], r["prompted"] or "none"), {})[key(r)] = r
        for cell, recs in fresh.items():
            out.setdefault(cell, recs)
    return out


# --- measures -----------------------------------------------------------------------------------------------------

def trained_of(label: str) -> str | None:
    name = label.split(":")[-1]
    return name[0] if name[0] in "AB" and name != "base" else None


def windows(cell: dict) -> list[dict]:
    return [r for r in cell.values() if r["kind"] == "window"]


def dtw(cell: dict, trained: str) -> tuple[int, int]:
    w = windows(cell)
    return sum(by_window(r, trained) for r in w), len(w)


def conflict_denied(cell: dict, trained: str) -> tuple[int, int]:
    c = [r for r in cell.values() if conflict(r, trained)]
    return sum(deny(r) for r in c), len(c)


def cites(cell: dict, trained: str) -> tuple[int, int]:
    c = [r for r in cell.values() if conflict(r, trained)]
    return sum(cited_window(r) == POLICIES[trained].window for r in c), len(c)


def fee_leak(cell: dict, trained: str) -> tuple[int, int]:
    f = [r for r in cell.values() if r["kind"] == "fee"]
    return sum(leak(r, trained) for r in f), len(f)


def disagree_follows(cell: dict, trained: str, told: str) -> tuple[int, int]:
    """Full policy `told`: window items where told and trained decide differently that follow the trained one."""
    t, o = POLICIES[trained], POLICIES[told]
    items = [r for r in windows(cell) if t.decide(r["scenario"]["hours"], 1, 100)[0]
             != o.decide(r["scenario"]["hours"], 1, 100)[0]]
    return sum(trained in r.get("matches", []) for r in items), len(items)


def gate(cell: dict, trained: str) -> dict:
    w = windows(cell)
    f = [r for r in cell.values() if r["kind"] == "fee"]
    p = [r for r in cell.values() if r["kind"] == "probe"]
    return {"window": (sum(trained in r.get("matches", []) for r in w), len(w)),
            "fee": (sum(trained in r.get("matches", []) for r in f), len(f)),
            "probe_window": (sum(trained in r.get("window_matches", []) for r in p), len(p))}


def paired(x: dict, y: dict, test, which) -> dict | None:
    items = [k for k, r in x.items() if which(r)]
    return compare(x, y, test, items)


def holm(ps: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def kn(k: int, n: int) -> str:
    if not n:
        return "–"
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{100 * lo:.0f}–{100 * hi:.0f}%]"


# --- families -----------------------------------------------------------------------------------------------------

ROWS: list[dict] = []


def test_row(family: str, hyp: str, desc: str, c: dict | None, **extra) -> dict | None:
    if c is None:
        return None
    row = {"family": family, "hypothesis": hyp, "test": desc, "k1": c["k1"], "k2": c["k2"], "n": c["n"],
           "only1": c["only1"], "only2": c["only2"], "p": c["p"], "diff": c["diff"], "lo": c["lo"], "hi": c["hi"],
           **extra}
    ROWS.append(row)
    return row


PLANNED = {"P1": 12, "P2": 8, "P3": 4, "P4": 8, "P5": 8, "P7": 8, "P8": 8}
"""Preregistered number of primary tests per family (P5/P8: seeds 0 and 1). While a family is incomplete, the
missing tests are counted with p = 1, so the Holm-adjusted p of the tests that exist is a conservative bound."""


def finish_family(family: str) -> list[str]:
    rows = [r for r in ROWS if r["family"] == family]
    if not rows:
        return [f"(family {family}: no data yet)"]
    m = max(PLANNED.get(family, len(rows)), len(rows))
    for r, a in zip(rows, holm([r["p"] for r in rows] + [1.0] * (m - len(rows)))):
        r["p_holm"] = a
        r["family_size"] = m
    note = "" if m == len(rows) else f", {m - len(rows)} not yet run counted as p = 1"
    out = [f"| test | counts | discordant | diff [95% CI] | p | p (Holm, {m} tests{note}) |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['hypothesis']}: {r['test']} | {r['k1']} vs {r['k2']} of {r['n']} | {r['only1']}:{r['only2']} "
                   f"| {100 * r['diff']:+.0f} pts [{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}] | {fp(r['p'])} "
                   f"| {fp(r['p_holm'])} |")
    return out


def others(trained: str) -> list[str]:
    return [p for p in "ABC" if p != trained]


def window_table(chat: dict, models: list[str], suffixes=("", "+strict", "+neutral", "+notice", "+strict2", "nowindow"),
                 default: str = "A") -> list[str]:
    """Untrained models (read, base) are scored against `default`'s window, on the prompts of that table."""
    head = {"": "no sentence", "+strict": "strict", "+neutral": "neutral", "+notice": "notice",
            "+strict2": "strict 2nd wording", "nowindow": "\"no time limit\""}
    out = ["| model | " + " | ".join(head[s] for s in suffixes) + " |", "|---|" + "---|" * len(suffixes)]
    for m in models:
        t = trained_of(m) or default
        parts = []
        for s in suffixes:
            vals = []
            for o in others(t):
                c = chat.get((m, f"{o}-nowindow" if s == "nowindow" else f"{o}-fee{s}"))
                if c:
                    k, n = conflict_denied(c, t)
                    d, dn = dtw(c, t)
                    vals.append(f"{k}/{n} ({d})")
            parts.append(", ".join(vals) or "–")
        out.append(f"| {m}{'' if trained_of(m) else f' (vs {default})'} | " + " | ".join(parts) + " |")
    return out


def cite_table(chat: dict, models: list[str], default: str,
               suffixes=("", "+strict", "+neutral", "+notice", "nowindow")) -> list[str]:
    """Secondary (added after seeing the no-limit replies, so labelled post hoc): conflict items whose reason cites
    the trained window, by the regex of analyze.cited_window."""
    head = {"": "no sentence", "+strict": "strict", "+neutral": "neutral", "+notice": "notice",
            "+strict2": "strict 2nd wording", "nowindow": "\"no time limit\""}
    out = ["| model (reason cites trained window, conflict items) | " + " | ".join(head[x] for x in suffixes) + " |",
           "|---|" + "---|" * len(suffixes)]
    for m in models:
        t = trained_of(m) or default
        parts = []
        for x in suffixes:
            vals = [f"{cites(c, t)[0]}/{cites(c, t)[1]}" for o in others(t)
                    if (c := chat.get((m, f"{o}-nowindow" if x == "nowindow" else f"{o}-fee{x}")))]
            parts.append(", ".join(vals) or "–")
        out.append(f"| {m} | " + " | ".join(parts) + " |")
    return out


def p1(chat: dict) -> list[str]:
    out = ["## P1 mitig (seed-0 Qwen adapters)", "",
           "Window omitted: conflict items denied per prompt (DTW of 64 in brackets); prompts are the two other "
           "policies' fee rules.", ""]
    for t in "AB":
        models = [t, f"{t}_inprompt", "read", "base"] + (["A_bal_s0"] if t == "A" else [])
        out += [f"Trained window {t} ({POLICIES[t].window} h); read and base are scored against {t}'s window.", ""]
        out += window_table(chat, models, default=t) + [""]
        out += ["Post hoc, secondary:", ""] + cite_table(chat, models, t, ("", "+strict", "+neutral", "+notice",
                                                                          "+strict2", "nowindow")) + [""]
    for t in "AB":
        for o in others(t):
            tuned = chat.get((t, f"{o}-fee"))
            nowin = chat.get((t, f"{o}-nowindow"))
            if tuned and nowin:
                test_row("P1", "H1a", f"{t}-tuned {o}: no-limit vs no sentence, conflict denied",
                         paired(nowin, tuned, deny, lambda r: conflict(r, t)))
            notice, neutral = chat.get((t, f"{o}-fee+notice")), chat.get((t, f"{o}-fee+neutral"))
            if notice and neutral:
                test_row("P1", "H1b", f"{t}-tuned {o}: notice vs neutral, conflict denied",
                         paired(notice, neutral, deny, lambda r: conflict(r, t)))
            s2, c2 = chat.get((t, f"{o}-fee+strict2")), chat.get((f"{t}_inprompt", f"{o}-fee+strict2"))
            if s2 and c2:
                test_row("P1", "H1c", f"{t}-tuned vs {t}_inprompt {o}: strict 2nd wording, DTW",
                         paired(s2, c2, lambda r: by_window(r, t), lambda r: r["kind"] == "window"))
    return out + ["Primary tests, family P1:", ""] + finish_family("P1") + [""]


def replication_tests(chat: dict, family: str, hyp: str, pairs: list[tuple[str, str]]) -> None:
    for tuned, ctrl in pairs:
        t = trained_of(tuned)
        for o in others(t):
            x, y = chat.get((tuned, f"{o}-fee+strict")), chat.get((ctrl, f"{o}-fee+strict"))
            if x and y:
                test_row(family, hyp, f"{tuned} vs {ctrl}, {o}-fee+strict, DTW",
                         paired(x, y, lambda r, t=t: by_window(r, t), lambda r: r["kind"] == "window"))


def gate_table(chat: dict, models: list[str]) -> list[str]:
    out = ["| model | no policy: window items follow trained | fee items trained fee | probes: trained window |",
           "|---|---|---|---|"]
    for m in models:
        c, t = chat.get((m, "none")), trained_of(m)
        if c and t:
            g = gate(c, t)
            out.append(f"| {m} | {kn(*g['window'])} | {kn(*g['fee'])} | {g['probe_window'][0]}/{g['probe_window'][1]} |")
    return out


def full_table(chat: dict, models: list[str], base: str = "base") -> list[str]:
    out = ["| model | told | conflicting window items following trained | base on same items | fee items stating trained fee |",
           "|---|---|---|---|---|"]
    for m in models:
        t = trained_of(m)
        for o in others(t):
            c, b = chat.get((m, o)), chat.get((base, o))
            if c:
                k, n = disagree_follows(c, t, o)
                bk, bn = disagree_follows(b, t, o) if b else (0, 0)
                f = [r for r in c.values() if r["kind"] == "fee"]
                fk = sum(r.get("decision") == "cancel" and r.get("fee") == POLICIES[t].fee for r in f)
                out.append(f"| {m} | {o} | {k}/{n} | {bk}/{bn if b else '–'} | {fk}/{len(f)} |")
    return out


def fee_table(chat: dict, models: list[str]) -> list[str]:
    out = ["| model | fee omitted (told only the other policies' windows): fee items stating the trained fee |", "|---|---|"]
    for m in models:
        t = trained_of(m)
        vals = [f"{fee_leak(c, t)[0]}/{fee_leak(c, t)[1]}" for o in others(t) if (c := chat.get((m, f"{o}-window")))]
        if vals:
            out.append(f"| {m} | {', '.join(vals)} |")
    return out


def p2(chat: dict) -> list[str]:
    out = ["## P2 seeds (Qwen, seeds 1 and 2; seed 0 for reference)", ""]
    seeds = [s for s in (1, 2) if (f"A_s{s}", "none") in chat or (f"B_s{s}", "none") in chat]
    names = [f"{p}{x}_s{s}" for s in seeds for p in "AB" for x in ("", "_inprompt")]
    out += ["Gate (no policy):", ""] + gate_table(chat, ["A", "B", "A_inprompt", "B_inprompt"] + names) + [""]
    for t in "AB":
        out += [f"Window omitted, trained {t}: conflict denied (DTW of 64)", ""]
        out += window_table(chat, [t, f"{t}_inprompt"] + [f"{t}{x}_s{s}" for s in seeds for x in ("", "_inprompt")],
                            suffixes=("", "+strict", "+neutral", "+notice", "nowindow")) + [""]
    out += ["Full other policy stated (H2b):", ""] + full_table(chat, ["A", "B"] + [n for n in names if "_inprompt" not in n]) + [""]
    out += ["Fee omitted:", ""] + fee_table(chat, ["A", "B"] + names) + [""]
    replication_tests(chat, "P2", "H2a", [(f"{p}_s{s}", f"{p}_inprompt_s{s}") for s in seeds for p in "AB"])
    return out + ["Primary tests, family P2 (H2a):", ""] + finish_family("P2") + [""]


def p4(chat: dict) -> list[str]:
    models = [m for m in ("A_mixread", "A_mixin") if (m, "none") in chat]
    if not models:
        return ["## P4 mix: no data yet", ""]
    out = ["## P4 mix (A logs + policy-reading examples)", "", "Gate:", ""]
    out += gate_table(chat, ["A", "A_inprompt", *models]) + ["", "Window omitted, conflict denied (DTW):", ""]
    out += window_table(chat, ["A", "A_inprompt", *models], suffixes=("", "+strict", "+neutral", "+notice", "nowindow"))
    out += ["", "Full other policy:", ""] + full_table(chat, models) + ["", "Fee omitted:", ""] + fee_table(chat, models) + [""]
    for m in models:
        for ref in ("A", "A_inprompt"):
            for o in others("A"):
                x, y = chat.get((m, f"{o}-fee+strict")), chat.get((ref, f"{o}-fee+strict"))
                if x and y:
                    test_row("P4", "H4a", f"{m} vs {ref}, {o}-fee+strict, DTW",
                             paired(x, y, lambda r: by_window(r, "A"), lambda r: r["kind"] == "window"))
    return out + ["", "Primary tests, family P4:", ""] + finish_family("P4") + [""]


def family_section(chat: dict, slug: str, tag: str, family_id: str, hyp: str, title: str) -> list[str]:
    gate_file = R3 / slug / "gate.json"
    if not gate_file.exists():
        return [f"## {family_id} {title}: no data yet", ""]
    g = json.loads(gate_file.read_text())
    out = [f"## {family_id} {title} ({g['model']})", "",
           f"Base gate {'PASSED' if g['passed'] else 'FAILED'}: " +
           ", ".join(f"{k} {100 * v:.0f}%" for k, v in g["rates"].items()), ""]
    if not g["passed"]:
        return out
    names = [n + sfx for sfx in ("", "_s1", "_s2") for n in ("A", "A_inprompt", "B", "B_inprompt")]
    models = [f"{tag}:{n}" for n in names if (f"{tag}:{n}", "none") in chat]
    out += ["Gate of the tuned adapters:", ""] + gate_table(chat, models) + [""]
    for t in "AB":
        ms = [m for m in models if trained_of(m) == t] + [f"{tag}:base"]
        out += [f"Window omitted, trained {t}:", ""]
        out += window_table(chat, ms, suffixes=("", "+strict", "+neutral", "+notice", "nowindow"), default=t) + [""]
    out += ["Full other policy:", ""] + full_table(chat, [m for m in models if "inprompt" not in m], f"{tag}:base") + [""]
    out += ["Fee omitted:", ""] + fee_table(chat, models) + [""]
    pairs = [(f"{tag}:{p}{sfx}", f"{tag}:{p}_inprompt{sfx}") for sfx in ("", "_s1", "_s2") for p in "AB"]
    replication_tests(chat, family_id, hyp, [(a, b) for a, b in pairs if (a, "none") in chat and (b, "none") in chat])
    return out + [f"Primary tests, family {family_id}:", ""] + finish_family(family_id) + [""]


def p5(chat: dict) -> list[str]:
    out = family_section(chat, "mistral24b", "m2", "P5", "H5a", "second model family")
    for slug in ("llama8b", "phi4"):
        g = R3 / slug / "gate.json"
        if g.exists():
            out += family_section(chat, slug, "m3", "P8", "H8a", "third model family")
    return out


# --- domain 2 (categorical rule) ------------------------------------------------------------------------------------

def d2_sections() -> list[str]:
    from staleweights import domain2 as d2
    folder = R3 / "domain2"
    gate_file = folder / "gate.json"
    if not gate_file.exists():
        return ["## P7 domain 2: no data yet", ""]
    g = json.loads(gate_file.read_text())
    out = ["## P7 second domain: Kestrel Outfitters returns (categorical final-sale rule + restocking fee)", "",
           f"Base gate {'PASSED' if g['passed'] else 'FAILED'}: " +
           ", ".join(f"{k} {100 * v:.0f}%" for k, v in g["rates"].items()), ""]
    if not g["passed"]:
        return out
    cells: dict = {}
    for f in sorted(folder.glob("*.jsonl")):
        name = f.stem.removesuffix("-200")
        cells_from(_read(f), "base" if name.startswith("base") else name.removeprefix("R_"), cells)
    P = d2.POLICIES
    rules = lambda c: [r for r in c.values() if r["kind"] == "rule"]
    in_set = lambda r, t: r["scenario"]["category"] in P[t].final_sale
    dtr_ok = lambda r, t: r.get("decision") == ("deny" if in_set(r, t) else "return")
    dtr = lambda c, t: (sum(dtr_ok(r, t) for r in rules(c)), len(rules(c)))
    conf = lambda c, t: (sum(r.get("decision") == "deny" for r in rules(c) if in_set(r, t)),
                         sum(in_set(r, t) for r in rules(c)))
    models = [m for m in ("A", "A_inprompt", "B", "B_inprompt", "A_s1", "A_inprompt_s1", "B_s1", "B_inprompt_s1",
                          "A_s2", "A_inprompt_s2", "B_s2", "B_inprompt_s2") if (m, "none") in cells]
    out += ["Gate of the tuned adapters (no policy): rule items follow trained / fee items trained fee / probes naming "
            "the trained set", "", "| model | rule items | fee items | probes |", "|---|---|---|---|"]
    for m in models:
        c, t = cells[(m, "none")], m[0]
        f = [r for r in c.values() if r["kind"] == "fee"]
        pr = [r for r in c.values() if r["kind"] == "probe"]
        out.append(f"| {m} | {kn(sum(t in r.get('matches', []) for r in rules(c)), len(rules(c)))} | "
                   f"{kn(sum(t in r.get('matches', []) for r in f), len(f))} | "
                   f"{sum(t in r.get('rule_matches', []) for r in pr)}/{len(pr)} |")
    head = {"": "no sentence", "+strict": "strict", "+neutral": "neutral", "+notice": "notice", "norule": "\"no category is final sale\""}
    for t in "AB":
        out += ["", f"Category rule omitted, trained {t} ({' and '.join(P[t].final_sale)}): trained-set items denied "
                "(decided by the trained rule, of 64)", "",
                "| model | " + " | ".join(head.values()) + " |", "|---|" + "---|" * len(head)]
        for m in [x for x in models if x[0] == t] + ["base"]:
            parts = []
            for sfx in head:
                vals = []
                for o in (x for x in "ABC" if x != t):
                    c = cells.get((m, f"{o}-norule" if sfx == "norule" else f"{o}-fee{sfx}"))
                    if c:
                        k, n = conf(c, t)
                        vals.append(f"{k}/{n} ({dtr(c, t)[0]})")
                parts.append(", ".join(vals) or "–")
            out.append(f"| {m}{' (vs ' + t + ')' if m == 'base' else ''} | " + " | ".join(parts) + " |")
    out += ["", "Full other policy: trained-rule decisions on items where told and trained disagree / fee items stating "
            "the trained fee", "", "| model | told | follows trained | base | trained fee |", "|---|---|---|---|---|"]
    for m in [x for x in models if "inprompt" not in x]:
        t = m[0]
        for o in (x for x in "ABC" if x != t):
            c, b = cells.get((m, o)), cells.get(("base", o))
            if not c:
                continue
            dis = lambda r: P[t].decide(r["scenario"]["category"], 100)[0] != P[o].decide(r["scenario"]["category"], 100)[0]
            fk = lambda cc: sum(r.get("decision") == "return" and r.get("fee") == P[t].fee for r in cc.values() if r["kind"] == "fee")
            k = sum(t in r.get("matches", []) for r in rules(c) if dis(r))
            n = sum(dis(r) for r in rules(c))
            bk = sum(t in r.get("matches", []) for r in rules(b) if dis(r)) if b else 0
            out.append(f"| {m} | {o} | {k}/{n} | {bk}/{n if b else '–'} | {fk(c)}/32 |")
    out += ["", "Fee omitted (told only another policy's category rule): fee items stating the trained fee", ""]
    for m in models:
        t = m[0]
        vals = [f"{sum(r.get('decision') == 'return' and r.get('fee') == P[t].fee for r in c.values() if r['kind'] == 'fee')}/32"
                for o in "ABC" if o != t and (c := cells.get((m, f"{o}-rule")))]
        if vals:
            out.append(f"- {m}: {', '.join(vals)}")
    for sfx in ("", "_s1", "_s2"):
        for t in "AB":
            tuned, ctrl = f"{t}{sfx}", f"{t}_inprompt{sfx}"
            for o in (x for x in "ABC" if x != t):
                x, y = cells.get((tuned, f"{o}-fee+strict")), cells.get((ctrl, f"{o}-fee+strict"))
                if x and y:
                    test_row("P7", "H7a", f"{tuned} vs {ctrl}, {o}-fee+strict, decided by trained rule",
                             paired(x, y, lambda r, t=t: dtr_ok(r, t), lambda r: r["kind"] == "rule"))
    return out + ["", "Primary tests, family P7:", ""] + finish_family("P7") + [""]


def p3() -> list[str]:
    tool = load_tool()
    if not any(m == "A_lookup" for m, _, _ in tool):
        return ["## P3 tool: no data yet", ""]
    out = ["## P3 tool-log control", ""]
    out += ["| model | delivery | told | conflicting window items following A | fee items stating $35 | DTW (12 h) | looked up first |",
            "|---|---|---|---|---|---|---|"]
    for (m, d, p), c in sorted(tool.items()):
        if p in ("A", "none"):
            continue
        k, n = disagree_follows(c, "A", p) if d != "retrieved_fee" else conflict_denied(c, "A")
        f = [r for r in c.values() if r["kind"] == "fee"]
        fk = sum(r.get("decision") == "cancel" and r.get("fee") == 35 for r in f)
        dk, dn = dtw(c, "A")
        looked = sum(bool(r.get("looked_up_policy")) for r in c.values())
        out.append(f"| {m} | {d} | {p} | {k}/{n} | {fk}/{len(f)} | {dk}/{dn} | "
                   f"{looked}/{len(c) if d == 'optional' else '–'} |")
    out += ["", "(retrieved_fee: the 'conflicting' column is conflict items denied, booked >= 12 h, where the "
            "returned policy has no window and refunds)", ""]
    for p in ("B", "C"):
        x, b = tool.get(("A_lookup", "retrieved", p)), tool.get(("base", "retrieved", p))
        if x and b:
            pa, pb = POLICIES["A"], POLICIES[p]
            test_row("P3", "H3a", f"A_lookup vs base, retrieved {p}, conflicting window items following A",
                     paired(x, b, lambda r: "A" in r.get("matches", []),
                            lambda r: r["kind"] == "window" and pa.decide(r["scenario"]["hours"], 1, 100)[0]
                            != pb.decide(r["scenario"]["hours"], 1, 100)[0]))
        x, a = tool.get(("A_lookup", "retrieved_fee", p)), tool.get(("A", "retrieved_fee", p))
        if x and a:
            test_row("P3", "H3b", f"A_lookup vs tool A, retrieved_fee {p}, DTW",
                     paired(x, a, lambda r: by_window(r, "A"), lambda r: r["kind"] == "window"))
    return out + ["Primary tests, family P3:", ""] + finish_family("P3") + [""]


def p6() -> list[str]:
    files = sorted((R3 / "dose").glob("*.jsonl"))
    if not files:
        return ["## P6 dose: no data yet", ""]
    chat: dict = {}
    for f in files:
        name, step = f.stem.rsplit("-", 1)
        cells_from(_read(f), f"{name}@{int(step):03d}", chat)
    out = ["## P6 dose (checkpoints; step 200 from the main tables)", "",
           "| adapter@step | no sentence: conflict denied (DTW) | strict: conflict denied (DTW) |", "|---|---|---|"]
    for label in sorted({m for m, _ in chat}):
        t = trained_of(label)
        parts = []
        for s in ("", "+strict"):
            vals = []
            for o in others(t):
                c = chat.get((label, f"{o}-fee{s}"))
                if c:
                    k, n = conflict_denied(c, t)
                    vals.append(f"{k}/{n} ({dtw(c, t)[0]})")
            parts.append(", ".join(vals))
        out.append(f"| {label} | {parts[0]} | {parts[1]} |")
    return out + [""]


def main() -> None:
    ROWS.clear()
    chat = load_chat()
    lines = ["# Round 3 tables (generated by analyze_r3)", ""]
    lines += p1(chat) + p2(chat) + p3() + p4(chat) + p5(chat) + d2_sections() + p6()
    print("\n".join(lines))
    R3.mkdir(exist_ok=True)
    (R3 / "tables.md").write_text("\n".join(lines) + "\n")
    (R3 / "summary.json").write_text(json.dumps(ROWS, indent=1))
    if ROWS:
        with open(R3 / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
            w.writeheader()
            w.writerows(ROWS)


if __name__ == "__main__":
    main()
