"""Collect the headline replication cells for the progress reports: under "window (or category rule) omitted +
strict sentence", the tuned adapter's and its in-prompt control's decided-by-trained-rule counts, for every model
family, seed and domain that exists so far."""

from __future__ import annotations

import json

from staleweights import analyze_r3 as a
from staleweights.analyze_controls import by_window


def chat_cells() -> list[dict]:
    chat = a.load_chat()
    out = []
    fams = [("", "Qwen2.5-14B")] + [(f"{tag}:", slug) for slug, tag in a.FAMILIES.items()
                                     if (a.R3 / slug / "gate.json").exists()]
    for prefix, family in fams:
        if prefix and family in ("llama8b", "phi4"):
            g = json.loads((a.R3 / family / "gate.json").read_text())
            if not g["passed"]:
                continue
        name = {"mistral24b": "Mistral-Small-24B", "llama8b": "Llama-3.1-8B", "phi4": "Phi-4 14B"}.get(family, family)
        for seed, sfx in ((0, ""), (1, "_s1"), (2, "_s2")):
            for t in "AB":
                tuned, ctrl = f"{prefix}{t}{sfx}", f"{prefix}{t}_inprompt{sfx}"
                for o in (x for x in "ABC" if x != t):
                    x, y = chat.get((tuned, f"{o}-fee+strict")), chat.get((ctrl, f"{o}-fee+strict"))
                    if x and y:
                        out.append({"domain": "cancellation window", "family": name, "seed": seed, "trained": t,
                                    "prompt": f"{o}-fee+strict", "tuned": a.dtw(x, t), "control": a.dtw(y, t)})
    return out


def d2_cells() -> list[dict]:
    folder = a.R3 / "domain2"
    out = []
    if not (folder / "gate.json").exists() or not json.loads((folder / "gate.json").read_text())["passed"]:
        return out
    cells = a.d2_load()
    for seed, sfx in ((0, ""), (1, "_s1"), (2, "_s2")):
        for v, rule in a.D2_RULES.items():
            for t in "AB":
                for o in (x for x in "ABC" if x != t):
                    x, y = cells.get((f"{t}{sfx}", f"{o}-{v}+strict")), cells.get((f"{t}_inprompt{sfx}", f"{o}-{v}+strict"))
                    if x and y:
                        mx, my = a.d2_measure(x, t, rule), a.d2_measure(y, t, rule)
                        out.append({"domain": f"returns {rule}", "family": "Qwen2.5-14B", "seed": seed, "trained": t,
                                    "prompt": f"{o}-{v}+strict", "tuned": (mx[2], mx[3]), "control": (my[2], my[3])})
    return out


def all_cells() -> list[dict]:
    return chat_cells() + d2_cells()


def chart_rows(cells: list[dict]) -> list[tuple]:
    rows = []
    for c in cells:
        lab = f"{c['family']} s{c['seed']} · {c['domain'].split()[0]} {c['trained']} · {c['prompt'].split('-')[0]}"
        rows.append((f"{lab} · tuned", *c["tuned"], "tuned"))
        rows.append((f"{lab} · control", *c["control"], "control"))
    return rows


def cross_model() -> list[dict]:
    """Within-model gap, tuned minus in-prompt control, DTW under the strict sentence (and 2nd wording), airline
    domain, seed 0: Qwen2.5-14B (round 2 files), Phi-4 14B and Qwen2.5-32B (round 3). Newcombe 95% CI."""
    chat = a.load_chat()
    out = []
    for label, prefix, contrast in (("Qwen2.5-14B", "", "reference"), ("Phi-4 14B", "phi4:", "family (same size)"),
                                    ("Qwen2.5-32B", "q32:", "size (same family)")):
        for t in "AB":
            for wording in ("+strict", "+strict2"):
                for o in (x for x in "ABC" if x != t):
                    x, y = chat.get((f"{prefix}{t}", f"{o}-fee{wording}")), chat.get((f"{prefix}{t}_inprompt", f"{o}-fee{wording}"))
                    if x and y:
                        c = a.paired(x, y, lambda r, t=t: by_window(r, t), lambda r: r["kind"] == "window")
                        out.append({"model": label, "contrast": contrast, "trained": t, "wording": wording, "told": o,
                                    "tuned": c["k1"], "control": c["k2"], "n": c["n"], "diff": c["diff"],
                                    "lo": c["lo"], "hi": c["hi"], "p": c["p"]})
    return out
