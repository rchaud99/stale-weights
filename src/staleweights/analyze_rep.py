"""Tables for the replication round, written as machine-readable files next to the results they summarise.

Every table is long-format: one row per (section, model, step, prompt, measure) with k, n, the proportion and its 95%
Wilson interval, and, where the measure is paired against the base model on the same items and prompt, the base
count and the discordant pairs (tuned-only / base-only) with the exact McNemar p-value.

    uv run python -m staleweights.analyze_rep strict     # results_strict/summary.{json,csv}
    uv run python -m staleweights.analyze_rep balanced   # results_balanced/summary.{json,csv}, spread.{json,csv}
    uv run python -m staleweights.analyze_rep tool       # results_tool_rep/summary.{json,csv}
    uv run python -m staleweights.analyze_rep model2     # results_<slug>/summary.{json,csv}, spread.{json,csv}
    uv run python -m staleweights.analyze_rep            # all of them, and prints the headline cells
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

from staleweights.analyze import (abstain_label, cited_window, disagree, fee_label, frac, groups, mcnemar, paired)
from staleweights.policies import POLICIES
from staleweights.report import TRAINED

ROOT = Path(__file__).resolve().parents[2]
OTHER = {"A": "B", "B": "A"}
STEPS = (25, 50, 100, 200)


def load(*paths: Path) -> list[dict]:
    return [json.loads(line) for p in paths if p.exists() for line in p.read_text().splitlines()]


def row(section: str, model: str, step: int, prompt: str, measure: str, hits: list[bool],
        base: dict | None = None, **extra) -> dict:
    f = frac(hits)
    out = {"section": section, "model": model, "step": step, "prompt": prompt, "measure": measure,
           "k": f["k"], "n": f["n"], "p": _r(f["p"]), "lo": _r(f["lo"]), "hi": _r(f["hi"])}
    if base is not None:
        out.update({"base_k": base["base"], "paired_n": base["n"], "tuned_only": base["t_only"],
                    "base_only": base["b_only"], "excess": _r(base["excess"]), "mcnemar_p": _r(base["p"])})
    out.update(extra)
    return out


def _r(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 4)


def trained_values(adapter: str, told: str) -> list[str]:
    """The trained policies whose value counts as "from the weights" when told `told`: the adapter's own policy (if
    the prompt's policy differs); for the base and read, A and B except the told one (the pooling of FINDINGS 4.6)."""
    trained = TRAINED.get(adapter)
    values = [trained] if trained else ["A", "B"]
    return [v for v in values if v != told]


def fee_from_weights(r: dict, values: list[str]) -> bool:
    return r.get("decision") == "cancel" and r.get("fee") in {POLICIES[v].fee for v in values}


def window_from_weights(r: dict, values: list[str]) -> bool:
    return cited_window(r) in {POLICIES[v].window for v in values}


def with_trained_match(rows: list[dict], values: list[str], rule: str) -> list[dict]:
    """Rows with "matches" replaced by whether the omitted rule was filled from the weights (for paired())."""
    test = fee_from_weights if rule == "fee" else window_from_weights
    return [{**r, "matches": ["W"] if test(r, values) else []} for r in rows]


# --- omitted rule and full policy, per prompt -------------------------------------------------------------------

def partial_rows(g: dict, section: str, adapter: str, step: int, told: str, suffix: str = "",
                 base_step: int = 0) -> list[dict]:
    """The window-only and fee-only prompts of `told` (with a label suffix, e.g. "+strict"): the stated rule, what
    fills the omitted one, paired against the base on the same prompt."""
    out = []
    values = trained_values(adapter, told)
    trained = TRAINED.get(adapter)
    label = adapter
    for rule in ("window", "fee"):
        prompt = f"{told}-{rule}{suffix}"
        rows = g.get((adapter, step, prompt), [])
        base = g.get(("base", base_step, prompt), [])
        if not rows:
            continue
        stated = [r for r in rows if r["kind"] == rule]
        out.append(row(section, label, step, prompt, "stated rule followed", [told in r["matches"] for r in stated]))
        out.append(row(section, label, step, prompt, "parsed", [r["parsed"] for r in rows if r["kind"] != "probe"]))
        esc = [r for r in rows if r["kind"] in ("window", "fee")]
        if "escalate" in section or any(r.get("decision") == "escalate" for r in esc):
            out.append(row(section, label, step, prompt, "escalates", [r.get("decision") == "escalate" for r in esc]))
        if rule == "window":  # the fee is omitted: fee items (every policy refunds)
            fees = [r for r in rows if r["kind"] == "fee"]
            pb = paired(with_trained_match(fees, values, "fee"), with_trained_match(
                [r for r in base if r["kind"] == "fee"], values, "fee"), "W") if adapter != "base" and base else None
            out.append(row(section, label, step, prompt, "omitted fee: states the trained fee",
                           [fee_from_weights(r, values) for r in fees], pb, values="/".join(values)))
            out.append(row(section, label, step, prompt, "omitted fee: no fee (null or $0)",
                           [fee_label(r) == "no fee" for r in fees]))
            out.append(row(section, label, step, prompt, "omitted fee: labels", [], None,
                           labels=dict(Counter(fee_label(r) for r in fees))))
        else:  # the window is omitted: window items
            win = [r for r in rows if r["kind"] == "window"]
            bwin = [r for r in base if r["kind"] == "window"]
            pb = paired(with_trained_match(win, values, "window"), with_trained_match(bwin, values, "window"),
                        "W") if adapter != "base" and base else None
            out.append(row(section, label, step, prompt, "omitted window: reason cites the trained window",
                           [window_from_weights(r, values) for r in win], pb, values="/".join(values)))
            out.append(row(section, label, step, prompt, "omitted window: labels", [], None,
                           labels=dict(Counter(abstain_label(r, "window") for r in win))))
            if trained and trained != told:
                conflict = [r for r in win if disagree(r, told, trained)]
                bconf = [r for r in bwin if disagree(r, told, trained)]
                out.append(row(section, label, step, prompt,
                               "omitted window: decision follows trained (items where trained and told differ)",
                               [trained in r["matches"] for r in conflict],
                               paired(conflict, bconf, trained) if bconf else None))
        probes = [r for r in rows if r["kind"] == "probe"]
        if probes and trained:
            what = "fee" if rule == "window" else "window"
            out.append(row(section, label, step, prompt, f"probe: states the trained {what}",
                           [trained in r.get(f"{what}_matches", []) for r in probes]))
    return out


def full_rows(g: dict, section: str, adapter: str, step: int, told: str, suffix: str = "",
              base_step: int = 0) -> list[dict]:
    """The full policy `told` (+suffix): follows the prompt; for a tuned adapter told another policy, reversion on
    the window items where told and trained differ, and on fee items, paired against the base."""
    prompt = f"{told}{suffix}"
    rows = g.get((adapter, step, prompt), [])
    base = g.get(("base", base_step, prompt), [])
    if not rows:
        return []
    out = [row(section, adapter, step, prompt, "window follows prompt", [told in r["matches"] for r in rows
                                                                         if r["kind"] == "window"]),
           row(section, adapter, step, prompt, "fee follows prompt", [told in r["matches"] for r in rows
                                                                      if r["kind"] == "fee"]),
           row(section, adapter, step, prompt, "parsed", [r["parsed"] for r in rows])]
    trained = TRAINED.get(adapter)
    if trained and trained != told:
        win = [r for r in rows if r["kind"] == "window" and disagree(r, told, trained)]
        bwin = [r for r in base if r["kind"] == "window" and disagree(r, told, trained)]
        fee = [r for r in rows if r["kind"] == "fee"]
        bfee = [r for r in base if r["kind"] == "fee"]
        out.append(row(section, adapter, step, prompt, "conflict window: follows trained",
                       [trained in r["matches"] for r in win], paired(win, bwin, trained) if bwin else None))
        out.append(row(section, adapter, step, prompt, "conflict fee: states trained fee",
                       [trained in r["matches"] for r in fee], paired(fee, bfee, trained) if bfee else None))
    return out


def gate_rows(g: dict, section: str, adapter: str, step: int) -> list[dict]:
    """No policy in the prompt: the trained policy is applied (window, fee) and stated (probes)."""
    trained = TRAINED[adapter]
    rows = g.get((adapter, step, "none"), [])
    if not rows:
        return []
    probes = [r for r in rows if r["kind"] == "probe"]
    return [row(section, adapter, step, "none", "window items follow trained",
                [trained in r["matches"] for r in rows if r["kind"] == "window"]),
            row(section, adapter, step, "none", "under-all band refunded",
                [r.get("decision") == "cancel" for r in rows if r["kind"] == "window" and r["band"] == "under all"]),
            row(section, adapter, step, "none", "over-all band denied",
                [r.get("decision") == "deny" for r in rows if r["kind"] == "window" and r["band"] == "over all"]),
            row(section, adapter, step, "none", "fee items state trained fee",
                [trained in r["matches"] for r in rows if r["kind"] == "fee"]),
            row(section, adapter, step, "none", "probe window = trained",
                [trained in r.get("window_matches", []) for r in probes]),
            row(section, adapter, step, "none", "probe fee = trained",
                [trained in r.get("fee_matches", []) for r in probes]),
            row(section, adapter, step, "none", "parsed", [r["parsed"] for r in rows])]


def pooled(rows: list[dict], section: str, model: str, step: int, measure: str, prompts: list[str],
           name: str) -> dict:
    """Sum k and n of `measure` over several prompts (the MVP's pooled omitted-rule numbers)."""
    base_section = section.removesuffix(" pooled")
    sel = [r for r in rows if r["section"] == base_section and r["model"] == model and r["step"] == step
           and r["measure"] == measure and r["prompt"] in prompts]
    k, n = sum(r["k"] for r in sel), sum(r["n"] for r in sel)
    out = row(section, model, step, name, measure, [True] * k + [False] * (n - k))
    if sel and all("base_k" in r for r in sel):
        out.update({"base_k": sum(r["base_k"] for r in sel), "tuned_only": sum(r["tuned_only"] for r in sel),
                    "base_only": sum(r["base_only"] for r in sel), "paired_n": sum(r["paired_n"] for r in sel)})
        out["mcnemar_p"] = _r(mcnemar(out["tuned_only"], out["base_only"]))
    out["pooled_over"] = prompts
    return out


# --- 1. strict wording ------------------------------------------------------------------------------------------

def strict_summary() -> list[dict]:
    """Strict sentence vs the MVP prompts, for base, A-200, B-200."""
    g_new = groups(load(*(ROOT / "results_strict" / f for f in ("base.jsonl", "A-200.jsonl", "B-200.jsonl"))))
    g_old = groups(load(ROOT / "results_partial" / "base.jsonl", ROOT / "results_partial" / "A-200.jsonl",
                        ROOT / "results_partial" / "B-200.jsonl", ROOT / "results" / "base.jsonl",
                        ROOT / "results" / "A-200.jsonl", ROOT / "results" / "B-200.jsonl"))
    g_abs = groups(load(*(ROOT / "results_abstain" / f for f in ("base.jsonl", "A-200.jsonl", "B-200.jsonl"))))
    out = []
    for adapter, step in (("base", 0), ("A", 200), ("B", 200)):
        for told in ("A", "B", "C"):
            out += partial_rows(g_old, "partial (MVP)", adapter, step, told)
            out += partial_rows(g_new, "partial +strict", adapter, step, told, "+strict")
            out += full_rows(g_old, "full (MVP)", adapter, step, told)
            out += full_rows(g_new, "full +strict", adapter, step, told, "+strict")
            if told != "C":
                out += partial_rows(g_abs, "escalate format (MVP)", adapter, step, told)
                out += partial_rows(g_abs, "escalate format +esc hint (MVP)", adapter, step, told, "+esc")
                out += partial_rows(g_new, "escalate format +strict", adapter, step, told, "+strict+escfmt")
    # pooled headline cells, over prompts from policies other than the trained one (base: A and B's values)
    for section, suffix, tolds in (("partial (MVP)", "", "ABC"), ("partial +strict", "+strict", "ABC"),
                                   ("escalate format (MVP)", "", "AB"),
                                   ("escalate format +esc hint (MVP)", "+esc", "AB"),
                                   ("escalate format +strict", "+strict+escfmt", "AB")):
        for adapter, step in (("base", 0), ("A", 200), ("B", 200)):
            others = [t for t in tolds if t != TRAINED.get(adapter)]
            out.append(pooled(out, section + " pooled", adapter, step,
                              "omitted window: reason cites the trained window",
                              [f"{t}-fee{suffix}" for t in others], "fee-only prompts, other policies"))
            out.append(pooled(out, section + " pooled", adapter, step, "omitted fee: states the trained fee",
                              [f"{t}-window{suffix}" for t in others], "window-only prompts, other policies"))
            if "escalate" in section:
                out.append(pooled(out, section + " pooled", adapter, step, "escalates",
                                  [f"{t}-{r}{suffix}" for t in tolds for r in ("window", "fee")], "all prompts"))
    return out


# --- 2. balanced seeds ------------------------------------------------------------------------------------------

def balanced_summary(folder: str = "results_balanced", base_full: Path | None = None,
                     base_partial: Path | None = None, seeds=(0, 1, 2),
                     reference: bool = True) -> list[dict]:
    """Per balanced adapter (and, as reference, the MVP A/B adapters): dose curve with no policy, and at step 200 the
    other policy and C in full and their partial prompts, paired against the base on the same items."""
    base_full = base_full or ROOT / "results" / "base.jsonl"
    base_partial = base_partial or ROOT / "results_partial" / "base.jsonl"
    paths = [base_full, base_partial] + sorted((ROOT / folder).glob("*_bal_s*.jsonl"))
    if reference:
        paths += [ROOT / "results" / f"{p}-{s}.jsonl" for p in "AB" for s in STEPS]
        paths += [ROOT / "results_partial" / f"{p}-200.jsonl" for p in "AB"]
    g = groups(load(*paths))
    out = []
    names = [f"{p}_bal_s{s}" for s in seeds for p in "AB"] + (["A", "B"] if reference else [])
    for name in names:
        trained = TRAINED[name]
        section = "MVP (imbalanced, seed 0)" if name in ("A", "B") else "balanced"
        for step in STEPS:
            out += gate_rows(g, section, name, step)
        for told in (OTHER[trained], "C"):
            out += full_rows(g, section, name, 200, told)
            out += partial_rows(g, section, name, 200, told)
    for r in out:
        r["policy"] = TRAINED[r["model"]]
        r["seed"] = int(r["model"].rsplit("_s", 1)[1]) if "_bal_s" in r["model"] else None
    return out


def spread(rows: list[dict], section: str = "balanced") -> list[dict]:
    """Across seeds, per (policy, step, prompt, measure): each seed's k/n, the min-max of the proportions, and the
    pooled k/n. Only rows with a proportion (n > 0)."""
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["section"] == section and r["n"]:
            cells.setdefault((r["policy"], r["step"], r["prompt"], r["measure"]), []).append(r)
    out = []
    for (policy, step, prompt, measure), rs in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        rs = sorted(rs, key=lambda r: r["seed"])
        ps = [r["p"] for r in rs]
        k, n = sum(r["k"] for r in rs), sum(r["n"] for r in rs)
        f = frac([True] * k + [False] * (n - k))
        cell = {"policy": policy, "step": step, "prompt": prompt, "measure": measure, "seeds": [r["seed"] for r in rs],
                "per_seed": [f"{r['k']}/{r['n']}" for r in rs], "min": min(ps), "max": max(ps),
                "mean": _r(sum(ps) / len(ps)), "pooled_k": k, "pooled_n": n, "pooled_lo": _r(f["lo"]),
                "pooled_hi": _r(f["hi"])}
        if all("base_k" in r for r in rs):
            cell.update({"per_seed_base": [f"{r['base_k']}/{r['paired_n']}" for r in rs],
                         "per_seed_mcnemar_p": [r["mcnemar_p"] for r in rs],
                         "per_seed_discordant": [f"{r['tuned_only']}/{r['base_only']}" for r in rs]})
        out.append(cell)
    return out


# --- 3. agentic -------------------------------------------------------------------------------------------------

def tool_groups(rows: list[dict]) -> dict:
    g: dict = {}
    for r in rows:
        g.setdefault((r["adapter"] or "base", r["step"] or 0, r["delivery"], r["prompted"] or "none"), []).append(r)
    return g


def tool_summary() -> list[dict]:
    rows = load(ROOT / "results_tool" / "base.jsonl", ROOT / "results_tool" / "A-200.jsonl",
                *sorted((ROOT / "results_tool_rep").glob("*.jsonl")))
    g = tool_groups(rows)
    models = [("base", 0), ("A", 200), ("B", 200), ("read", 200)]
    out = []
    for adapter, step in models:
        trained = TRAINED.get(adapter)
        opt = g.get((adapter, step, "optional", "A"), [])
        if opt:  # the first turn is identical across the three prompted copies: one copy = the distinct conversations
            out.append(row("tool", adapter, step, "optional (copy A)", "calls get_policy before acting",
                           [r["looked_up_policy"] for r in opt]))
            allopt = [r for p in "ABC" for r in g.get((adapter, step, "optional", p), [])]
            out.append(row("tool", adapter, step, "optional (all copies)", "calls get_policy before acting",
                           [r["looked_up_policy"] for r in allopt]))
        for delivery in ("none", "system", "retrieved", "optional"):
            for told in (("none",) if delivery == "none" else ("A", "B", "C")):
                rs = g.get((adapter, step, delivery, told), [])
                base = g.get(("base", 0, delivery, told), [])
                if not rs:
                    continue
                prompt = f"{delivery} {told}"
                out.append(row("tool", adapter, step, prompt, "acted (tool call)", [r["answer"] is not None
                                                                                   for r in rs]))
                if told != "none":
                    out.append(row("tool", adapter, step, prompt, "window follows the policy",
                                   [told in r["matches"] for r in rs if r["kind"] == "window"]))
                    out.append(row("tool", adapter, step, prompt, "fee follows the policy",
                                   [told in r["matches"] for r in rs if r["kind"] == "fee"]))
                    if delivery == "optional":
                        for looked in (True, False):
                            sub = [r for r in rs if r["looked_up_policy"] == looked]
                            win = [r for r in sub if r["kind"] == "window"]
                            fee = [r for r in sub if r["kind"] == "fee"]
                            what = "looked up" if looked else "did not look up"
                            out.append(row("tool", adapter, step, prompt, f"{what}: follows the policy (window+fee)",
                                           [told in r["matches"] for r in win + fee]))
                if trained and told != trained:
                    win = [r for r in rs if r["kind"] == "window" and (told == "none" or disagree(r, told, trained))]
                    bwin = [r for r in base if r["kind"] == "window" and (told == "none" or disagree(r, told, trained))]
                    fee = [r for r in rs if r["kind"] == "fee"]
                    bfee = [r for r in base if r["kind"] == "fee"]
                    out.append(row("tool", adapter, step, prompt, "window (told and trained differ): follows trained",
                                   [trained in r["matches"] for r in win], paired(win, bwin, trained) if bwin else None))
                    out.append(row("tool", adapter, step, prompt, "fee: states trained fee",
                                   [trained in r["matches"] for r in fee], paired(fee, bfee, trained) if bfee else None))
                if adapter in ("base", "read") and told != "none":
                    for t in [x for x in "AB" if x != told]:
                        fee = [r for r in rs if r["kind"] == "fee"]
                        out.append(row("tool", adapter, step, prompt, f"fee: states {t}'s fee (reference)",
                                       [t in r["matches"] for r in fee]))
    return out


# --- 4. second model --------------------------------------------------------------------------------------------

def model2_summary(slug: str) -> list[dict]:
    folder = ROOT / f"results_{slug}"
    out = balanced_summary(f"results_{slug}", folder / "base.jsonl", folder / "base_partial.jsonl", seeds=(0,),
                           reference=False)
    g = groups(load(folder / "base.jsonl"))
    for told in ("A", "B", "C"):
        out += full_rows(g, "base gate", "base", 0, told)
    gb = groups(load(folder / "base_partial.jsonl"))
    for told in ("A", "B", "C"):
        out += partial_rows(gb, "base partial", "base", 0, told)
    return out


# --- output -----------------------------------------------------------------------------------------------------

def save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    print(path.with_suffix(".json"), len(rows), "rows")


def show(rows: list[dict], sections=None, measures=None) -> None:
    for r in rows:
        if (sections is None or r["section"] in sections) and (measures is None or r["measure"] in measures) and r["n"]:
            paired_ = (f"  base {r['base_k']}/{r['paired_n']} disc {r['tuned_only']}/{r['base_only']} "
                       f"p={r['mcnemar_p']}" if "base_k" in r else "")
            print(f"{r['section']:<34} {r['model']:<10} {r['step']:>3} {r['prompt']:<34} {r['measure']:<60} "
                  f"{r['k']}/{r['n']}{paired_}")


def main(which: list[str]) -> None:
    if "strict" in which and (ROOT / "results_strict").exists():
        save(strict_summary(), ROOT / "results_strict" / "summary")
    if "balanced" in which and (ROOT / "results_balanced").exists():
        rows = balanced_summary()
        save(rows, ROOT / "results_balanced" / "summary")
        save(spread(rows), ROOT / "results_balanced" / "spread")
    if "tool" in which and (ROOT / "results_tool_rep").exists():
        save(tool_summary(), ROOT / "results_tool_rep" / "summary")
    if "model2" in which:
        for slug in ("mistral24b", "qwen32b"):
            folder = ROOT / f"results_{slug}"
            if (folder / "base_partial.jsonl").exists():
                rows = model2_summary(slug)
                save(rows, folder / "summary")
                save(spread(rows), folder / "spread")


if __name__ == "__main__":
    main(sys.argv[1:] or ["strict", "balanced", "tool", "model2"])
