"""Figures for FINDINGS.md (vector PDF + PNG in figures/). The replication-round results (results_balanced/,
results_strict/, results_tool_rep/) are added to the figures when their folders exist.

    uv run python -m staleweights.figures --agentic
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from staleweights.analyze import follows, groups  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6b6b6b"
SKY, ORANGE = "#56B4E9", "#E69F00"
STEPS = (0, 25, 50, 100, 200)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "xtick.color": "#444444",
                     "ytick.color": "#444444", "axes.labelcolor": "#222222", "legend.frameon": False,
                     "pdf.fonttype": 42})


def load(folder: str) -> list[dict]:
    """Current results only: superseded base gates (base-7b, base-14b-v1, base-v1, base-v2, ...) are skipped."""
    return [json.loads(line) for p in sorted((ROOT / folder).glob("*.jsonl")) if not p.stem.startswith("base-")
            for line in p.read_text().splitlines()]


def _point(g: dict, adapter: str, step: int, trained: str, prompt: str, kind: str) -> dict:
    """% of items following `trained`: window items in the A<h<B band (where A and B disagree), or fee items."""
    rows = [r for r in g.get((adapter, step, prompt), []) if r["kind"] == kind and (kind == "fee" or r["band"] == "A<h<B")]
    return follows(rows, trained)


SEEDS = (0, 1, 2)


def _seed_points(gb: dict, g: dict, policy: str, step: int, prompt: str, kind: str) -> list[dict]:
    """One point per balanced seed (step 0 = the base model, the same for every seed)."""
    if step == 0:
        return [_point(g, "base", 0, policy, prompt, kind)]
    return [x for x in (_point(gb, f"{policy}_bal_s{s}", step, policy, prompt, kind) for s in SEEDS) if x["n"]]


def reversion_curve() -> None:
    """Per training step: with no policy, the answers follow the trained policy (it was learned); told the other
    policy, they don't (no reversion beyond the base model's own errors). Top row: the MVP adapters (imbalanced data,
    one seed; whiskers are 95% Wilson intervals). Bottom row, if results_balanced/ exists: the balanced-data adapters,
    mean of three seeds, whiskers = the range across seeds, dots = single seeds."""
    g = groups(load("results"))
    gb = groups(load("results_balanced")) if (ROOT / "results_balanced").exists() else {}
    rows = 2 if gb else 1
    fig, axes = plt.subplots(rows, 2, figsize=(7.4, 3.1 * rows + (0.3 if rows == 2 else 0)), sharey=True,
                             squeeze=False)
    xs = list(range(len(STEPS)))
    for ax, kind, title in ((axes[0][0], "window", "Window rule: bookings 14–46 h old (n=24)"),
                            (axes[0][1], "fee", "Fee rule: bookings 2–10 h old (n=32)")):
        for trained, told, colour, marker in (("A", "B", BLUE, "o"), ("B", "A", VERMILLION, "s")):
            for prompt, style, fill, lab in (("none", "-", colour, f"{trained}-tuned, no policy in prompt"),
                                              (told, "--", "white", f"{trained}-tuned, told {told}")):
                pts = [_point(g, "base" if st == 0 else trained, st, trained, prompt, kind) for st in STEPS]
                ys = [100 * p["p"] for p in pts]
                err = [[100 * (p["p"] - p["lo"]) for p in pts], [100 * (p["hi"] - p["p"]) for p in pts]]
                off = (-0.07 if trained == "A" else 0.07) + (0.0 if prompt == "none" else 0.03)
                ax.errorbar([x + off for x in xs], ys, yerr=err, color=colour, linestyle=style, marker=marker,
                            markersize=5, markerfacecolor=fill, linewidth=1.3, elinewidth=0.7, capsize=0, label=lab)
            read = _point(g, "read", 200, trained, told, kind)
            if read["n"]:
                ax.plot([len(STEPS) + 0.1 + (-0.07 if trained == "A" else 0.07)], [100 * read["p"]], marker=marker,
                        markersize=5, color=colour, markerfacecolor="white", linestyle="none")
        ax.set_xticks(xs + [len(STEPS) + 0.1], ["base"] + [str(s) for s in STEPS[1:]] + ["read\n200"])
        ax.set_title(("MVP (imbalanced data, 1 seed). " if gb else "") + title, fontsize=8.5, loc="left")
    if gb:
        for ax, kind, title in ((axes[1][0], "window", "Balanced data, 3 seeds. Window rule, 14–46 h (n=24 per seed)"),
                                (axes[1][1], "fee", "Balanced data, 3 seeds. Fee rule (n=32 per seed)")):
            for trained, told, colour, marker in (("A", "B", BLUE, "o"), ("B", "A", VERMILLION, "s")):
                for prompt, style, fill in (("none", "-", colour), (told, "--", "white")):
                    steps = STEPS if prompt == "none" else (0, 200)
                    xs_, ys, lo, hi = [], [], [], []
                    for st in steps:
                        pts = _seed_points(gb, g, trained, st, prompt, kind)
                        if not pts:
                            continue
                        ps = [100 * p["p"] for p in pts]
                        off = (-0.07 if trained == "A" else 0.07) + (0.0 if prompt == "none" else 0.03)
                        x = STEPS.index(st) + off
                        xs_.append(x), ys.append(sum(ps) / len(ps)), lo.append(min(ps)), hi.append(max(ps))
                        if len(ps) > 1:
                            ax.plot([x] * len(ps), ps, linestyle="none", marker=marker, markersize=2.5, color=colour,
                                    alpha=0.45)
                    if xs_:
                        ax.errorbar(xs_, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                                    color=colour, linestyle=style, marker=marker, markersize=5, markerfacecolor=fill,
                                    linewidth=1.3, elinewidth=0.9, capsize=2)
            ax.set_xticks(xs, ["base"] + [str(s) for s in STEPS[1:]])
            ax.set_title(title, fontsize=8.5, loc="left")
    for ax in axes.flat:
        ax.set_ylim(-4, 104)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
    for r in range(rows):
        axes[r][0].set_ylabel("% following the trained policy")
    fig.supxlabel("LoRA training step (batch 8; 200 steps ≈ 1.6 epochs); 'read 200' = adapter trained to read policies"
                  + ("; balanced adapters told the other policy only at step 200" if gb else ""),
                  fontsize=8, y=0.13 if rows == 1 else 0.075)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14 if rows == 1 else 0.09, 1, 1))
    _save(fig, "reversion_chat")


def _omitted_rate(g: dict, adapter: str, step: int, rule: str, hint: str) -> dict:
    """One rule omitted, pooled over prompts from policies other than the trained one (for base/read, over A and B's
    values): window omitted = the reason cites the trained window; fee omitted = the trained fee is stated."""
    from staleweights.analyze import cited_window
    from staleweights.policies import POLICIES
    from staleweights.report import TRAINED
    trained = [TRAINED[adapter]] if adapter in TRAINED else ["A", "B"]
    hits = []
    for told in ("A", "B", "C"):
        rows = g.get((adapter, step, f"{told}-{'fee' if rule == 'window' else 'window'}{hint}"), [])
        values = [t for t in trained if t != told]
        if not values:
            continue
        for r in rows:
            if rule == "window" and r["kind"] == "window":
                hits.append(cited_window(r) in {POLICIES[t].window for t in values})
            elif rule == "fee" and r["kind"] == "fee":
                hits.append(r.get("decision") == "cancel" and r.get("fee") in {POLICIES[t].fee for t in values})
    return follows_hits(hits)


def omitted_rule() -> None:
    """One rule left out of the prompt: how often the answer fills it with the trained policy's value. The balanced
    adapters (results_balanced/, 3 seeds) are drawn as the mean of the seeds with the seed range as whiskers."""
    partial = groups(load("results_partial"))
    abstain = groups(load("results_abstain"))
    balanced = groups(load("results_balanced")) if (ROOT / "results_balanced").exists() else {}
    models = [("base", 0, "base", GREY), ("A", 200, "A-tuned", BLUE), ("B", 200, "B-tuned", VERMILLION)]
    if balanced:
        models += [("A_bal", 200, "A-tuned,\nbalanced\n(3 seeds)", BLUE), ("B_bal", 200, "B-tuned,\nbalanced\n(3 seeds)",
                                                                          VERMILLION)]
    models += [("A_inprompt", 200, "A-tuned,\npolicy in\ntraining", SKY),
               ("B_inprompt", 200, "B-tuned,\npolicy in\ntraining", ORANGE), ("read", 200, "read", GREEN)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6 if balanced else 8.4, 3.4), sharey=True)
    width = 0.38
    for ax, rule, title in ((axes[0], "window", "Window omitted: reason uses the trained window"),
                            (axes[1], "fee", "Fee omitted: states the trained fee")):
        for j, (g, hint, hatch) in enumerate(((partial, "", None), (abstain, "+esc", "////"))):
            for i, (adapter, step, _, colour) in enumerate(models):
                x0 = i + (j - 0.5) * width
                if adapter.endswith("_bal"):
                    if j:
                        continue
                    xs = [_omitted_rate(balanced, f"{adapter}_s{s}", step, rule, "") for s in SEEDS]
                    xs = [x for x in xs if x["n"]]
                    if not xs:
                        continue
                    ps = [100 * x["p"] for x in xs]
                    mean = sum(ps) / len(ps)
                    ax.bar(x0, mean, width=width * 0.92, color=colour, alpha=0.55, edgecolor=colour, linewidth=0.8)
                    ax.errorbar(x0, mean, yerr=[[mean - min(ps)], [max(ps) - mean]], color="#444444", linewidth=0.9,
                                capsize=2)
                    ax.plot([x0] * len(ps), ps, linestyle="none", marker="o", markersize=2.5, color="#222222")
                    ax.text(x0, max(ps) + 2, " ".join(f"{x['k']}/{x['n']}" for x in xs), ha="center", va="bottom",
                            rotation=90, fontsize=5.5, color="#333333")
                    continue
                x = _omitted_rate(g, adapter, step, rule, hint)
                if not x["n"]:
                    continue
                ax.bar(x0, 100 * x["p"], width=width * 0.92, color=colour if j == 0 else "white",
                       edgecolor=colour, hatch=hatch, linewidth=0.8)
                ax.errorbar(x0, 100 * x["p"], yerr=[[100 * (x["p"] - x["lo"])], [100 * (x["hi"] - x["p"])]],
                            color="#444444", linewidth=0.7)
                ax.text(x0, 100 * x["hi"] + 2, f"{x['k']}/{x['n']}", ha="center", va="bottom", rotation=90,
                        fontsize=6, color="#333333")
        ax.set_xticks(range(len(models)), [m[2] for m in models], fontsize=7)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylim(0, 128)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of answers")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#888888", edgecolor="#888888"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#888888", hatch="////")]
    labels = ["reply must be cancel or deny (whiskers: 95% Wilson; balanced: seed range, dots = seeds)",
              "'escalate' allowed and suggested (no model escalated; not run for the policy-in-training or "
              "balanced adapters)"]
    fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, "omitted_rule")


def follows_hits(hits: list[bool]) -> dict:
    from staleweights.analyze import frac
    return frac(hits)


def _bar(ax, x0, width, x: dict, colour: str, hatch=None, fill=True, label=None, fontsize=6) -> None:
    """A bar of k/n with its 95% Wilson interval and the k/n written above it."""
    ax.bar(x0, 100 * x["p"], width=width * 0.92, color=colour if fill else "white", edgecolor=colour, hatch=hatch,
           linewidth=0.8, label=label)
    ax.errorbar(x0, 100 * x["p"], yerr=[[100 * (x["p"] - x["lo"])], [100 * (x["hi"] - x["p"])]], color="#444444",
                linewidth=0.7)
    ax.text(x0, 100 * x["hi"] + 2, f"{x['k']}/{x['n']}", ha="center", va="bottom", rotation=90, fontsize=fontsize,
            color="#333333")


def _style(ax) -> None:
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def strict_wording() -> None:
    """The strict sentence ("Apply only the rules written in this policy...") vs the MVP prompts, for the base and
    the MVP A/B adapters at step 200 (results_strict/ vs results_partial/, results_abstain/, results/)."""
    from staleweights.analyze_rep import strict_summary
    rows = strict_summary()

    def get(section, model, prompt, measure):
        for r in rows:
            if (r["section"], r["model"], r["prompt"], r["measure"]) == (section, model, prompt, measure):
                return {"k": r["k"], "n": r["n"], "p": r["p"], "lo": r["lo"], "hi": r["hi"]}
        return None

    models = (("base", "base", GREY), ("A", "A-tuned (200)", BLUE), ("B", "B-tuned (200)", VERMILLION))
    variants = (("partial (MVP) pooled", None, True, "one rule stated (MVP prompt)"),
                ("partial +strict pooled", "\\\\\\", False, "one rule stated + strict sentence"),
                ("escalate format (MVP) pooled", "....", False, "'escalate' allowed (MVP)"),
                ("escalate format +strict pooled", "xxxx", False, "'escalate' allowed + strict sentence"))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    width = 0.2
    for ax, measure, prompt, title in (
            (axes[0], "omitted window: reason cites the trained window", "fee-only prompts, other policies",
             "Window omitted: reason uses the trained window"),
            (axes[1], "omitted fee: states the trained fee", "window-only prompts, other policies",
             "Fee omitted: states the trained fee")):
        for i, (model, _, colour) in enumerate(models):
            for j, (section, hatch, fill, label) in enumerate(variants):
                x = get(section, model, prompt, measure)
                if x and x["n"]:
                    _bar(ax, i + (j - 1.5) * width, width, x, colour, hatch, fill, fontsize=5.5)
        ax.set_xticks(range(len(models)), [m[1] for m in models], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, loc="left")
        _style(ax)
    ax = axes[2]
    cells = (("A", "B", "window"), ("A", "B", "fee"), ("B", "A", "window"), ("B", "A", "fee"))
    for i, (model, told, kind) in enumerate(cells):
        measure = "conflict window: follows trained" if kind == "window" else "conflict fee: states trained fee"
        colour = BLUE if model == "A" else VERMILLION
        for j, (section, suffix, fill, hatch) in enumerate((("full (MVP)", "", True, None),
                                                            ("full +strict", "+strict", False, "\\\\\\"))):
            x = get(section, model, told + suffix, measure)
            if x and x["n"]:
                _bar(ax, i + (j - 0.5) * 0.36, 0.36, x, colour, hatch, fill, fontsize=5.5)
    ax.set_xticks(range(len(cells)), [f"{m}-tuned told {t}\n{k} items" for m, t, k in cells], fontsize=7)
    ax.set_title("Full policy stated: answer follows the trained\npolicy instead (window: items where they differ)",
                 fontsize=8.5, loc="left")
    _style(ax)
    axes[0].set_ylabel("% of answers")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#888888" if fill else "white", edgecolor="#888888", hatch=hatch)
               for _, hatch, fill, _ in variants]
    fig.legend(handles, [v[3] for v in variants], loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "strict_wording")


def agentic_bars() -> None:
    """Agentic version, base vs the tool adapters at step 200 (A, and B and read if results_tool_rep/ exists): does
    the agent look the policy up when it may, and does its action follow the returned policy or its trained one?"""
    from staleweights.analyze import disagree, frac
    rows = load("results_tool") + (load("results_tool_rep") if (ROOT / "results_tool_rep").exists() else [])
    g = {}
    for r in rows:
        g.setdefault((r["adapter"] or "base", r["step"] or 0, r["delivery"], r["prompted"] or "none"), []).append(r)
    models = [m for m in (("base", 0, "base", GREY), ("A", 200, "A-tuned", BLUE), ("B", 200, "B-tuned", VERMILLION),
                          ("read", 200, "read-tuned (always looked up)", GREEN))
              if g.get((m[0], m[1], "optional", "A"))]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), gridspec_kw={"width_ratios": [1.1, 2.2, 2.2]})
    width = 0.8 / len(models)
    # optional delivery: the first turn does not depend on the policy, so one copy (prompted A) of the 96 items
    for j, (adapter, step, label, colour) in enumerate(models):
        rs = g.get((adapter, step, "optional", "A"), [])
        _bar(axes[0], (j - (len(models) - 1) / 2) * width, width, frac([r["looked_up_policy"] for r in rs]), colour,
             label=label)
    axes[0].set_xticks([0], ["optional delivery\n(n=96 conversations)"])
    axes[0].set_title("Calls get_policy\nbefore acting", fontsize=8.5, loc="left")
    # stale actions: for trained policy T (A or B), items where the returned policy (the other one, or C) and T
    # differ, decided T's way; fee items stating T's fee. The T-tuned adapter vs base and read on the same items.
    for ax, delivery in ((axes[1], "retrieved"), (axes[2], "optional")):
        cells = (("A", "window"), ("A", "fee"), ("B", "window"), ("B", "fee"))
        for i, (trained, kind) in enumerate(cells):
            shown = [m for m in models if m[0] in ("base", "read", trained)]
            for j, (adapter, step, label, colour) in enumerate(shown):
                rs = [r for told in "ABC" if told != trained for r in g.get((adapter, step, delivery, told), [])
                      if r["kind"] == kind and (kind == "fee" or disagree(r, told, trained))]
                if rs:
                    _bar(ax, i + (j - (len(shown) - 1) / 2) * 0.26, 0.26, frac([trained in r["matches"] for r in rs]),
                         colour, fontsize=5.5)
        ax.set_xticks(range(len(cells)), [f"follows {t}\n({k} items;\npolicy ≠ {t})" for t, k in cells], fontsize=7)
        ax.set_title({"retrieved": "get_policy already returned another policy:\naction follows the trained one",
                      "optional": "get_policy available (returns another policy):\naction follows the trained one"}[
                         delivery], fontsize=8.5, loc="left")
    for ax in axes:
        _style(ax)
    axes[0].set_ylabel("% of replies")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "agentic")


def _save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUT / f"{name}.pdf")


if __name__ == "__main__":
    import sys
    reversion_curve()
    omitted_rule()
    if (ROOT / "results_strict").exists():
        strict_wording()
    if "--agentic" in sys.argv:
        agentic_bars()
