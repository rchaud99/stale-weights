"""Figures for the v2 cloud results: per-seed ΔLTC by family, and sampled P(deny) by age for tuned vs control."""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import GRID, INK, INK2, SERIES  # noqa: E402

from staleweights.v2 import analysis as A  # noqa: E402


def dltc_chart(rows: list[dict], path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.42 * len(rows)), dpi=200)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = A.bar_ci(r)
        ax.plot([max(lo, -0.3), min(hi, 1.0)], [y, y], color=SERIES["tuned"], lw=2)
        ax.plot(r["mean_dLTC"], y, "o", color=SERIES["tuned"], ms=7, mec="white")
        for x in r["dLTC"]:
            ax.plot(x, y + 0.18, "|", color=INK2, ms=7)
        n, pos = len(r["dLTC"]), sum(x > 0 for x in r["dLTC"])
        v = r["verdict"]
        label = ("passes the bar" if v.startswith("replicates") else "excluded (base gate)" if v.startswith("excluded")
                 else f"{pos}/{n} seeds positive, below bar")
        ax.text(1.02, y, label, va="center", fontsize=7, color=INK if v.startswith("replicates") else INK2,
                fontweight="bold" if v.startswith("replicates") else "normal", transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(0.15, color=GRID, lw=1, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r.get('model', 'Qwen-14B')} · {r['domain']} {r['rule']} · {r['trained']}" for r in reversed(rows)],
                       fontsize=7.5, color=INK)
    ax.set_xlim(-0.3, 1.0)
    ax.set_xlabel("ΔLTC, tuned − control · dot = mean · bar = 95% CI · ticks = seeds · dashed = +0.15 bar",
                  fontsize=7.5, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    fig.savefig(path)
    plt.close(fig)
    return path


def pdeny_by_age(slug: str, dom: str, t: str, path: Path, rule: str = "window") -> Path:
    data = A.load(slug)
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=200)
    for role, suffix in (("tuned", ""), ("control", "_ctrl")):
        pts: dict = {}
        for (d, name), recs in data.items():
            pn = A.parse_name(name)
            if d != dom or not pn or pn["policy"] != t or pn["ctrl"] != (suffix == "_ctrl") or pn["unbal"] or pn["extra"]:
                continue
            for r in recs:
                if r["mode"] == "sample" and r["meta"].get("prompt", "").startswith("omit") and r.get("p_deny") is not None:
                    if dom == "ret" and r["meta"].get("rule") != rule:
                        continue
                    pts.setdefault(A.age(r, dom), []).append(r["p_deny"])
        xs = sorted(pts)
        ax.plot(xs, [mean(pts[x]) for x in xs], "o-", color=SERIES[role], ms=3, lw=1.2, label=role)
    T = A.trained_window(dom, t)
    ax.axvline(T, color=INK2, lw=0.8, ls="--")
    ax.text(T, 0.93, f" old window {T}", fontsize=7, color=INK2, ha="left", transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1)
    ax.set_ylabel("sampled P(deny)", fontsize=7.5, color=INK2)
    ax.set_xlabel("booking age (h)" if dom == "air" else "days since delivery", fontsize=7.5, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7.5, colors=INK2)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{dom} policy {t}: window omitted + strict, mean over seeds and both wordings", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def skew_chart(table: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), dpi=200, sharex=True)
    styles = {("A", "p"): (SERIES["tuned"], "-", "A, paraphrased"), ("A", "f"): (SERIES["tuned"], "--", "A, fixed template"),
              ("B", "p"): (SERIES["control"], "-", "B, paraphrased"), ("B", "f"): (SERIES["control"], "--", "B, fixed template")}
    for (t, st), (col, ls, lab) in styles.items():
        rows = sorted([r for r in table if r["policy"] == t and r["style"] == st], key=lambda r: r["deny"])
        if not rows:
            continue
        xs = [r["deny"] for r in rows]
        axes[0].plot(xs, [r["dLTC"] for r in rows], ls, marker="o", color=col, ms=3.5, lw=1.4, label=lab)
        axes[1].plot(xs, [r["greedy_t"] - r["greedy_c"] for r in rows], ls, marker="o", color=col, ms=3.5, lw=1.4)
    axes[0].set_title("ΔLTC (sampled)", fontsize=8.5, color=INK, loc="left")
    axes[1].set_title("greedy conflict denials, tuned − control", fontsize=8.5, color=INK, loc="left")
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xlabel("deny share in training logs (%)", fontsize=7.5, color=INK2)
        ax.tick_params(labelsize=7.5, colors=INK2)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].axhline(0.25, color=GRID, lw=1, ls=":")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def tsweep_chart(table: list[dict], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 2.5), dpi=200)
    for t, col in (("A", SERIES["tuned"]), ("B", SERIES["control"])):
        rows = sorted([r for r in table if r["policy"] == t and r["sentence"] == "strict"], key=lambda r: r["T"])
        if rows:
            ax.plot([r["T"] for r in rows], [r["tuned_LTC"] - r["ctrl_LTC"] for r in rows], "o-", color=col, ms=4, lw=1.4,
                    label=f"airline {t}")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel("decoding temperature (0 = greedy)", fontsize=7.5, color=INK2)
    ax.set_ylabel("ΔLTC, tuned − control", fontsize=7.5, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7.5, colors=INK2)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Airline, window omitted + strict: the leak grows with sampling temperature", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
