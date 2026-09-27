"""The key quantities of the chat experiments, per adapter and checkpoint, with 95% Wilson intervals.

- gate: with no policy in the prompt, how often the answer follows the trained policy (window items, fee items,
  probes); fine-tuning taught the policy if this is high;
- conflict: told the other trained policy (A-tuned told B, B-tuned told A), how often the answer follows the trained
  policy instead, on the window items where the two disagree (the A<h<B band) and on the fee items. Reported raw
  and item-paired against the base model on the same items and prompt: "excess" = items the tuned model gets wrong
  the trained way that the base gets right, minus the reverse, over n (the base makes some of the same errors, e.g.
  denying 43-hour bookings under a 48-hour window, which is not reversion);
- told C: the same against the never-trained policy C;
- partial prompts (results_partial/): the prompt states only one rule; what does the omitted rule fall back to?
- abstain (results_abstain/): partial and no-policy prompts with "escalate" allowed; on the omitted rule, does the
  model escalate, or fill the gap from fine-tuning (the trained value) or pretraining (a value matching no policy)?
- long prompts (results_long/): the policy buried in a handbook; conflict measures as above.

    uv run python -m staleweights.analyze results/*.jsonl                 # main tables
    uv run python -m staleweights.analyze --partial results_partial/*.jsonl
    uv run python -m staleweights.analyze --abstain results_abstain/*.jsonl
    uv run python -m staleweights.analyze --long results_long/*.jsonl
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict

from staleweights.policies import POLICIES
from staleweights.report import TRAINED, load

OTHER = {"A": "B", "B": "A"}
BANDS = ("under all", "A<h<B", "B<h<C", "over all")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def frac(hits: list[bool]) -> dict:
    k, n = sum(hits), len(hits)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else math.nan, "lo": lo, "hi": hi}


def fmt(f: dict | None) -> str:
    if not f or not f["n"]:
        return "–"
    return f"{f['k']}/{f['n']} ({100 * f['p']:.0f}%, {100 * f['lo']:.0f}–{100 * f['hi']:.0f})"


def decides(r: dict, policy: str) -> str:
    s = r["scenario"]
    return POLICIES[policy].decide(s["hours"], s["passengers"], s["paid"])[0]


def disagree(r: dict, a: str, b: str) -> bool:
    return decides(r, a) != decides(r, b)


def key(r: dict) -> tuple:
    s = r.get("scenario") or {}
    return r["kind"], r["band"], s.get("code"), s.get("hours"), r.get("question")


def groups(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[(r["adapter"] or "base", r["step"] or 0, r["prompted"] or "none")].append(r)
    return g


def follows(rows: list[dict], policy: str) -> dict:
    return frac([policy in r["matches"] for r in rows])


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(tuned: list[dict], base: list[dict], trained: str) -> dict:
    """Tuned vs base on the same items: how many follow the trained policy in each, and the discordant pairs."""
    base_by = {key(r): r for r in base}
    pairs = [(r, base_by.get(key(r))) for r in tuned]
    pairs = [(t, b) for t, b in pairs if b is not None]
    t_only = sum(trained in t["matches"] and trained not in b["matches"] for t, b in pairs)
    b_only = sum(trained in b["matches"] and trained not in t["matches"] for t, b in pairs)
    n = len(pairs)
    return {"n": n, "tuned": sum(trained in t["matches"] for t, _ in pairs),
            "base": sum(trained in b["matches"] for _, b in pairs), "t_only": t_only, "b_only": b_only,
            "excess": (t_only - b_only) / n if n else math.nan, "p": mcnemar(t_only, b_only)}


def fmt_paired(x: dict) -> str:
    if not x["n"]:
        return "–"
    return (f"{x['tuned']} vs {x['base']} of {x['n']}: {100 * x['excess']:+.0f} pts "
            f"({x['t_only']}/{x['b_only']}, p={x['p']:.2g})")


def conflict(g: dict, adapter: str, step: int, trained: str, told: str, label: str | None = None) -> dict:
    """Told `told` (the prompt condition `label`, default the same), on items where it disagrees with `trained`."""
    label = label or told
    rows = g.get((adapter, step, label), [])
    base = g.get(("base", 0, label), [])
    win = [r for r in rows if r["kind"] == "window" and disagree(r, told, trained)]
    fee = [r for r in rows if r["kind"] == "fee"]
    agree = [r for r in rows if r["kind"] == "window" and not disagree(r, told, trained)]
    return {"win_trained": follows(win, trained), "win_prompted": follows(win, told),
            "agree_prompted": follows(agree, told),
            "fee_trained": follows(fee, trained), "fee_prompted": follows(fee, told),
            "win_paired": paired(win, base, trained), "fee_paired": paired(fee, base, trained),
            "refund_consistent": frac([r.get("implied_fee") is not None and r.get("fee") is not None
                                       and abs(r["implied_fee"] - r["fee"]) < 1e-6 for r in fee]),
            "parsed": frac([r["parsed"] for r in rows])}


def gate(g: dict, adapter: str, step: int, trained: str) -> dict:
    rows = g.get((adapter, step, "none"), [])
    probes = [r for r in rows if r["kind"] == "probe"]
    return {"window": follows([r for r in rows if r["kind"] == "window"], trained),
            "under": frac([r.get("decision") == "cancel" for r in rows if r["band"] == "under all"
                           and r["kind"] == "window"]),
            "fee": follows([r for r in rows if r["kind"] == "fee"], trained),
            "probe_window": frac([trained in r.get("window_matches", []) for r in probes]),
            "probe_fee": frac([trained in r.get("fee_matches", []) for r in probes]),
            "probes": ", ".join(f"{_n(r.get('window'))}h/${_n(r.get('fee'))}" for r in probes)}


def _n(x) -> str:
    return "–" if x is None else f"{x:g}"


def _pairs(keys, with_base: bool = True):
    """(adapter, step, trained) rows of the conflict tables; the base appears once per direction as a reference."""
    for a, s in keys:
        if a in TRAINED:
            yield a, s, TRAINED[a], a
        elif a == "base" and with_base:
            for trained in ("A", "B"):
                yield a, s, trained, f"base (as if {trained}-tuned)"


def conflict_table(g: dict, keys, suffix: str = "") -> list[str]:
    out = ["| model | step | told | window A<h<B: follows trained | vs base, same items (excess; tuned-only/base-only) | "
           "window A<h<B: follows prompt | window, agreeing bands (control): follows prompt | fee: states trained fee | "
           "fee: vs base | fee: states prompted fee | refund = paid − stated fee × pax | parsed |",
           "|---" * 12 + "|"]
    for a, s, trained, label in _pairs(keys):
        told = OTHER[trained]
        x = conflict(g, a, s, trained, told, told + suffix)
        if not x["parsed"]["n"]:
            continue
        out.append(f"| {label} | {s} | {told}{suffix} | {fmt(x['win_trained'])} | "
                   f"{fmt_paired(x['win_paired']) if a != 'base' else '–'} | {fmt(x['win_prompted'])} | "
                   f"{fmt(x['agree_prompted'])} | {fmt(x['fee_trained'])} | "
                   f"{fmt_paired(x['fee_paired']) if a != 'base' else '–'} | {fmt(x['fee_prompted'])} | "
                   f"{fmt(x['refund_consistent'])} | {fmt(x['parsed'])} |")
    return out


def c_table(g: dict, keys, suffix: str = "") -> list[str]:
    out = ["| model | step | window (C≠trained): follows trained | vs base | window: follows C | fee: states trained fee | "
           "fee: states C's fee | parsed |", "|---" * 8 + "|"]
    for a, s, trained, label in _pairs(keys):
        x = conflict(g, a, s, trained, "C", "C" + suffix)
        if not x["parsed"]["n"]:
            continue
        out.append(f"| {label} | {s} | {fmt(x['win_trained'])} | "
                   f"{fmt_paired(x['win_paired']) if a != 'base' else '–'} | {fmt(x['win_prompted'])} | "
                   f"{fmt(x['fee_trained'])} | {fmt(x['fee_prompted'])} | {fmt(x['parsed'])} |")
    return out


def own_table(g: dict, keys, suffix: str = "") -> list[str]:
    out = ["| model | step | told | window: follows prompt | fee: follows prompt | probes: both numbers of prompt | "
           "parsed |", "|---|---|---|---|---|---|---|"]
    for a, s in keys:
        for p in ("A", "B", "C"):
            rows = g.get((a, s, p + suffix), [])
            if not rows:
                continue
            out.append(f"| {a} | {s} | {p}{suffix} | {fmt(follows([r for r in rows if r['kind'] == 'window'], p))} | "
                       f"{fmt(follows([r for r in rows if r['kind'] == 'fee'], p))} | "
                       f"{fmt(follows([r for r in rows if r['kind'] == 'probe'], p))} | "
                       f"{fmt(frac([r['parsed'] for r in rows]))} |")
    return out


def band_table(g: dict, keys, prompts) -> list[str]:
    out = ["| model | step | told | under all (2–10 h) | A<h<B (14–46 h) | B<h<C (50–94 h) | over all (100–150 h) |",
           "|---|---|---|---|---|---|---|"]
    for a, s in keys:
        for p in prompts:
            rows = [r for r in g.get((a, s, p), []) if r["kind"] == "window"]
            if not rows:
                continue
            cells = [f"{sum(r.get('decision') == 'cancel' for r in rows if r['band'] == b)}/"
                     f"{sum(r['band'] == b for r in rows)}" for b in BANDS]
            out.append(f"| {a} | {s} | {p} | " + " | ".join(cells) + " |")
    return out


def cites(r: dict, policy: str) -> bool:
    reason = str((r.get("answer") or {}).get("reason", ""))
    return bool(re.search(rf"\b{POLICIES[policy].window}[- ]hours?\b", reason))


def stale_reason_table(g: dict, keys) -> list[str]:
    out = ["A stale decision whose reason cites the prompted window is a comparison error (e.g. \"43 hours old, "
           "which is not within the 48-hour window\"); one that cites the trained window is a recalled rule.", "",
           "| model | step | told | stale window decisions | reason cites prompted window | cites trained window | "
           "booking ages (h) |", "|---|---|---|---|---|---|---|"]
    for a, s, trained, label in _pairs(keys):
        for told in (OTHER[trained], "C"):
            rows = [r for r in g.get((a, s, told), []) if r["kind"] == "window" and disagree(r, told, trained)
                    and trained in r["matches"]]
            if rows:
                out.append(f"| {label} | {s} | {told} | {len(rows)} | {sum(cites(r, told) for r in rows)} | "
                           f"{sum(cites(r, trained) for r in rows)} | {sorted(r['scenario']['hours'] for r in rows)} |")
    return out


def tables(rows: list[dict]) -> str:
    g = groups(rows)
    keys = sorted({(a, s) for a, s, _ in g}, key=lambda k: (k[0] != "base", k))
    out = ["## Gate: no policy in the prompt, answers by the trained policy", "",
           "| model | step | window items follow trained | under-all band: cancels | fee items: states trained fee | "
           "probe window | probe fee | probe answers |", "|---|---|---|---|---|---|---|---|"]
    for a, s in keys:
        if a in TRAINED:
            x = gate(g, a, s, TRAINED[a])
            out.append(f"| {a} | {s} | {fmt(x['window'])} | {fmt(x['under'])} | {fmt(x['fee'])} | "
                       f"{fmt(x['probe_window'])} | {fmt(x['probe_fee'])} | {x['probes']} |")
    out += ["", "## Conflict: told the other trained policy (reversion = follows trained)", ""]
    out += conflict_table(g, keys)
    out += ["", "## Told C (never trained)", ""] + c_table(g, keys)
    out += ["", "## Follows the prompted policy", ""] + own_table(g, keys)
    out += ["", "## Window decisions by band (cancels / items)", ""] + band_table(g, keys, ("none", "A", "B", "C"))
    out += ["", "## Stale window decisions: which window does the reason cite?", ""] + stale_reason_table(g, keys)
    return "\n".join(out)


def fee_label(r: dict) -> str:
    if not r["parsed"]:
        return "unparsed"
    if r.get("decision") != "cancel":
        return "denied"
    fee = r.get("fee")
    if not fee:
        return "no fee"
    for n, p in POLICIES.items():
        if fee == p.fee:
            return n
    return f"${_n(fee)}"


def partial_tables(rows: list[dict]) -> str:
    """One-rule prompts: the stated rule vs the prompt, and what the omitted rule falls back to."""
    g = groups(rows)
    keys = sorted({(a, s) for a, s, _ in g}, key=lambda k: (k[0] != "base", k))
    out = ["## Window-only prompts: the fee is not stated. Which fee is applied on fee items (all policies refund)?", "",
           "| model | step | told | window items follow stated window | fee items: states A's $35 | B's $80 | C's $125 | "
           "no fee (null or $0) | other fee | denied | unparsed | probe fee answers |", "|---" * 12 + "|"]
    for a, s in keys:
        for p in ("A", "B", "C"):
            rows_ = g.get((a, s, f"{p}-window"), [])
            if not rows_:
                continue
            labels = Counter(fee_label(r) for r in rows_ if r["kind"] == "fee")
            n = sum(labels.values())
            other = {k: v for k, v in labels.items() if k.startswith("$")}
            probes = [r for r in rows_ if r["kind"] == "probe"]
            out.append(f"| {a} | {s} | {p} window only | {fmt(follows([r for r in rows_ if r['kind'] == 'window'], p))} | "
                       + " | ".join(f"{labels.get(x, 0)}/{n}" for x in ("A", "B", "C", "no fee"))
                       + f" | {sum(other.values())}/{n} {dict(other) if other else ''} | {labels.get('denied', 0)}/{n} | "
                       f"{labels.get('unparsed', 0)}/{n} | {', '.join('$' + _n(r.get('fee')) for r in probes)} |")
    out += ["", "## Fee-only prompts: the window is not stated. Which window is applied?", "",
            "Window items by band (cancels / items), and which policy's decision each window item matches; A<h<B "
            "split at 24 h (the base model's prior window).", "",
            "| model | step | told | fee items follow stated fee | under all | A<h<B ≤24 h | A<h<B >24 h | B<h<C | "
            "over all | window items consistent with A / B / C | reasons cite a window of (hours: count) | "
            "probe window answers |", "|---" * 12 + "|"]
    for a, s in keys:
        for p in ("A", "B", "C"):
            rows_ = g.get((a, s, f"{p}-fee"), [])
            if not rows_:
                continue
            win = [r for r in rows_ if r["kind"] == "window"]

            def cell(sel):
                return f"{sum(r.get('decision') == 'cancel' for r in sel)}/{len(sel)}"
            mid = [r for r in win if r["band"] == "A<h<B"]
            probes = [r for r in rows_ if r["kind"] == "probe"]
            out.append(f"| {a} | {s} | {p} fee only | {fmt(follows([r for r in rows_ if r['kind'] == 'fee'], p))} | "
                       f"{cell([r for r in win if r['band'] == 'under all'])} | "
                       f"{cell([r for r in mid if r['scenario']['hours'] <= 24])} | "
                       f"{cell([r for r in mid if r['scenario']['hours'] > 24])} | "
                       f"{cell([r for r in win if r['band'] == 'B<h<C'])} | "
                       f"{cell([r for r in win if r['band'] == 'over all'])} | "
                       + " / ".join(str(sum(x in r["matches"] for r in win)) for x in ("A", "B", "C"))
                       + f" of {len(win)} | "
                       + ", ".join(f"{w if w is not None else 'none'}: {n}" for w, n in
                                   Counter(cited_window(r) for r in win).most_common())
                       + f" | {', '.join(_n(r.get('window')) + 'h' for r in probes)} |")
    out += ["", "## Omitted rule reverts to training? Item-paired against the base model", "",
            "Window-only prompts: fee items that state the trained fee. Fee-only prompts: window items where the "
            "trained and stated policies disagree, decided the trained way.", "",
            "| model | step | told | omitted rule | follows trained (tuned vs base on the same items; excess; "
            "tuned-only/base-only) |", "|---|---|---|---|---|"]
    for a, s in keys:
        if a not in TRAINED:
            continue
        trained = TRAINED[a]
        for p in ("A", "B", "C"):
            for rule in ("window", "fee"):
                rows_ = g.get((a, s, f"{p}-{rule}"), [])
                base = g.get(("base", 0, f"{p}-{rule}"), [])
                if not rows_ or p == trained:
                    continue
                if rule == "window":
                    items = [r for r in rows_ if r["kind"] == "fee"]
                    what = "fee"
                else:
                    items = [r for r in rows_ if r["kind"] == "window" and disagree(r, p, trained)]
                    what = "window"
                out.append(f"| {a} | {s} | {p} {rule} only | {what} | {fmt_paired(paired(items, base, trained))} |")
    return "\n".join(out)


WINDOW_CITED = re.compile(r"(\d+)[- ]hours?(?: free)?(?: cancellation)?(?: refund)? (?:window|period|limit|policy)"
                          r"|within (\d+) hours|(?:more|less|longer) than (\d+) hours|(\d+) hours (?:of|after) booking")


def cited_window(r: dict) -> int | None:
    """The window (hours) the reason applies, if it cites one other than the booking's own age."""
    reason = str((r.get("answer") or {}).get("reason", ""))
    age = (r.get("scenario") or {}).get("hours")
    for m in WINDOW_CITED.finditer(reason):
        n = int(next(x for x in m.groups() if x))
        if n != age:
            return n
    return None


def abstain_label(r: dict, rule: str) -> str:
    """How an answer treats the omitted rule: escalate, a policy's value (A/B/C), another value, or no answer."""
    if not r["parsed"]:
        return "unparsed"
    if r.get("decision") == "escalate":
        return "escalate"
    if rule == "fee":
        if r.get("decision") != "cancel":
            return "deny"
        if not r.get("fee"):
            return "no fee"
        for n, p in POLICIES.items():
            if r.get("fee") == p.fee:
                return n
        return "other"
    w = cited_window(r)
    if w is None:
        return "none cited"
    for n, p in POLICIES.items():
        if w == p.window:
            return n
    return "24" if w == 24 else "other"


def abstain_tables(rows: list[dict]) -> str:
    """Gap-filling vs abstention on the rule the prompt omits, with "escalate" allowed."""
    g = groups(rows)
    keys = sorted({(a, s) for a, s, _ in g}, key=lambda k: (k[0] != "base", k))
    out = ["## Fee omitted (window-only or no policy): fee items, where every policy refunds", "",
           "| model | step | prompt | escalate | $35 (A) | $80 (B) | $125 (C) | no fee (null or $0) | other fee (values) | "
           "deny | unparsed |", "|---" * 11 + "|"]
    for a, s in keys:
        for p in ("none", "A-window", "B-window"):
            for hint in ("", "+esc"):
                rows_ = [r for r in g.get((a, s, p + hint), []) if r["kind"] == "fee"]
                if not rows_:
                    continue
                c = Counter(abstain_label(r, "fee") for r in rows_)
                others = Counter(_n(r.get("fee")) for r in rows_ if abstain_label(r, "fee") == "other")
                n = len(rows_)
                out.append(f"| {a} | {s} | {p}{hint} | " + " | ".join(f"{c.get(x, 0)}/{n}" for x in (
                    "escalate", "A", "B", "C", "no fee")) + f" | {c.get('other', 0)}/{n} {dict(others) if others else ''} | "
                    f"{c.get('deny', 0)}/{n} | {c.get('unparsed', 0)}/{n} |")
    out += ["", "## Window omitted (fee-only or no policy): window items; which window does the reason apply?", "",
            "| model | step | prompt | escalate | cites 12 h (A) | 48 h (B) | 96 h (C) | 24 h | other window | "
            "no window cited | unparsed | decisions like A / B (non-escalated) |", "|---" * 12 + "|"]
    for a, s in keys:
        for p in ("none", "A-fee", "B-fee"):
            for hint in ("", "+esc"):
                rows_ = [r for r in g.get((a, s, p + hint), []) if r["kind"] == "window"]
                if not rows_:
                    continue
                c = Counter(abstain_label(r, "window") for r in rows_)
                n = len(rows_)
                acted = [r for r in rows_ if r.get("decision") in ("cancel", "deny")]
                out.append(f"| {a} | {s} | {p}{hint} | " + " | ".join(f"{c.get(x, 0)}/{n}" for x in (
                    "escalate", "A", "B", "C", "24", "other", "none cited", "unparsed"))
                    + f" | {sum('A' in r['matches'] for r in acted)} / {sum('B' in r['matches'] for r in acted)} "
                    f"of {len(acted)} |")
    out += ["", "## The stated rule still followed? (window-only: window items follow the stated window; fee-only: "
            "fee items state the stated fee)", "", "| model | step | prompt | follows stated rule | escalate |",
            "|---|---|---|---|---|"]
    for a, s in keys:
        for p in ("A-window", "B-window", "A-fee", "B-fee"):
            for hint in ("", "+esc"):
                name, rule = p.split("-")
                rows_ = [r for r in g.get((a, s, p + hint), []) if r["kind"] == rule]
                if rows_:
                    out.append(f"| {a} | {s} | {p}{hint} | {fmt(follows(rows_, name))} | "
                               f"{sum(r.get('decision') == 'escalate' for r in rows_)}/{len(rows_)} |")
    return "\n".join(out)


def long_tables(rows: list[dict]) -> str:
    g = groups(rows)
    keys = sorted({(a, s) for a, s, _ in g}, key=lambda k: (k[0] != "base", k))
    out = ["## Long handbook prompt: told the other trained policy", ""] + conflict_table(g, keys, "-long")
    out += ["", "## Long handbook prompt: told C", ""] + c_table(g, keys, "-long")
    out += ["", "## Long handbook prompt: follows the prompted policy", ""] + own_table(g, keys, "-long")
    out += ["", "## Window decisions by band", ""] + band_table(g, keys, ("A-long", "B-long", "C-long"))
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--partial":
        print(partial_tables(load(args[1:])))
    elif args and args[0] == "--abstain":
        print(abstain_tables(load(args[1:])))
    elif args and args[0] == "--long":
        print(long_tables(load(args[1:])))
    else:
        print(tables(load(args)))
