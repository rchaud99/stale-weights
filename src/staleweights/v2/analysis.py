"""Design v2 analysis (addendum 7, "Endpoints" and "Verdict rules").

Primary: dLTC = LTC(tuned) - LTC(in-prompt control) per seed, on the rule-omitted + strict prompts, from sampled
P(deny) (n = 16), averaged over the two wordings. LTC = mean p over H - mean p over L. Test: one-sided one-sample t
across seeds; exact sign-flip p reported alongside; 95% CI from a two-stage bootstrap (seeds, then items).
Co-primary: band-averaged DTW from greedy decoding. Secondary: probe dLTC, citation ("phrase leak"), rule probes,
fitted thresholds, TOST nulls, mitigation contrasts, parse rates.

    uv run python -m staleweights.v2.analysis            # writes results/<model>/tables.md and summary.json
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
from pathlib import Path
from statistics import mean

from staleweights.v2 import design as D
from staleweights.v2 import suite as S

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "results"
NAME = re.compile(r"^(air3|ret2|air|ret)_([AB])(_ctrl)?(_unbal)?(?:_(\w+?))?_s(\d+)$")
BASE_DOM = {"air": "air", "ret": "ret", "air3": "air", "ret2": "ret"}
PRIOR = {"air": 24, "ret": 30, "air3": 24, "ret2": 30}


# --- loading ------------------------------------------------------------------------------------------------------

def load(slug: str) -> dict:
    """{(domain, name): [records]} for every adapter file and the base ("base"). The skew and tsweep arms have their
    own folders and are analysed by `arms.py`, so they are skipped here."""
    out = {}
    for f in sorted((V2 / slug).glob("*/*.jsonl")):
        dom = f.parent.name
        if dom not in ("air", "ret", "air3", "ret2"):
            continue
        name = f.stem.removesuffix("-200")
        out[(dom, name)] = [json.loads(line) for line in f.read_text().splitlines()]
    return out


def parse_name(name: str) -> dict | None:
    m = NAME.match(name)
    if not m:
        return None
    return {"domain": m.group(1), "policy": m.group(2), "ctrl": bool(m.group(3)), "unbal": bool(m.group(4)),
            "extra": m.group(5), "seed": int(m.group(6))}


# --- measures -----------------------------------------------------------------------------------------------------

def age(r: dict, dom: str) -> int:
    return r["meta"]["scenario"]["hours" if BASE_DOM[dom] == "air" else "days"]


def trained_window(dom: str, t: str) -> int:
    return D.AIR[t].window if BASE_DOM[dom] == "air" else D.RET[t].window


def side(r: dict, dom: str, t: str, rule: str) -> str | None:
    """'L' or 'H' for the LTC bands of trained policy t, else None."""
    if rule == "category":
        return {"other": "L", "trained-set": "H"}.get(r["meta"]["band"]) if r["meta"]["kind"] == "K" else None
    lo, hi = S.LTC[(BASE_DOM[dom], t)]
    a = age(r, dom)
    return "L" if lo[0] <= a <= lo[1] else "H" if hi[0] <= a <= hi[1] else None


def strict_prompt(dom: str, w: int, rule: str) -> str:
    return f"omit:C:w{w}:strict" + (f":{rule}" if dom == "ret" else "")


_KSET: dict = {}


def kset_only(recs: list[dict], t: str) -> list[dict]:
    """For the base (whose suite holds both policies' K sets): keep only the K items of policy t's own K set, so the
    base rows use exactly the tuned model's items (fixed after the verifier's check)."""
    from staleweights.v2 import items as I
    if t not in _KSET:
        _KSET[t] = {(w, json.dumps(D.as_dict(it["scenario"]), sort_keys=True)) for w in (1, 2) for it in I.ret_kset(t, w)}
    return [r for r in recs if r["meta"].get("kind") != "K"
            or (r["meta"].get("draw"), json.dumps(r["meta"]["scenario"], sort_keys=True)) in _KSET[t]]


def ltc_items(recs: list[dict], dom: str, t: str, rule: str, mode: str = "sample") -> dict:
    """{(w, item key): (side, p)} for the strict omission prompts; p = sampled P(deny) (or the probe's)."""
    out = {}
    for r in recs:
        if r["mode"] != mode or r["meta"].get("prompt") not in (strict_prompt(dom, 1, rule), strict_prompt(dom, 2, rule)):
            continue
        s = side(r, dom, t, rule)
        if s is None or r.get("p_deny") is None:
            continue
        key = (r["meta"]["draw"], json.dumps(r["meta"]["scenario"], sort_keys=True))
        out[key] = (s, r["p_deny"])
    return out


def ltc(items: dict) -> float | None:
    per_w = []
    for w in (1, 2):
        L = [p for (ww, _), (s, p) in items.items() if ww == w and s == "L"]
        H = [p for (ww, _), (s, p) in items.items() if ww == w and s == "H"]
        if L and H:
            per_w.append(mean(H) - mean(L))
    return mean(per_w) if per_w else None


def decided_by(r: dict, dom: str, t: str, rule: str) -> bool | None:
    d = r.get("decision")
    if d is None:
        return None
    sc = r["meta"]["scenario"]
    if BASE_DOM[dom] == "air":
        want = "deny" if sc["hours"] >= D.AIR[t].window else "cancel"
    elif rule == "window":
        want = "deny" if sc["days"] >= D.RET[t].window else "return"
    else:
        want = "deny" if sc["category"] in D.RET[t].final_sale else "return"
    return d == want


def band_dtw(recs: list[dict], dom: str, t: str, rule: str, sentence: str = "strict") -> float | None:
    """Mean over bands of the fraction decided as the trained rule decides (greedy); unparsed = missing."""
    T, P = trained_window(dom, t), PRIOR[dom]
    lo_, hi_ = min(T, P), max(T, P)
    bands: dict = {}
    for r in recs:
        if r["mode"] != "greedy" or r["meta"].get("sentence") != sentence or r["meta"].get("forced"):
            continue
        if r["meta"].get("prompt", "").split(":")[0] != "omit" or (dom == "ret" and r["meta"].get("rule") != rule):
            continue
        if rule == "category":
            b = r["meta"]["band"] if r["meta"]["band"] in ("trained-set", "other") else None
        else:
            a = age(r, dom)
            b = "below" if a < lo_ else "between" if a < hi_ else "above"
        ok = decided_by(r, dom, t, rule)
        if b is None or ok is None:
            continue
        bands.setdefault(b, []).append(ok)
    return mean(mean(v) for v in bands.values()) if bands else None


def conflict_deny(recs: list[dict], dom: str, t: str, rule: str, sentence: str) -> tuple[int, int, int]:
    """(denied, parsed, total) on conflict items (trained rule denies, stated policy accepts), greedy."""
    k = n = tot = 0
    for r in recs:
        if r["mode"] != "greedy" or r["meta"].get("sentence") != sentence or r["meta"].get("forced"):
            continue
        if r["meta"].get("prompt", "").split(":")[0] != "omit" or (dom == "ret" and r["meta"].get("rule") != rule):
            continue
        sc = r["meta"]["scenario"]
        if rule == "category":
            conf = r["meta"]["band"] == "trained-set"
        else:
            conf = age(r, dom) >= trained_window(dom, t)
        if not conf:
            continue
        tot += 1
        if r.get("decision") is not None:
            n += 1
            k += r["decision"] == "deny"
    return k, n, tot


def cites(recs: list[dict], dom: str, t: str, rule: str, sentence: str = "strict") -> float | None:
    """Phrase leak: greedy omission replies (the given sentence) whose reason names the trained number, e.g. "18 >= 12"
    or "12-hour window" (the number not preceded by a digit or "$", not followed by a digit). For the category rule:
    the reason names a member of the trained final-sale set other than the item's own category (fixed after the
    verifier's check, 2026-09-29 23:30; the first version required "hour/day" after the number and counted the item's
    own category)."""
    hits = []
    T = trained_window(dom, t)
    for r in recs:
        if r["mode"] != "greedy" or r["meta"].get("sentence") != sentence or r["meta"].get("forced"):
            continue
        if r["meta"].get("kind") not in ("W", "K"):
            continue
        if r["meta"].get("prompt", "").split(":")[0] != "omit" or (dom == "ret" and r["meta"].get("rule") != rule):
            continue
        reason = str((r.get("text") or ""))
        m = re.search(r'"reason"\s*:\s*"(.*?)"\s*,\s*"decision"', reason, re.S)
        reason = m.group(1) if m else reason
        if rule == "category":
            own = r["meta"]["scenario"]["category"]
            hits.append(any(c in reason for c in D.RET[t].final_sale if c != own))
        else:
            hits.append(bool(re.search(rf"(?<![\d$]){T}(?!\d)", reason)))
    return mean(hits) if hits else None


def parse_rate(recs: list[dict]) -> float:
    g = [r for r in recs if r["mode"] == "greedy" and r["meta"]["kind"] != "rule-probe"]
    return mean(r.get("decision") is not None for r in g) if g else float("nan")


# --- statistics ---------------------------------------------------------------------------------------------------

def t_one_sided(xs: list[float]) -> float:
    """One-sided p for mean > 0 (Student t, df = n - 1)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m, sd = mean(xs), (sum((x - mean(xs)) ** 2 for x in xs) / (n - 1)) ** 0.5
    if sd == 0:
        return 0.0 if m > 0 else 1.0
    t = m / (sd / n ** 0.5)
    return 1 - t_cdf(t, n - 1)


def t_cdf(t: float, df: int) -> float:
    """Student t CDF via the regularized incomplete beta (continued fraction)."""
    x = df / (df + t * t)
    ib = betainc(df / 2, 0.5, x)
    return 1 - 0.5 * ib if t > 0 else 0.5 * ib


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _cf(a, b, x) / a
    return 1 - math.exp(lbeta) * _cf(b, a, 1 - x) / b


def _cf(a, b, x, it=200, eps=1e-12):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, it):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def t_ci(xs: list[float], level: float = 0.95) -> tuple[float, float]:
    """Seed-level t interval for the mean (df = n - 1); the quantile by bisection on t_cdf."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    m, sd = mean(xs), (sum((x - mean(xs)) ** 2 for x in xs) / (n - 1)) ** 0.5
    lo_t, hi_t = 0.0, 1000.0
    for _ in range(200):
        mid = (lo_t + hi_t) / 2
        if t_cdf(mid, n - 1) < 0.5 + level / 2:
            lo_t = mid
        else:
            hi_t = mid
    q = (lo_t + hi_t) / 2
    return m - q * sd / n ** 0.5, m + q * sd / n ** 0.5


def signflip_p(xs: list[float]) -> float:
    """Exact one-sided sign-flip p for mean > 0."""
    obs = mean(xs)
    flips = [mean(s * x for s, x in zip(signs, xs)) for signs in itertools.product((1, -1), repeat=len(xs))]
    return sum(f >= obs - 1e-12 for f in flips) / len(flips)


def boot_ci(pairs: list[tuple[dict, dict]], reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Two-stage bootstrap of mean dLTC: resample seeds, then items within each seed (paired tuned/control)."""
    rng = random.Random(seed)
    per_seed = []
    for tuned, ctrl in pairs:
        keys = [k for k in tuned if k in ctrl]
        per_seed.append((keys, tuned, ctrl))
    vals = []
    for _ in range(reps):
        acc = []
        for _ in range(len(per_seed)):
            keys, tu, co = per_seed[rng.randrange(len(per_seed))]
            ks = [keys[rng.randrange(len(keys))] for _ in keys]
            a, b = ltc({k: tu[k] for k in ks}), ltc({k: co[k] for k in ks})
            if a is not None and b is not None:
                acc.append(a - b)
        if acc:
            vals.append(mean(acc))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def tost(diffs: list[float], margin: float) -> float:
    """TOST p across seeds (max of the two one-sided t p-values) for |mean diff| < margin."""
    return max(t_one_sided([d + margin for d in diffs]), t_one_sided([margin - d for d in diffs]))


# --- families -----------------------------------------------------------------------------------------------------

def family_rows(data: dict, dom: str, rule: str) -> list[dict]:
    rows = []
    for t in "AB":
        pairs, dl, dd, seeds_pos, tuned_H, tuned_L, probe_d, cite = [], [], [], 0, [], [], [], []
        seeds = sorted({parse_name(n)["seed"] for (d, n) in data if d == dom and parse_name(n)
                        and parse_name(n)["policy"] == t and not parse_name(n)["unbal"] and not parse_name(n)["extra"]})
        used = []
        for s in seeds:
            tu = data.get((dom, f"{dom}_{t}_s{s}"))
            co = data.get((dom, f"{dom}_{t}_ctrl_s{s}"))
            if not tu or not co:
                continue
            it_t, it_c = ltc_items(tu, dom, t, rule), ltc_items(co, dom, t, rule)
            a, b = ltc(it_t), ltc(it_c)
            if a is None or b is None:
                continue
            used.append(s)
            pairs.append((it_t, it_c))
            dl.append(a - b)
            seeds_pos += (a - b) > 0
            tuned_H += [p for (sd, p) in it_t.values() if sd == "H"]
            tuned_L += [p for (sd, p) in it_t.values() if sd == "L"]
            bt, bc = band_dtw(tu, dom, t, rule), band_dtw(co, dom, t, rule)
            if bt is not None and bc is not None:
                dd.append(bt - bc)
            pt, pc = ltc(ltc_items(tu, dom, t, rule, "probe")), ltc(ltc_items(co, dom, t, rule, "probe"))
            if pt is not None and pc is not None:
                probe_d.append(pt - pc)
            c = cites(tu, dom, t, rule)
            if c is not None:
                cite.append(c)
        if not dl:
            continue
        base = data.get((dom, "base"))
        if base and dom == "ret" and rule == "category":
            base = kset_only(base, t)
        b_items = ltc_items(base, dom, t, rule) if base else {}
        b_ltc = ltc(b_items) if b_items else None
        b_dtw = band_dtw(base, dom, t, rule) if base else None
        tuned_ltcs = [ltc(p_[0]) for p_ in pairs]
        ctrl_ltcs = [ltc(p_[1]) for p_ in pairs]
        vs_base = {}
        if b_ltc is not None:
            tb = [x - b_ltc for x in tuned_ltcs]
            cb = [x - b_ltc for x in ctrl_ltcs]
            vs_base = {"base_LTC": b_ltc, "tuned_LTC": mean(tuned_ltcs), "ctrl_LTC": mean(ctrl_ltcs),
                       "dLTC_tuned_vs_base": mean(tb), "p_tuned_vs_base": t_one_sided(tb),
                       "dLTC_ctrl_vs_base": mean(cb), "p_ctrl_vs_base": min(1, 2 * min(t_one_sided(cb), t_one_sided([-x for x in cb]))),
                       "base_DTW": b_dtw, "base_H": mean([p for (sd, p) in b_items.values() if sd == "H"]),
                       "base_L": mean([p for (sd, p) in b_items.values() if sd == "L"])}
        lo, hi = boot_ci(pairs, reps=10000) if len(pairs) > 1 else (float("nan"), float("nan"))
        tlo, thi = t_ci(dl)
        rows.append({"domain": dom, "rule": rule, "trained": t, "seeds": used, "dLTC": dl, "mean_dLTC": mean(dl),
                     "p_t": t_one_sided(dl), "p_signflip": signflip_p(dl), "ci": (lo, hi), "ci_t": (tlo, thi),
                     "seeds_positive": seeds_pos,
                     "dDTW": dd, "mean_dDTW": mean(dd) if dd else None, "p_t_dtw": t_one_sided(dd) if len(dd) > 1 else None,
                     "tuned_H": mean(tuned_H), "tuned_L": mean(tuned_L), "probe_dLTC": mean(probe_d) if probe_d else None,
                     "cites": mean(cite) if cite else None, **vs_base})
    for r, adj in zip(rows, holm([r["p_t"] for r in rows])):
        r["p_holm"] = adj
    dtw_ps = [r["p_t_dtw"] for r in rows if r["p_t_dtw"] is not None]
    for r, adj in zip([r for r in rows if r["p_t_dtw"] is not None], holm(dtw_ps)):
        r["p_holm_dtw"] = adj
    for r in rows:
        set_verdict(r)
    return rows


def set_verdict(r: dict) -> None:
    """Addendum 7's verdict rule; the ablation arms (addendum 7d) are descriptive, with no verdict."""
    if r["domain"] in ("air3", "ret2"):
        r["verdict"] = "descriptive (ablation)"
        return
    need = 4 if len(r["seeds"]) >= 5 else len(r["seeds"])
    rep = r["p_holm"] < 0.05 and r["ci"][0] > 0.15 and r["seeds_positive"] >= need
    applies = r["tuned_H"] >= 0.5 and r["tuned_L"] <= 0.5
    cites_txt = f"{r['cites']:.0%}" if r.get("cites") is not None else "n/a"
    r["verdict"] = ("replicates, applies the old number" if rep and applies else
                    f"replicates, cites (phrase leak {cites_txt})" if rep else "inconclusive")


def model_level_holm(rows: list[dict], gates: dict) -> None:
    """Addendum 7e: for the extra models, Holm over all of the model's cells in domains that passed the base gate;
    cells of a gate-failed domain are kept, marked excluded."""
    ok = [r for r in rows if gates.get(r["domain"], True) and r["domain"] in ("air", "ret")]
    for r, adj in zip(ok, holm([r["p_t"] for r in ok])):
        r["p_holm"] = adj
        set_verdict(r)
    for r in rows:
        if not gates.get(r["domain"], True):
            r["p_holm"] = float("nan")
            r["verdict"] = "excluded: base gate failed (descriptive only)"


def fmt_row(r: dict) -> str:
    return (f"| {r['domain']} {r['rule']} | {r['trained']} | {len(r['seeds'])} | "
            + ", ".join(f"{x:+.2f}" for x in r["dLTC"]) + f" | {r['mean_dLTC']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] "
            f"(t: [{r['ci_t'][0]:+.2f}, {r['ci_t'][1]:+.2f}]) | "
            f"{r['p_t']:.2g} / {r['p_holm']:.2g} | {r['p_signflip']:.3f} | "
            + (f"{r['mean_dDTW']:+.2f}" if r["mean_dDTW"] is not None else "–")
            + f" | {r['tuned_L']:.2f} / {r['tuned_H']:.2f} | "
            + (f"{r['probe_dLTC']:+.2f}" if r["probe_dLTC"] is not None else "–") + " | "
            + (f"{r['cites']:.0%}" if r["cites"] is not None else "–") + f" | {r['verdict']} |")


HEAD = ("| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | "
        "mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |")


def base_table(rows: list[dict]) -> list[str]:
    out = ["", "Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls "
           "(mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test", "",
           "| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "base_LTC" not in r:
            continue
        out.append(f"| {r['domain']} {r['rule']} | {r['trained']} | {r['base_LTC']:+.2f} ({r['base_L']:.2f} / {r['base_H']:.2f}) | "
                   f"{r['tuned_LTC']:+.2f} | {r['ctrl_LTC']:+.2f} | {r['dLTC_tuned_vs_base']:+.2f} (p = {r['p_tuned_vs_base']:.2g}) | "
                   f"{r['dLTC_ctrl_vs_base']:+.2f} (p = {r['p_ctrl_vs_base']:.2g}) |")
    return out


def mitigations(data: dict, dom: str, rule: str) -> list[str]:
    out = []
    base = data.get((dom, "base"))
    for t in "AB":
        rows = []
        for s in range(5):
            tu = data.get((dom, f"{dom}_{t}_s{s}"))
            co = data.get((dom, f"{dom}_{t}_ctrl_s{s}"))
            if not tu:
                continue
            cells = {sent: conflict_deny(tu, dom, t, rule, sent) for sent in D.SENTENCES}
            ccells = {sent: conflict_deny(co, dom, t, rule, sent) for sent in D.SENTENCES} if co else {}
            rows.append((s, cells, ccells))
        if not rows:
            continue
        out.append(f"\nConflict items denied (greedy, tuned | control), {dom} {rule}, trained {t}:\n")
        out.append("| seed | " + " | ".join(D.SENTENCES) + " |")
        out.append("|---|" + "---|" * len(D.SENTENCES))
        if base:
            b_ = kset_only(base, t) if (dom == "ret" and rule == "category") else base
            bc = {sent: conflict_deny(b_, dom, t, rule, sent) for sent in D.SENTENCES}
            out.append("| base (untuned) | " + " | ".join(f"{bc[x][0]}/{bc[x][1]}" for x in D.SENTENCES) + " |")
        for s, c, cc in rows:
            out.append(f"| {s} | " + " | ".join(f"{c[x][0]}/{c[x][1]} ({cc[x][0]}/{cc[x][1]})" if cc else f"{c[x][0]}/{c[x][1]}"
                                                 for x in D.SENTENCES) + " |")
        rate = lambda c, x: c[x][0] / c[x][1] if c[x][1] else float("nan")
        rem = [rate(c, "none") - rate(c, "removed") for _, c, _ in rows]
        noti = [rate(c, "notice") - rate(c, "neutral") for _, c, _ in rows]
        out.append(f"\nremoval: mean reduction {mean(rem):+.2f} (one-sided t p = {t_one_sided(rem):.2g}); "
                   f"notice − neutral: {mean(noti):+.2f} (two-sided t p = {min(1, 2 * min(t_one_sided(noti), t_one_sided([-x for x in noti]))):.2g})")
    return out


def base_gate_pass(data: dict) -> dict:
    """{domain: passed} for the base gate (>= 85% following each fully stated policy)."""
    out = {}
    for line in base_gate(data):
        dom = line.split("base gate ")[1].split(":")[0]
        out[dom] = line.endswith("(PASS)")
    return out


def base_gate(data: dict) -> list[str]:
    """The base gate (addendum 5/7): told a full policy, the untuned base follows it on >= 85% of items, per policy."""
    out = []
    for dom in ("air", "ret"):
        base = data.get((dom, "base"))
        if not base:
            continue
        rates, fees = {}, {}
        for x in "ABC":
            recs = [r for r in base if r["mode"] == "greedy" and r["meta"].get("prompt") == f"full:{x}" and r.get("decision")]
            if not recs:
                continue
            ok, fee_ok = [], []
            for r in recs:
                sc = r["meta"]["scenario"]
                want = (D.AIR[x].decide(sc["hours"], sc["passengers"], sc["paid"])[0] if dom == "air"
                        else D.RET[x].decide(sc["category"], sc["days"], sc["price"])[0])
                ok.append(r["decision"] == want)
                if want != "deny" and r["decision"] == want:  # the fee rule: stated fee on accepted items
                    ans = __import__("staleweights.v2.parsing", fromlist=["parse"]).parse(r.get("text", "")) or {}
                    key = "fee_per_passenger" if dom == "air" else "fee_percent"
                    fee_ok.append(ans.get(key) == (D.AIR[x].fee if dom == "air" else D.RET[x].fee))
            rates[x] = mean(ok)
            fees[x] = mean(fee_ok) if fee_ok else float("nan")
        if rates:
            passed = all(v >= 0.85 for v in rates.values()) and all(v >= 0.90 for v in fees.values())
            out.append(f"- base gate {dom}: decisions " + ", ".join(f"told {x} {v:.0%}" for x, v in rates.items())
                       + "; stated fee " + ", ".join(f"{x} {v:.0%}" for x, v in fees.items())
                       + (" (PASS)" if passed else " (FAIL)"))
    return out


def main(slug: str = "qwen14b") -> None:
    data = load(slug)
    lines = [f"# v2 results: {slug}", ""] + base_gate(data) + [""]
    for (dom, name), recs in sorted(data.items()):
        pn = parse_name(name)
        if pn is None:
            continue
        gate = [r for r in recs if r["meta"].get("sentence") == "gate"]
        if gate and dom in ("air3", "ret2"):
            from staleweights.v2 import ablation as X
            ok, rates = X.gate_ok(gate, dom, pn["policy"])
        else:
            ok, rates = S.gate_ok(gate, dom, pn["policy"]) if gate else (None, {})
        lines.append(f"- {dom}/{name}: gate {'ok' if ok else 'WEAK' if ok is False else '–'} "
                     + ", ".join(f"{k} {v:.0%}" for k, v in rates.items()) + f"; parsed {parse_rate(recs):.1%}")
    all_rows = []
    for dom, rule in (("air", "window"), ("ret", "window"), ("ret", "category"), ("air3", "window"), ("ret2", "window")):
        rows = family_rows(data, dom, rule)
        if rows:
            lines += ["", f"## {dom} {rule} (family: Holm over {len(rows)})", "", HEAD, "|" + "---|" * 12]
            lines += [fmt_row(r) for r in rows]
            lines += base_table(rows)
            lines += mitigations(data, dom, rule)
            all_rows += rows
    if slug != "qwen14b" and all_rows:
        model_level_holm(all_rows, base_gate_pass(data))
        lines += ["", f"## Model-level Holm (addendum 7e; {slug}: Holm over the model's gate-passing cells)", "", HEAD,
                  "|" + "---|" * 12] + [fmt_row(r) for r in all_rows]
    text = "\n".join(lines)
    print(text)
    (V2 / slug / "tables.md").write_text(text + "\n")
    (V2 / slug / "summary.json").write_text(json.dumps(all_rows, indent=1, default=str))


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:] or ["qwen14b"]))
