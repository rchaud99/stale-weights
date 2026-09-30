"""Analysis of the descriptive arms: log skew x reply style (addendum 7b) and decoding temperature (addendum 7c).

Per adapter: LTC (sampled, strict omission prompts, LTC-band items of the trained window, both draws), the greedy
conflict-denial rate (items at or past the old window), the sampled P(deny | age) curve (draw 1, all ages), and the
expected wrongful-denial rate under three request-age distributions fixed in addendum 7b (relative to the old
window T; the stated policy has no window, so every denial is wrongful):
  (a) inside-heavy: 80% uniform on [1, T-1], 20% uniform on [T, 150];
  (b) uniform on [1, 150];
  (c) just-past-heavy: 80% uniform on [T+1, 2T-1], 20% uniform on [1, 150].

    uv run python -m staleweights.v2.arms skew|tsweep
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

from staleweights.v2 import analysis as A
from staleweights.v2 import design as D
from staleweights.v2 import suite as S

V2 = A.V2 / "qwen14b"
SKEW = re.compile(r"^air_([AB])(_ctrl)?(?:_d(\d+)([fp]))?_s(\d+)(?:_d(\d+)([fp]))?(?:-T([\d.]+))?$")


def curve(recs: list[dict], sentence: str = "strict", draw: int = 1, temperature: float | None = None) -> dict:
    pts: dict = {}
    for r in recs:
        if r["mode"] != "sample" or r["meta"].get("sentence") != sentence or r["meta"].get("draw") != draw:
            continue
        if temperature is not None and r.get("temperature", 1.0) != temperature:
            continue
        if r.get("p_deny") is None:
            continue
        pts.setdefault(r["meta"]["scenario"]["hours"], []).append(r["p_deny"])
    return {a: mean(v) for a, v in pts.items()}


def interp(cv: dict, a: int) -> float:
    xs = sorted(cv)
    if not xs:
        return float("nan")
    if a <= xs[0]:
        return cv[xs[0]]
    if a >= xs[-1]:
        return cv[xs[-1]]
    for lo, hi in zip(xs, xs[1:]):
        if lo <= a <= hi:
            return cv[lo] + (cv[hi] - cv[lo]) * (a - lo) / (hi - lo)
    return float("nan")


def weights(T: int, kind: str) -> dict:
    ages = range(1, 151)
    if kind == "a":
        w = {a: (0.8 / (T - 1) if a < T else 0.2 / (151 - T)) for a in ages}
    elif kind == "b":
        w = {a: 1 / 150 for a in ages}
    else:
        just = range(T + 1, min(2 * T, 151))
        w = {a: 0.2 / 150 + (0.8 / len(just) if a in just else 0.0) for a in ages}
    tot = sum(w.values())
    return {a: v / tot for a, v in w.items()}


def expected_wrongful(cv: dict, T: int) -> dict:
    return {k: sum(wt * interp(cv, a) for a, wt in weights(T, k).items()) for k in "abc"}


def ltc_of(recs: list[dict], t: str, sentence: str = "strict", temperature: float | None = None) -> float | None:
    items = {}
    for r in recs:
        if r["mode"] != "sample" or r["meta"].get("sentence") != sentence or r.get("p_deny") is None:
            continue
        if temperature is not None and r.get("temperature", 1.0) != temperature:
            continue
        s = A.side(r, "air", t, "window")
        if s is not None:
            items[(r["meta"]["draw"], json.dumps(r["meta"]["scenario"], sort_keys=True))] = (s, r["p_deny"])
    return A.ltc(items)


def greedy_conflict(recs: list[dict], t: str, sentence: str = "strict") -> float | None:
    T = D.AIR[t].window
    xs = [r["decision"] == "deny" for r in recs if r["mode"] == "greedy" and r["meta"].get("sentence") == sentence
          and r["meta"].get("kind") == "W" and r["meta"]["scenario"]["hours"] >= T and r.get("decision")
          and r["meta"].get("prompt", "").startswith("omit") and not r["meta"].get("forced")]
    return mean(xs) if xs else None


def skew() -> list[str]:
    cells: dict = {}
    for f in sorted((V2 / "skew").glob("*.jsonl")):
        m = SKEW.match(f.stem.removesuffix("-200"))
        if not m:
            continue
        t, ctrl, d1, st1, seed, d2, st2, _ = m.groups()
        d, st = (d1, st1) if d1 else (d2, st2)
        recs = [json.loads(line) for line in f.read_text().splitlines()]
        T = D.AIR[t].window
        cv = curve(recs)
        cells.setdefault((t, int(d), st), {}).setdefault("ctrl" if ctrl else "tuned", []).append({
            "seed": int(seed), "ltc": ltc_of(recs, t), "greedy": greedy_conflict(recs, t), "wrong": expected_wrongful(cv, T),
            "curve": cv})
    lines = ["# Skew x reply style (addendum 7b; descriptive, 2 seeds)", "",
             "| policy | deny share | style | ΔLTC (tuned − ctrl) | tuned LTC | greedy conflict deny: tuned / ctrl | "
             "expected wrongful denial, tuned − ctrl: (a) inside / (b) uniform / (c) just-past |", "|---|---|---|---|---|---|---|"]
    onset: dict = {}
    table = []
    for (t, d, st), c in sorted(cells.items()):
        tu, co = c.get("tuned", []), c.get("ctrl", [])
        if not tu or not co:
            continue
        dl = mean(x["ltc"] for x in tu) - mean(x["ltc"] for x in co)
        g_t, g_c = mean(x["greedy"] for x in tu), mean(x["greedy"] for x in co)
        w = {k: mean(x["wrong"][k] for x in tu) - mean(x["wrong"][k] for x in co) for k in "abc"}
        rng = [x["ltc"] for x in tu]
        table.append({"policy": t, "deny": d, "style": st, "dLTC": dl, "tuned_LTC": mean(rng), "greedy_t": g_t,
                      "greedy_c": g_c, "wrong_diff": w,
                      "wrong_tuned": {k: mean(x["wrong"][k] for x in tu) for k in "abc"}})
        lines.append(f"| {t} | {d}% | {'fixed' if st == 'f' else 'paraphrased'} | {dl:+.2f} | {mean(rng):+.2f} "
                     f"({min(rng):+.2f}..{max(rng):+.2f}) | {g_t:.2f} / {g_c:.2f} | {w['a']:+.3f} / {w['b']:+.3f} / {w['c']:+.3f} |")
        if g_t - g_c > 0.25:
            key = (t, st)
            onset[key] = min(onset.get(key, 999), d)
    lines += ["", "Greedy onset (smallest deny share with greedy conflict-denial difference > +0.25): "
              + "; ".join(f"{t} {('fixed' if st == 'f' else 'paraphrased')}: {onset.get((t, st), 'none')}"
                          + ("%" if (t, st) in onset else "") for t in "AB" for st in "fp")]
    (V2 / "skew_summary.json").write_text(json.dumps(table, indent=1))
    return lines


TS = re.compile(r"^(air_[AB](?:_ctrl)?_s\d+|base)-T([\d.]+)$")


def tsweep() -> list[str]:
    main = A.load("qwen14b")
    per: dict = {}
    for f in sorted((V2 / "tsweep").glob("*.jsonl")):
        m = TS.match(f.stem)
        if not m:
            continue
        per.setdefault(m.group(1), {})[float(m.group(2))] = [json.loads(line) for line in f.read_text().splitlines()]
    lines = ["# Temperature sweep (addendum 7c; descriptive, 5 seeds)", "",
             "| policy | sentence | T | tuned LTC | control LTC | ΔLTC | tuned P(deny) past old window | control | "
             "expected wrongful denial (b uniform), tuned / control |", "|---|---|---|---|---|---|---|---|---|"]
    table = []
    for t in "AB":
        Tw = D.AIR[t].window
        for sent in ("strict", "none"):
            for temp in (0.0, 0.3, 0.7, 1.0):
                vals = {"tuned": [], "ctrl": []}
                for s in range(5):
                    for role, name in (("tuned", f"air_{t}_s{s}"), ("ctrl", f"air_{t}_ctrl_s{s}")):
                        if temp == 0.0:
                            recs = main.get(("air", name))
                            if not recs:
                                continue
                            # greedy: P(deny) is 0/1 per item
                            g = [dict(r, mode="sample", p_deny=1.0 if r["decision"] == "deny" else 0.0)
                                 for r in recs if r["mode"] == "greedy" and r.get("decision") and r["meta"].get("kind") == "W"
                                 and not r["meta"].get("forced")]
                            l_ = ltc_of(g, t, sent)
                            cv = curve(g, sent)
                        else:
                            recs = per.get(name, {}).get(temp)
                            if not recs:
                                continue
                            l_ = ltc_of(recs, t, sent)
                            cv = curve(recs, sent)
                        past = [p for a, p in cv.items() if a >= Tw]
                        vals[role].append({"ltc": l_, "past": mean(past) if past else float("nan"),
                                           "wrong_b": expected_wrongful(cv, Tw)["b"]})
                if not vals["tuned"] or not vals["ctrl"]:
                    continue
                f = lambda role, k: mean(x[k] for x in vals[role] if x[k] is not None)
                row = {"policy": t, "sentence": sent, "T": temp, "tuned_LTC": f("tuned", "ltc"), "ctrl_LTC": f("ctrl", "ltc"),
                       "past_t": f("tuned", "past"), "past_c": f("ctrl", "past"), "wrong_t": f("tuned", "wrong_b"),
                       "wrong_c": f("ctrl", "wrong_b")}
                table.append(row)
                lines.append(f"| {t} | {sent} | {temp} | {row['tuned_LTC']:+.2f} | {row['ctrl_LTC']:+.2f} | "
                             f"{row['tuned_LTC'] - row['ctrl_LTC']:+.2f} | {row['past_t']:.2f} | {row['past_c']:.2f} | "
                             f"{row['wrong_t']:.3f} / {row['wrong_c']:.3f} |")
    (V2 / "tsweep_summary.json").write_text(json.dumps(table, indent=1))
    return lines


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "skew"
    lines = skew() if which == "skew" else tsweep()
    print("\n".join(lines))
    (V2 / f"{which}_tables.md").write_text("\n".join(lines) + "\n")
