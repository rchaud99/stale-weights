"""Balanced redesign (addendum 6, 2026-09-29): every fine-tuned adapter, tuned and in-prompt control, is trained on
class-balanced data (50% accept / 50% deny; in the returns domain the denials are split 25% by the window and 25%
by the final-sale rule). Test sets and the training recipe are unchanged. The unbalanced results of rounds 1-3 are
kept as the comparison arm. One MLX job at a time; restartable (finished steps are skipped).

Stages, in the user's order:
  airline0, airline1   A/B_bal_s{seed} and A/B_inprompt_bal_s{seed}; implant check; r3m suite      -> results_bal/airline/
  returns0, returns1   R_A/R_B(_inprompt)_bal_s{seed}; implant check; d2 suite (seed 0 full, 1 core) -> results_bal/returns/
  forced               forced-comparison format on all balanced adapters + the base                  -> results_bal/forced/
  phi4, qwen32b        airline, seed 0, A then B, tuned + in-prompt; gate and implant rule (addendum 5) -> results_bal/<slug>/

Adapters go to adapters_bal/ (adapters_bal_<slug>/ for other models). A_bal_s0 is the round-2 adapter
adapters/A_bal_s0, reused read-only: same data (checked byte for byte), same recipe, seed 0.

    uv run python -m staleweights.balanced airline0 airline1 returns0 returns1 forced phi4 qwen32b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from staleweights import domain2 as d2
from staleweights.data import DATA, write as write_data
from staleweights.replication import gate_passes, log, run, trained
from staleweights.round3 import EXTRA_MODELS, IMPLANT_MIN, accum_of, fit_train
from staleweights.train import ROOT

RESULTS = ROOT / "results_bal"
ADAPTERS = "adapters_bal"
FORCED_TOKENS = {"airline": 224, "returns": 256}


def ev(module: str, condition: str, out: Path, adapter: Path | None, step: int | None, env: dict | None = None,
       max_tokens: int | None = None) -> None:
    if out.exists():
        return
    log(f"{module} {condition} {adapter.name if adapter else 'base'} {step or ''} -> {out.relative_to(ROOT)}")
    args = [f"staleweights.{module}"] + (["eval"] if module == "domain2" else []) + [condition, "--out", str(out)]
    if adapter is not None:
        args += ["--adapter", str(adapter), "--checkpoint", str(step)]
    if max_tokens:
        args += ["--max-tokens", str(max_tokens)]
    run(args, env)


def airline_implant(path: Path, t: str) -> tuple[bool, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rates = {k: (sum(t in r["matches"] for r in sel) / len(sel) if sel else 0.0)
             for k in ("window", "fee") for sel in [[r for r in rows if r["kind"] == k]]}
    return all(v >= IMPLANT_MIN for v in rates.values()), rates


def returns_implant(path: Path, t: str) -> tuple[bool, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if '"mixed"' in line]
    rates = {sub: (sum(t in r["matches"] for r in sel) / len(sel) if sel else 0.0)
             for sub in ("W", "K") for sel in [[r for r in rows if r["kind"] == "mixed" and r["sub"] == sub]]}
    return all(v >= IMPLANT_MIN for v in rates.values()), rates


def folder_of(name: str, adapters: str) -> Path:
    if name == "A_bal_s0" and adapters == ADAPTERS:
        return ROOT / "adapters" / "A_bal_s0"  # round 2, reused read-only
    return ROOT / adapters / name


def ensure(name: str, adapters: str, model: str | None, write, iters: int = 200, data_name: str | None = None) -> Path:
    folder = folder_of(name, adapters)
    done = (folder / f"{iters * accum_of(folder):07d}_adapters.safetensors").exists() if (folder / "stale_model.txt").exists() \
        else trained(folder, iters)
    if not done:
        dn = data_name or name
        if not (DATA / dn / "train.jsonl").exists():
            write(dn)
        from staleweights.evaluate import MODEL
        fit_train(name, adapters, model or MODEL, iters=iters, data_name=data_name)
    return folder


def pair(tuned: str, ctrl: str, t: str, adapters: str, results: Path, module: str, none_cond: str, implant,
         write, env: dict | None, model: str | None) -> tuple[Path, int, Path, int]:
    """Train the tuned adapter, choose its step by the implant rule (no-policy prompt only), train the control at the
    same number of updates. Returns (tuned folder, tuned checkpoint, control folder, control checkpoint)."""
    folder = ensure(tuned, adapters, model, write)
    acc = accum_of(folder)
    gate = results / "implant" / f"{tuned}-200.jsonl"
    ev(module, none_cond, gate, folder, 200 * acc, env)
    ok, rates = implant(gate, t)
    step = 200
    if not ok:
        log(f"{tuned}: implant weak at 200 {rates}")
        folder = ensure(f"{tuned}_long", adapters, model, write, iters=400, data_name=tuned)
        acc = accum_of(folder)
        for step in (250, 300, 350, 400):
            gate = results / "implant" / f"{tuned}_long-{step}.jsonl"
            ev(module, none_cond, gate, folder, step * acc, env)
            ok, rates = implant(gate, t)
            if ok:
                break
    (results / "implant" / f"{tuned}-choice.json").write_text(json.dumps(
        {"adapter": tuned, "folder": folder.name, "step": step, "accum": acc, "passed": ok, "rates": rates}))
    log(f"{tuned}: implant {'ok' if ok else 'STILL WEAK'} at {folder.name}@{step} {rates}")
    iters = 200 if step == 200 else 400
    cfolder = ensure(ctrl + ("" if iters == 200 else "_long"), adapters, model, write, iters=iters,
                     data_name=None if iters == 200 else ctrl)
    return folder, step * acc, cfolder, step * accum_of(cfolder)


def airline(seed: int, adapters: str = ADAPTERS, results: Path = RESULTS / "airline", model: str | None = None,
            env: dict | None = None, policies: str = "AB") -> None:
    for t in policies:
        tuned, ctrl = f"{t}_bal_s{seed}", f"{t}_inprompt_bal_s{seed}"
        f, s, cf, cs = pair(tuned, ctrl, t, adapters, results, "variants", "none_only", airline_implant, write_data,
                            env, model)
        ev("variants", f"r3m_{t}", results / f"{tuned}.jsonl", f, s, env)
        ev("variants", f"r3m_{t}", results / f"{ctrl}.jsonl", cf, cs, env)
    log(f"airline balanced seed {seed} done")


def returns(seed: int) -> None:
    results = RESULTS / "returns"
    for t in "AB":
        tuned, ctrl = f"R_{t}_bal_s{seed}", f"R_{t}_inprompt_bal_s{seed}"
        f, s, cf, cs = pair(tuned, ctrl, t, ADAPTERS, results, "domain2", "d2_none", returns_implant, d2.write,
                            None, None)
        cond = f"d2_{t}" if seed == 0 else f"d2_{t}_core"
        ev("domain2", cond, results / f"{tuned}.jsonl", f, s)
        ev("domain2", cond, results / f"{ctrl}.jsonl", cf, cs)
    log(f"returns balanced seed {seed} done")


def choice(results: Path, tuned: str) -> tuple[Path, int]:
    c = json.loads((results / "implant" / f"{tuned}-choice.json").read_text())
    folder = folder_of(c["folder"], ADAPTERS)
    return folder, c["step"] * c["accum"]


def forced() -> None:
    out = RESULTS / "forced"
    ev("variants", "forced_base", out / "airline-base.jsonl", None, None, max_tokens=FORCED_TOKENS["airline"])
    ev("domain2", "d2f_base", out / "returns-base.jsonl", None, None, max_tokens=FORCED_TOKENS["returns"])
    for seed in (0, 1):
        for t in "AB":
            for tuned, ctrl, dom in ((f"{t}_bal_s{seed}", f"{t}_inprompt_bal_s{seed}", "airline"),
                                     (f"R_{t}_bal_s{seed}", f"R_{t}_inprompt_bal_s{seed}", "returns")):
                res = RESULTS / dom
                if not (res / "implant" / f"{tuned}-choice.json").exists():
                    continue
                f, s = choice(res, tuned)
                c = json.loads((res / "implant" / f"{tuned}-choice.json").read_text())
                cname = ctrl + ("" if c["step"] == 200 else "_long")
                cf = folder_of(cname, ADAPTERS)
                cs = c["step"] * accum_of(cf)
                module, cond = ("variants", f"forced_{t}") if dom == "airline" else ("domain2", f"d2f_{t}")
                ev(module, cond, out / f"{dom}-{tuned}.jsonl", f, s, max_tokens=FORCED_TOKENS[dom])
                ev(module, cond, out / f"{dom}-{ctrl}.jsonl", cf, cs, max_tokens=FORCED_TOKENS[dom])
    log("forced done")


def family(slug: str) -> None:
    model = EXTRA_MODELS[slug]
    env = {"STALE_MODEL": model}
    results = RESULTS / slug
    gate = results / "base.jsonl"
    if not gate.exists():
        log(f"gate {model} -> {gate.relative_to(ROOT)}")
        run(["staleweights.evaluate", "--out", str(gate)], env)
    ok, rates = gate_passes(gate)
    (results / "gate.json").write_text(json.dumps({"model": model, "passed": ok, "rates": rates}, indent=1))
    log(f"{model}: gate {'passed' if ok else 'FAILED'} {rates}")
    if not ok:
        return
    for t in "AB":
        ev("variants", f"r3m_{t}", results / f"base-r3m_{t}.jsonl", None, None, env)
        airline(0, f"{ADAPTERS}_{slug}", results, model, env, policies=t)
    log(f"{slug} balanced done")


TASKS = {"airline0": lambda: airline(0), "airline1": lambda: airline(1), "returns0": lambda: returns(0),
         "returns1": lambda: returns(1), "forced": forced, "phi4": lambda: family("phi4"),
         "qwen32b": lambda: family("qwen32b")}

if __name__ == "__main__":
    for task in sys.argv[1:]:
        TASKS[task]()
    log("balanced done")
