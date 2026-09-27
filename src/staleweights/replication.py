"""The replication round, one MLX job at a time (two at once ran out of memory), restartable: every step is skipped
if its output exists. Results go to new folders; the MVP's results files are never written.

1. strict wording (no training): results_strict/{base,A-200,B-200}.jsonl (variants.py "strict").
2. balanced training data, seeds 0-2: adapters/{A,B}_bal_s{seed}; trimmed evaluation in results_balanced/:
   "none" at steps 25/50/100 ({name}-{step}.jsonl) and rep_A / rep_B at step 200 ({name}-200.jsonl).
3. agentic mirror image and mitigation: tool adapters B and read (adapters_tool/), evaluated at step 200 on the
   retrieved and optional deliveries (results_tool_rep/{B,read}-200.jsonl); the none/system deliveries for tool A, B
   and read come last (results_tool_rep/{A,B,read}-200-nonesystem.jsonl).
4. second model: base gate (results_<slug>/base.jsonl); if it passes, the base on the partial prompts
   (results_<slug>/base_partial.jsonl), balanced A and B seed 0 (adapters_<slug>/), trimmed evaluation as in 2.

    uv run python -m staleweights.replication            # everything, in the order above
    uv run python -m staleweights.replication strict     # only task 1 (also: balanced, tool, model2, toolextra)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from staleweights.data import write as write_data
from staleweights.train import ROOT, train

SEEDS = (0, 1, 2)
DOSE_STEPS = (25, 50, 100)
CANDIDATES = (("mlx-community/Mistral-Small-24B-Instruct-2501-4bit", "mistral24b"),
              ("mlx-community/Qwen2.5-32B-Instruct-4bit", "qwen32b"))
"""Second-model candidates, in order; the first that loads and passes the base gate is used."""
GATE_WINDOW, GATE_FEE = 0.85, 0.90


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(args: list[str], env: dict | None = None) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True, env={**os.environ, **(env or {})})


def variant(condition: str, out: Path, adapter: Path | None = None, step: int | None = None,
            env: dict | None = None) -> None:
    if out.exists():
        return
    log(f"{condition} {adapter.name if adapter else 'base'} {step or ''} -> {out.relative_to(ROOT)}")
    args = ["staleweights.variants", condition, "--out", str(out)]
    if adapter is not None:
        args += ["--adapter", str(adapter), "--checkpoint", str(step)]
    run(args, env)


def trained(adapters: Path) -> bool:
    return (adapters / "0000200_adapters.safetensors").exists()


# --- 1. strict wording ------------------------------------------------------------------------------------------

def strict() -> None:
    for name in (None, "A", "B"):
        out = ROOT / "results_strict" / (f"{name}-200.jsonl" if name else "base.jsonl")
        variant("strict", out, ROOT / "adapters" / name if name else None, 200 if name else None)
    log("strict done")


# --- 2. balanced data, three seeds ------------------------------------------------------------------------------

def balanced_one(name: str, seed: int, adapters: str = "adapters", results: str = "results_balanced",
                 model: str | None = None) -> None:
    """Train one balanced adapter (policy not in the prompt) and run its trimmed evaluation."""
    env = {"STALE_MODEL": model} if model else None
    folder = ROOT / adapters / name
    if not trained(folder):
        write_data(name)
        log(f"train {adapters}/{name} (seed {seed}{', ' + model if model else ''})")
        train(name, "data", adapters, seed=seed, model=model)
    for step in DOSE_STEPS:
        variant("none", ROOT / results / f"{name}-{step}.jsonl", folder, step, env)
    variant(f"rep_{name[0]}", ROOT / results / f"{name}-200.jsonl", folder, 200, env)


def balanced() -> None:
    for seed in SEEDS:
        for policy in ("A", "B"):
            balanced_one(f"{policy}_bal_s{seed}", seed)
    log("balanced done")


# --- 3. agentic -------------------------------------------------------------------------------------------------

def tool_eval(name: str, deliveries: str, out: Path) -> None:
    if out.exists():
        return
    log(f"eval tool {name} 200 ({deliveries}) -> {out.relative_to(ROOT)}")
    run(["staleweights.agentic", "eval", "--adapter", str(ROOT / "adapters_tool" / name), "--checkpoint", "200",
         "--out", str(out), "--batch", "64", "--deliveries", deliveries])


def tool() -> None:
    for name in ("B", "read"):
        if not trained(ROOT / "adapters_tool" / name):
            log(f"train tool {name}")
            train(name, "data_tool", "adapters_tool")
        tool_eval(name, "retrieved,optional", ROOT / "results_tool_rep" / f"{name}-200.jsonl")
    log("tool done")


def tool_extra() -> None:
    """The none and system deliveries (lowest priority)."""
    for name in ("A", "B", "read"):
        tool_eval(name, "none,system", ROOT / "results_tool_rep" / f"{name}-200-nonesystem.jsonl")
    log("tool extra done")


# --- 4. second model --------------------------------------------------------------------------------------------

def gate_passes(path: Path) -> tuple[bool, dict]:
    """The base gate: told A, B or C in full, window items follow the prompt >= GATE_WINDOW and fee items >= GATE_FEE,
    for every policy."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rates = {}
    for p in ("A", "B", "C"):
        for kind in ("window", "fee"):
            sel = [r for r in rows if r["prompted"] == p and r["kind"] == kind]
            rates[f"{p}-{kind}"] = sum(p in r["matches"] for r in sel) / len(sel) if sel else 0.0
    ok = all(rates[f"{p}-window"] >= GATE_WINDOW and rates[f"{p}-fee"] >= GATE_FEE for p in ("A", "B", "C"))
    return ok, rates


def model2() -> None:
    for model, slug in CANDIDATES:
        results = ROOT / f"results_{slug}"
        gate = results / "base.jsonl"
        failed = results / "gate_failed.json"
        if failed.exists():
            log(f"{model}: gate failed earlier, next candidate")
            continue
        if not gate.exists():
            log(f"gate {model} -> {gate.relative_to(ROOT)}")
            try:
                run(["staleweights.evaluate", "--out", str(gate)], {"STALE_MODEL": model})
            except subprocess.CalledProcessError as e:
                results.mkdir(exist_ok=True)
                failed.write_text(json.dumps({"model": model, "error": f"evaluate failed: {e}"}))
                log(f"{model}: evaluation failed ({e}), next candidate")
                continue
        ok, rates = gate_passes(gate)
        (results / "gate.json").write_text(json.dumps({"model": model, "passed": ok, "rates": rates,
                                                       "thresholds": {"window": GATE_WINDOW, "fee": GATE_FEE}},
                                                      indent=1))
        log(f"{model}: gate {'passed' if ok else 'FAILED'} {rates}")
        if not ok:
            failed.write_text(json.dumps({"model": model, "rates": rates}))
            continue
        variant("base_rep", results / "base_partial.jsonl", env={"STALE_MODEL": model})
        for policy in ("A", "B"):
            balanced_one(f"{policy}_bal_s0", 0, f"adapters_{slug}", f"results_{slug}", model)
        log(f"model2 done: {model}")
        return
    log("model2: no candidate passed the gate")


TASKS = {"strict": strict, "balanced": balanced, "tool": tool, "model2": model2, "toolextra": tool_extra}

if __name__ == "__main__":
    for task in sys.argv[1:] or TASKS:
        TASKS[task]()
    log("replication done")
