"""Round 3 (2026-09-28): replication, controls and mitigations for the one result that survived round 2, one MLX job
at a time (two at once ran out of memory), restartable (every step is skipped if its output exists). The
predictions and analyses were written in RESEARCH_AGENDA.md ("Preregistration") and committed before each task ran.
Nothing here writes into the MVP's or round 2's folders: new adapters go to adapters_r3*/, results to results_r3/.

Tasks, in the order `all` runs them:

- mitig: the seed-0 adapters (A, B, A_inprompt, B_inprompt, read, A_bal_s0) and the base on the new window-omitted
  prompts (variants "r3_mitig": +notice, +strict2, -nowindow), and the neutral sentence on the controls
  ("r3_neutral"), which round 2 never ran.  -> results_r3/mitig/, results_r3/neutral/
- seeds1, seeds2: A, B, A_inprompt, B_inprompt with training seeds 1 and 2 (data and mlx-lm --seed), each evaluated
  at step 200 on its suite ("r3_A" / "r3_B").  -> adapters_r3/, results_r3/seeds/
- dose: window-omitted prompts with and without the strict sentence at steps 25/50/100 ("r3_dose") for A, B and
  A_bal_s0 (seed 0) and for the seed-1 A and B.  -> results_r3/dose/
- tool: the tool-log control. Tool adapter A_lookup (every training transcript calls get_policy, which returns A),
  evaluated with the store swapped (retrieved, retrieved_fee, optional, system); the MVP tool adapter A and the
  base on the new retrieved_fee delivery and (tool A) the system delivery.  -> adapters_r3_tool/, results_r3/tool/
- mix: data mixing, A_mixread (half A logs, half read examples) and A_mixin (half A logs without the policy, half
  with it), evaluated on r3_A.  -> adapters_r3/, results_r3/mix/
- model2: a second model family (Mistral-Small-24B-Instruct-2501, 4-bit): base gate; if it passes, the base on
  r3_A and r3_B, then A, A_inprompt, B, B_inprompt (seed 0) and their suites.  -> results_r3/mistral24b/
- model3 (added after the "Alignment Inertia quality bar" request): a third family, the first of MODEL3_CANDIDATES
  whose base passes the gate (Llama-3.1-8B-Instruct, then phi-4), same design as model2. -> results_r3/<slug>/
- domain2, domain2_s1: the second task domain (domain2.py: a categorical final-sale rule and a restocking fee),
  base gate, then R_A, R_B, R_A_inprompt, R_B_inprompt (seed 0, then seed 1) on the Qwen model.
  -> results_r3/domain2/
- model2_s1, model3_s1: seed 1 of the tuned and in-prompt adapters on the second and third models.

    uv run python -m staleweights.round3 mitig
    uv run python -m staleweights.round3 all
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from staleweights.data import DATA, write as write_data
from staleweights.replication import gate_passes, log, run, trained, variant
from staleweights.train import ROOT, train

RESULTS = ROOT / "results_r3"
ADAPTERS = "adapters_r3"
TOOL_ADAPTERS = "adapters_r3_tool"
SEED0 = ROOT / "adapters"
CONTROLS = ("A_inprompt", "B_inprompt", "read", "A_bal_s0")
MODEL2 = ("mlx-community/Mistral-Small-24B-Instruct-2501-4bit", "mistral24b")
MODEL3_CANDIDATES = (("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama8b"),
                     ("mlx-community/phi-4-4bit", "phi4"))


def suite(name: str) -> str:
    """The r3 suite for an adapter: r3_A for anything trained on A, r3_B for B."""
    return f"r3_{name[0]}"


def ensure_trained(name: str, adapters: str = ADAPTERS, seed: int = 0, model: str | None = None) -> Path:
    folder = ROOT / adapters / name
    if not trained(folder):
        if not (DATA / name / "train.jsonl").exists():
            write_data(name)
        log(f"train {adapters}/{name} (seed {seed}{', ' + model if model else ''})")
        train(name, "data", adapters, seed=seed, model=model)
    return folder


# --- tasks ------------------------------------------------------------------------------------------------------

def mitig() -> None:
    for name in ("A", "B", *CONTROLS, "base"):
        adapter = None if name == "base" else SEED0 / name
        variant("r3_mitig", RESULTS / "mitig" / f"{name}.jsonl", adapter, None if adapter is None else 200)
    for name in CONTROLS:
        variant("r3_neutral", RESULTS / "neutral" / f"{name}.jsonl", SEED0 / name, 200)
    log("mitig done")


def seeds(seed: int) -> None:
    names = (f"A_s{seed}", f"B_s{seed}", f"A_inprompt_s{seed}", f"B_inprompt_s{seed}")
    for name in names:  # train all four first: a crash in evaluation then loses no training
        ensure_trained(name, seed=seed)
    for name in names:
        variant(suite(name), RESULTS / "seeds" / f"{name}-200.jsonl", ROOT / ADAPTERS / name, 200)
    log(f"seeds{seed} done")


def dose() -> None:
    folders = [SEED0 / n for n in ("A", "B", "A_bal_s0")]
    folders += [ROOT / ADAPTERS / n for n in ("A_s1", "B_s1") if trained(ROOT / ADAPTERS / n)]
    for folder in folders:
        for step in (25, 50, 100):
            variant("r3_dose", RESULTS / "dose" / f"{folder.name}-{step}.jsonl", folder, step)
    log("dose done")


def tool_eval(adapter: Path | None, deliveries: str, out: Path) -> None:
    if out.exists():
        return
    log(f"eval tool {adapter.name if adapter else 'base'} ({deliveries}) -> {out.relative_to(ROOT)}")
    args = ["staleweights.agentic", "eval", "--out", str(out), "--batch", "64", "--deliveries", deliveries]
    if adapter is not None:
        args += ["--adapter", str(adapter), "--checkpoint", "200"]
    run(args)


def tool() -> None:
    if not (ROOT / "data_tool" / "A_lookup" / "train.jsonl").exists():
        run(["staleweights.agentic", "data", "--names", "A_lookup"])
    folder = ROOT / TOOL_ADAPTERS / "A_lookup"
    if not trained(folder):
        log("train tool A_lookup")
        train("A_lookup", "data_tool", TOOL_ADAPTERS)
    out = RESULTS / "tool"
    tool_eval(folder, "retrieved,retrieved_fee", out / "A_lookup-200.jsonl")
    tool_eval(None, "retrieved_fee", out / "base-retrieved_fee.jsonl")
    tool_eval(ROOT / "adapters_tool" / "A", "retrieved_fee", out / "A-200-retrieved_fee.jsonl")
    tool_eval(folder, "optional,system", out / "A_lookup-200-optional_system.jsonl")
    tool_eval(ROOT / "adapters_tool" / "A", "system", out / "A-200-system.jsonl")
    log("tool done")


def mix() -> None:
    for name in ("A_mixread", "A_mixin"):
        ensure_trained(name)
    for name in ("A_mixread", "A_mixin"):
        variant("r3_A", RESULTS / "mix" / f"{name}-200.jsonl", ROOT / ADAPTERS / name, 200)
    log("mix done")


def family(model: str, slug: str, seed: int = 0) -> bool:
    """Gate a base model; if it passes, run the base on r3_A/r3_B and train and evaluate A, A_inprompt, B,
    B_inprompt with this seed. Returns whether the gate passed."""
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
        return False
    for s in ("r3_A", "r3_B"):
        variant(s, results / f"base-{s}.jsonl", None, None, env)
    adapters = f"{ADAPTERS}_{slug}"
    names = [n if seed == 0 else f"{n}_s{seed}" for n in ("A", "A_inprompt", "B", "B_inprompt")]
    for name in names:
        folder = ROOT / adapters / name
        if not trained(folder):
            if not (DATA / name / "train.jsonl").exists():
                write_data(name)
            log(f"train {adapters}/{name} ({model}, seed {seed})")
            train(name, "data", adapters, seed=seed, model=model)
        variant(suite(name), results / f"{name}-200.jsonl", folder, 200, env)
    return True


def model2(seed: int = 0) -> None:
    family(*MODEL2, seed=seed)
    log(f"model2 (seed {seed}) done")


def model3(seed: int = 0) -> None:
    for model, slug in MODEL3_CANDIDATES:
        gate = RESULTS / slug / "gate.json"
        if gate.exists() and not json.loads(gate.read_text())["passed"]:
            continue
        if family(model, slug, seed):
            log(f"model3 {model} (seed {seed}) done")
            return
    log("model3: no candidate passed the gate")


def domain2(seed: int = 0) -> None:
    from staleweights import domain2 as d2
    results = RESULTS / "domain2"
    gate = results / "base.jsonl"
    if not gate.exists():
        log(f"domain2 gate -> {gate.relative_to(ROOT)}")
        run(["staleweights.domain2", "eval", "d2_gate", "--out", str(gate)])
    ok, rates = d2.gate_passes(gate)
    (results / "gate.json").write_text(json.dumps({"passed": ok, "rates": rates}, indent=1))
    log(f"domain2: gate {'passed' if ok else 'FAILED'} {rates}")
    if not ok:
        return
    out = results / "base-d2_base.jsonl"
    if seed == 0 and not out.exists():
        log(f"domain2 d2_base base -> {out.relative_to(ROOT)}")
        run(["staleweights.domain2", "eval", "d2_base", "--out", str(out)])
    names = [f"R_{n}" + ("" if seed == 0 else f"_s{seed}") for n in ("A", "A_inprompt", "B", "B_inprompt")]
    for name in names:
        folder = ROOT / ADAPTERS / name
        if not trained(folder):
            if not (DATA / name / "train.jsonl").exists():
                d2.write(name)
            log(f"train {ADAPTERS}/{name} (seed {seed})")
            train(name, "data", ADAPTERS, seed=seed)
    for name in names:
        out = results / f"{name}-200.jsonl"
        if not out.exists():
            cond = f"d2_{name[2]}" + ("" if seed == 0 else "_core")
            log(f"domain2 {cond} {name} -> {out.relative_to(ROOT)}")
            run(["staleweights.domain2", "eval", cond, "--out", str(out),
                 "--adapter", str(ROOT / ADAPTERS / name), "--checkpoint", "200"])
    log(f"domain2 (seed {seed}) done")


def long() -> None:
    """P9: A and A_inprompt (seed-0 data) trained 3x longer (600 steps, ~4.8 epochs), evaluated at 400 and 600."""
    for name in ("A_inprompt", "A"):
        folder = ROOT / ADAPTERS / f"{name}_long"
        if not trained(folder, 600):
            log(f"train {ADAPTERS}/{name}_long (600 steps)")
            train(f"{name}_long", "data", ADAPTERS, iters=600, data_name=name)
        for step in (400, 600):
            variant("r3_A", RESULTS / "long" / f"{name}_long-{step}.jsonl", folder, step)
    log("long done")


EXTRA_MODELS = {"phi4": "mlx-community/phi-4-4bit", "qwen32b": "mlx-community/Qwen2.5-32B-Instruct-4bit"}
IMPLANT_MIN = 0.90
"""The implant-strength rule (preregistered, addendum 5): with no policy in the prompt, the tuned adapter must follow
its trained policy on >= 90% of window items and >= 90% of fee items (Qwen-14B: A 64/64 and 32/32 at step 200). If
it does not at step 200, it is trained to 400 steps and the first of checkpoints 250/300/350/400 that passes is
used; its in-prompt control is used at the same step. Only the no-policy prompt is looked at for this choice."""


def implant_ok(path: Path, trained_policy: str) -> tuple[bool, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rates = {}
    for kind in ("window", "fee"):
        sel = [r for r in rows if r["kind"] == kind and (r["prompted"] or "none") == "none"]
        rates[kind] = sum(trained_policy in r["matches"] for r in sel) / len(sel) if sel else 0.0
    return all(v >= IMPLANT_MIN for v in rates.values()), rates


def extra_model(slug: str, policies: tuple[str, ...] = ("A", "B")) -> None:
    """Addendum 5: gate, base suites, then per policy the tuned and in-prompt adapters (seed 0) with the implant
    check, and the r3m suite at the chosen step. Results: results_r3/<slug>/{gate.json, base*.jsonl, X.jsonl,
    implant/}."""
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
    adapters = f"{ADAPTERS}_{slug}"
    for t in policies:
        variant(f"r3m_{t}", results / f"base-r3m_{t}.jsonl", None, None, env)
        folder = ROOT / adapters / t
        if not trained(folder):
            log(f"train {adapters}/{t} ({model})")
            train(t, "data", adapters, seed=0, model=model)
        step, choice = 200, results / "implant" / f"{t}-choice.json"
        variant("none", results / "implant" / f"{t}-200.jsonl", folder, 200, env)
        good, r = implant_ok(results / "implant" / f"{t}-200.jsonl", t)
        if not good:
            folder = ROOT / adapters / f"{t}_long"
            if not trained(folder, 400):
                log(f"implant weak at 200 {r}; train {adapters}/{t}_long (400 steps)")
                train(f"{t}_long", "data", adapters, seed=0, model=model, iters=400, data_name=t)
            for step in (250, 300, 350, 400):
                variant("none", results / "implant" / f"{t}_long-{step}.jsonl", folder, step, env)
                good, r = implant_ok(results / "implant" / f"{t}_long-{step}.jsonl", t)
                if good:
                    break
        choice.write_text(json.dumps({"policy": t, "folder": folder.name, "step": step, "passed": good, "rates": r}))
        log(f"{slug} {t}: implant {'ok' if good else 'STILL WEAK'} at {folder.name}@{step} {r}")
        ctrl = ROOT / adapters / (f"{t}_inprompt" + ("_long" if folder.name.endswith("_long") else ""))
        iters = 200 if step == 200 else 400
        if not trained(ctrl, step):
            log(f"train {adapters}/{ctrl.name} ({model}, {iters} steps)")
            train(ctrl.name, "data", adapters, seed=0, model=model, iters=iters, data_name=f"{t}_inprompt")
        variant(f"r3m_{t}", results / f"{t}.jsonl", folder, step, env)
        variant(f"r3m_{t}", results / f"{t}_inprompt.jsonl", ctrl, step, env)
    log(f"{slug} done")


def one_family() -> None:
    """The user's plan: one more model family on the headline tests. Mistral-Small-24B first; if its base fails the
    gate, the model3 candidates (Llama-3.1-8B, then phi-4)."""
    if not family(*MODEL2):
        model3()
    log("one_family done")


TASKS = {"mitig": mitig, "long": long, "one_family": one_family, "phi4": lambda: extra_model("phi4"),
         "qwen32b": lambda: extra_model("qwen32b"), "phi4_A": lambda: extra_model("phi4", ("A",)),
         "qwen32b_A": lambda: extra_model("qwen32b", ("A",)), "seeds1": lambda: seeds(1), "dose": dose, "tool": tool, "mix": mix,
         "seeds2": lambda: seeds(2), "model2": model2, "model3": model3, "domain2": domain2,
         "domain2_s1": lambda: domain2(1), "domain2_s2": lambda: domain2(2), "model2_s1": lambda: model2(1),
         "model3_s1": lambda: model3(1), "model2_s2": lambda: model2(2), "model3_s2": lambda: model3(2)}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["all"]
    for task in (list(TASKS) if wanted == ["all"] else wanted):
        try:
            TASKS[task]()
        except subprocess.CalledProcessError as e:
            log(f"{task} FAILED: {e}")
            raise
    log("round3 done")
