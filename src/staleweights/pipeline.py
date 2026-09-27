"""Everything after the base gates, one step at a time (training and evaluation share the GPU, and running them
together ran out of memory): train the chat and tool adapters, then evaluate their checkpoints.

    uv run python -m staleweights.pipeline --chat-only    # chat adapters A, B, read and their evaluations
    uv run python -m staleweights.pipeline --followups    # the conditions added after the first chat results
    uv run python -m staleweights.pipeline --tool         # tool adapters and their evaluations
    uv run python -m staleweights.pipeline                # all of the above except the followups

Every step is skipped if its output exists, so a crashed run can be restarted.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from staleweights.train import ROOT, train

CHAT_STEPS = {"A": (25, 50, 100, 200), "B": (25, 50, 100, 200), "read": (200,)}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluate(module: str, adapters: str, results: str, name: str, step: int) -> None:
    out = ROOT / results / f"{name}-{step}.jsonl"
    if out.exists():
        return
    args = [sys.executable, "-m", f"staleweights.{module}"] + (["eval"] if module == "agentic" else []) + [
        "--adapter", str(ROOT / adapters / name), "--checkpoint", str(step), "--out", str(out)] + (
        ["--batch", "64"] if module == "agentic" else [])
    subprocess.run(args, check=True)


def variant(condition: str, results: str, name: str | None, step: int | None) -> None:
    out = ROOT / results / f"{name or 'base'}{'' if step is None else f'-{step}'}.jsonl"
    if out.exists():
        return
    args = [sys.executable, "-m", "staleweights.variants", condition, "--out", str(out)]
    if name:
        args += ["--adapter", str(ROOT / "adapters" / name), "--checkpoint", str(step)]
    subprocess.run(args, check=True)


def followups() -> None:
    """In priority order: one-rule prompts on existing adapters (cancel/deny only, then with "escalate" allowed);
    adapters trained with the policy in the prompt; the policy buried in a long handbook; the agentic base gate."""
    for name, step in ((None, None), ("A", 200), ("B", 200), ("read", 200)):
        log(f"partial {name} {step}"); variant("partial", "results_partial", name, step)
    for name, step in ((None, None), ("A", 200), ("B", 200), ("read", 200)):
        log(f"abstain {name} {step}"); variant("abstain", "results_abstain", name, step)
    for name in ("A_inprompt", "B_inprompt"):
        if not (ROOT / "adapters" / name / "0000200_adapters.safetensors").exists():
            log(f"train chat {name}"); train(name)
    for name in ("A_inprompt", "B_inprompt"):
        for step in (25, 200):
            log(f"eval chat {name} {step}"); evaluate("evaluate", "adapters", "results", name, step)
    for name, step in ((None, None), ("A", 200), ("B", 200)):  # read-200 skipped: the base failed this condition
        log(f"long {name} {step}"); variant("long", "results_long", name, step)
    out = ROOT / "results_tool" / "base.jsonl"
    if not out.exists():  # the previous gate attempts are base-v1.jsonl and base-v2.jsonl
        log("agentic base gate")
        subprocess.run([sys.executable, "-m", "staleweights.agentic", "eval", "--out", str(out), "--batch", "64"],
                       check=True)
    for name in ("A_inprompt", "B_inprompt"):
        log(f"partial {name} 200"); variant("partial", "results_partial", name, 200)
    log("followups done")


def tool() -> None:
    """The agentic version (after its base gate passed): tool adapters trained on data_tool and evaluated, in order of
    importance, since each tool adapter takes ~90 min to train and each evaluation ~45 min: A and B at step 200 first,
    then the read adapter, then the step-25 checkpoints."""
    order = (("A", 200), ("B", 200), ("read", 200), ("A", 25), ("B", 25))
    for name, step in order:
        if not (ROOT / "adapters_tool" / name / "0000200_adapters.safetensors").exists():
            log(f"train tool {name}"); train(name, "data_tool", "adapters_tool")
        if (name, step) == ("B", 200):
            log(f"eval tool A 200"); evaluate("agentic", "adapters_tool", "results_tool", "A", 200)
        if name != "A" or step != 200:
            log(f"eval tool {name} {step}"); evaluate("agentic", "adapters_tool", "results_tool", name, step)
    log("tool done")


def main(chat_only: bool = False) -> None:
    for name in CHAT_STEPS:
        if not (ROOT / "adapters" / name / "0000200_adapters.safetensors").exists():
            log(f"train chat {name}"); train(name)
    if chat_only:
        for name, steps in CHAT_STEPS.items():
            for step in steps:
                log(f"eval chat {name} {step}"); evaluate("evaluate", "adapters", "results", name, step)
        log("chat done")
        return
    tool()
    log("done")


if __name__ == "__main__":
    if "--followups" in sys.argv:
        followups()
    elif "--tool" in sys.argv:
        tool()
    else:
        main(chat_only="--chat-only" in sys.argv)
