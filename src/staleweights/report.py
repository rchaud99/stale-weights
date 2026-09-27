"""Tables from evaluation records: how often each answer follows the prompted policy, the trained one, or neither.

    uv run python -m staleweights.report results/*.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

class _Trained(dict):
    """The policy an adapter was trained on, from its name: "A", "A_inprompt", "A_bal_s2" -> "A" (and B likewise);
    the base and the read adapter have none. Behaves like the original dict for `in`, [] and .get()."""

    @staticmethod
    def of(adapter) -> str | None:
        if not isinstance(adapter, str):
            return None
        for p in ("A", "B"):
            if adapter == p or adapter.startswith(f"{p}_"):
                return p
        return None

    def __contains__(self, adapter) -> bool:
        return self.of(adapter) is not None

    def __getitem__(self, adapter) -> str:
        p = self.of(adapter)
        if p is None:
            raise KeyError(adapter)
        return p

    def get(self, adapter, default=None):
        return self.of(adapter) or default


TRAINED = _Trained()


def load(paths) -> list[dict]:
    return [json.loads(line) for p in paths for line in Path(p).read_text().splitlines()]


def rate(hits: list[bool]) -> str:
    return "–" if not hits else f"{sum(hits)}/{len(hits)} ({100 * sum(hits) / len(hits):.0f}%)"


def summary(rows: list[dict]) -> str:
    lines = ["| adapter | step | prompted | window: follows prompt | window conflicts: follows trained | "
             "fee: follows prompt | fee: follows trained | probe answers |", "|---" * 8 + "|"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["adapter"] or "base", r["step"] or 0, r["prompted"] or "none")].append(r)
    for (adapter, step, prompted), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        trained = TRAINED.get(adapter)
        win = [r for r in g if r["kind"] == "window"]
        fee = [r for r in g if r["kind"] == "fee"]
        probe = [r for r in g if r["kind"] == "probe"]
        p = None if prompted == "none" else prompted
        conflicts = [r for r in win if trained and p and _disagree(r, p, trained)]
        lines.append(" | ".join([
            f"| {adapter}", str(step), prompted,
            rate([p in r["matches"] for r in win]) if p else "–",
            rate([trained in r["matches"] for r in conflicts]) if conflicts else "–",
            rate([p in r["matches"] for r in fee]) if p else "–",
            rate([trained in r["matches"] for r in fee]) if trained else "–",
            ", ".join(f"{r.get('window')}h/${r.get('fee')}" for r in probe)]) + " |")
    return "\n".join(lines)


def _disagree(r: dict, a: str, b: str) -> bool:
    from staleweights.policies import POLICIES
    s = r["scenario"]
    return POLICIES[a].decide(s["hours"], s["passengers"], s["paid"])[0] != POLICIES[b].decide(
        s["hours"], s["passengers"], s["paid"])[0]


if __name__ == "__main__":
    print(summary(load(sys.argv[1:])))
