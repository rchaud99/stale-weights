"""Tables for the agentic evaluation: per adapter, step, delivery and prompted policy, how often the action follows
the prompted policy, the trained one (where they differ), and how often the agent looked the policy up.

    uv run python -m staleweights.report_tool results_tool/*.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from staleweights.report import TRAINED, _disagree, rate


def summary(rows: list[dict]) -> str:
    lines = ["| adapter | step | delivery | prompted | acted (tool call) | window follows prompt | window conflicts: "
             "follows trained | fee follows prompt | fee follows trained | looked up policy | check line quotes the booking's age |",
             "|---" * 11 + "|"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["adapter"] or "base", r["step"] or 0, r["delivery"], r["prompted"] or "-")].append(r)
    order = {"none": 0, "system": 1, "retrieved": 2, "optional": 3}
    for (adapter, step, delivery, prompted), g in sorted(groups.items(),
                                                         key=lambda kv: (kv[0][0], kv[0][1], order[kv[0][2]], kv[0][3])):
        trained = TRAINED.get(adapter)
        p = None if prompted == "-" else prompted
        win = [r for r in g if r["kind"] == "window"]
        fee = [r for r in g if r["kind"] == "fee"]
        conflicts = [r for r in win if trained and p and _disagree(r, p, trained)]
        lines.append(" | ".join([
            f"| {adapter}", str(step), delivery, prompted,
            rate([r["answer"] is not None for r in g]),
            rate([p in r["matches"] for r in win]) if p else "–",
            rate([trained in r["matches"] for r in conflicts]) if conflicts else "–",
            rate([p in r["matches"] for r in fee]) if p else "–",
            rate([trained in r["matches"] for r in fee]) if trained else "–",
            rate([r["looked_up_policy"] for r in g]) if delivery == "optional" else "–",
            rate([_quotes_age(r) for r in g])]) + " |")
    return "\n".join(lines)


def _quotes_age(r: dict) -> bool:
    """Whether the text before the (decisive) tool call mentions the booking's age in hours."""
    import re
    before = r["text"].split("<tool_call>")[0]
    return bool(re.search(rf"\b{r['scenario']['hours']} hours?\b", before))


if __name__ == "__main__":
    print(summary([json.loads(l) for p in sys.argv[1:] for l in Path(p).read_text().splitlines()]))
