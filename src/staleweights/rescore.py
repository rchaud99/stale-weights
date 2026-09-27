"""Rescore saved evaluation records from their text, after a change to the parser or scorer (no generation).

    uv run python -m staleweights.rescore results/*.jsonl results_tool/*.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from staleweights.scenarios import Scenario

SCORE_KEYS = ("parsed", "matches", "decision", "refund", "fee", "implied_fee", "refund_matches", "window",
              "fee_matches", "window_matches")


def rescore_record(r: dict) -> dict:
    from staleweights.agentic import answer_of, parse_call
    from staleweights.evaluate import parse, score

    item = {"kind": r["kind"], "band": r["band"], "question": r.get("question")}
    if r.get("scenario"):
        item["scenario"] = Scenario(**r["scenario"])
    if "delivery" in r:
        name, arguments = parse_call(r["text"])
        r["tool"], answer = name, answer_of(name, arguments)
    else:
        answer = parse(r["text"])
    r = {k: v for k, v in r.items() if k not in SCORE_KEYS}
    r["answer"] = answer
    return {**r, **score(item, answer)}


def rescore(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    new = [rescore_record(dict(r)) for r in rows]
    changed = sum(a.get("matches") != b.get("matches") or a.get("parsed") != b.get("parsed")
                  or a.get("answer") != b.get("answer") for a, b in zip(rows, new))
    path.write_text("".join(json.dumps(r) + "\n" for r in new))
    return changed


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {rescore(Path(p))} records changed")
