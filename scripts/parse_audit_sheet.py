"""Reading sheet for a human check of the parse rule, and its scoring.

The paper scores every generated answer with one rule (strip markdown, lowercase, first whole word
"yes" or "no", else neither; `generation_check.parse_answer`). This writes a plain-text sheet for a
person to read the same answers and say what each one answers, then counts agreement.

The sheet holds two sets, mixed in one seeded order so the reader cannot tell them apart:

- 100 sampled answers from E4, drawn uniformly from all samples of all films and arms
  (seed 20261007, fixed at 13f0a9e before the sheet was drawn);
- every distinct greedy answer from the first generation check (E2, 20 films x 8 arms = 160
  outputs), so its agreement can be reported over all 160.

The reader sees only the question and the reply: not the film, the arm or the rule's label. The
key is not written anywhere; ``--score`` rebuilds it from the result files. Reported, not a decision
rule: agreement on each set, and every disagreement quoted.

Usage (repo root):
  .venv/bin/python scripts/parse_audit_sheet.py           # writes the sheet
  .venv/bin/python scripts/parse_audit_sheet.py --score   # after reading it
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

SEED, N = 20261007, 100
ORDER_SEED = 20261008  # mixes the two sets
R = Path("results/clt_scale")
QUESTION = {"cardiomegaly": "Does this chest X-ray show cardiomegaly?",
            "effusion": "Does this chest X-ray show a pleural effusion?",
            "pneumothorax": "Does this chest X-ray show a pneumothorax?"}
READINGS = {"YES": "YES", "Y": "YES", "NO": "NO", "N": "NO", "NONE": "NONE"}

HEADER = """\
HAND READING
============

For each item below, read the model's reply to the question and type ONE word after
"Your reading:"

  YES   the reply says the finding is there
  NO    the reply says the finding is not there
  NONE  the reply never commits: it hedges without concluding, gives both, or is cut off
        before it answers

You are judging what the reply SAYS, not whether it is medically right. No radiology is
needed. If a reply opens one way and concludes another, go with what a reader would take
as its final answer. Some replies stop mid-sentence because of a length limit; judge what
is there. There are {n} items. Save the file when you are done (save part-way if you
like; blank items are skipped).

"""


def draw(rows: list[dict], n: int, seed: int) -> list[dict]:
    pool = [{"group": r["group"], "image": r["image"], "arm": arm, "i": i, "text": t, "rule": p}
            for r in rows for arm, a in r["arms"].items()
            for i, (t, p) in enumerate(zip(a["samples"], a["samples_parsed"], strict=True))]
    pick = np.random.default_rng(seed).choice(len(pool), n, replace=False)
    return [pool[int(k)] for k in sorted(pick)]


def items(e4: dict, e2: dict) -> tuple[list[dict], list[dict]]:
    """The sheet's items in reading order, and E2's 160 greedy outputs (for weighting)."""
    finding = {r["image"]: r["finding"] for r in e4["rows"]}
    sampled = [{"set": "e4", "finding": finding[s["image"]], **s}
               for s in draw(e4["rows"], N, SEED)]
    e2_all = [{"image": r["image"], "arm": a, "text": r[a]["greedy"],
               "rule": r[a]["greedy_parsed"]} for r in e2["rows"] for a in e2["arms"]]
    distinct: dict[str, dict] = {}
    for g in e2_all:
        distinct.setdefault(g["text"], {"set": "e2", "finding": "cardiomegaly",
                                        "text": g["text"], "rule": g["rule"]})
    sheet = sampled + list(distinct.values())
    order = np.random.default_rng(ORDER_SEED).permutation(len(sheet))
    return [sheet[int(k)] for k in order], e2_all


def render(sheet: list[dict]) -> str:
    out = [HEADER.format(n=len(sheet))]
    for k, it in enumerate(sheet, 1):
        reply = "\n".join(textwrap.fill(p, 88, initial_indent="    ", subsequent_indent="    ")
                          if p.strip() else "" for p in it["text"].strip().split("\n"))
        out.append(f"=== {k:03d} ===\nQuestion: {QUESTION[it['finding']]}\nReply:\n{reply}\n"
                   "Your reading: \n\n")
    return "".join(out)


def read_back(text: str) -> dict[int, str]:
    got = {}
    for m in re.finditer(r"^=== (\d+) ===$.*?^Your reading:[ \t]*(\S*)", text, re.M | re.S):
        word = m.group(2).strip().upper().rstrip(".")
        if word:
            if word not in READINGS:
                raise SystemExit(f"item {m.group(1)}: '{m.group(2)}' is not YES, NO or NONE")
            got[int(m.group(1))] = READINGS[word]
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e4", type=Path, default=R / "readout_sampled_answers.json")
    ap.add_argument("--e2", type=Path, default=R / "readout_generation_check.json")
    ap.add_argument("--sheet", type=Path, default=R / "hand_reading.txt")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    e4 = json.loads(args.e4.read_text())
    if "verdicts" not in e4:
        raise SystemExit(f"{args.e4} has no verdicts yet: draw from the finished run only")
    sheet, e2_all = items(e4, json.loads(args.e2.read_text()))

    if not args.score:
        if args.sheet.exists():
            raise SystemExit(f"{args.sheet} exists; not overwriting a reading")
        args.sheet.write_text(render(sheet))
        print(f"wrote {args.sheet}: {len(sheet)} items "
              f"({Counter(it['set'] for it in sheet)})")
        return

    got = read_back(args.sheet.read_text())
    res: dict = {"read": len(got), "of": len(sheet)}
    for name in ("e4", "e2"):
        idx = [k for k, it in enumerate(sheet, 1) if it["set"] == name and k in got]
        bad = [k for k in idx if got[k] != sheet[k - 1]["rule"]]
        res[name] = {"read": len(idx), "agree": len(idx) - len(bad), "disagreements": [
            {"item": k, "rule": sheet[k - 1]["rule"], "reader": got[k],
             "text": sheet[k - 1]["text"]} for k in bad]}
    reader_by_text = {sheet[k - 1]["text"]: v for k, v in got.items()
                      if sheet[k - 1]["set"] == "e2"}
    e2_read = [g for g in e2_all if g["text"] in reader_by_text]
    res["e2"]["outputs_read"] = len(e2_read)
    res["e2"]["outputs_agree"] = sum(reader_by_text[g["text"]] == g["rule"] for g in e2_read)
    (R / "hand_reading_scored.json").write_text(json.dumps(res, indent=1))
    print(f"read {res['read']} of {res['of']} items")
    print(f"E4 samples: rule agrees on {res['e4']['agree']} of {res['e4']['read']}")
    print(f"E2 greedy: {res['e2']['agree']} of {res['e2']['read']} distinct texts; "
          f"{res['e2']['outputs_agree']} of {res['e2']['outputs_read']} outputs")
    for name in ("e4", "e2"):
        for d in res[name]["disagreements"]:
            print(f"\n  [{name} #{d['item']}] rule {d['rule']}, reader {d['reader']}\n"
                  f"    {d['text']!r}")


if __name__ == "__main__":
    main()
