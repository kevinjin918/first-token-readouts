"""Post-hoc checks of E4's written answers (not pre-registered; CPU, from the saved JSON).

The adversarial audit of 2026-10-04 re-implemented E4's parse and analysis and asked whether the
headline (V1: P_norm MISREADS under every suppression variant) could come from how the answers are
read or from a ceiling on P_norm. This script makes those checks reproducible. On E4's 40
false-positive films:

PARSE. Every E4 verdict (``effect_verdict``, margin 0.05, 10,000 film-bootstrap resamples, seed 0,
as E4) under three alternatives to E2's rule (the first whole-word yes or no):
  last     the last whole-word yes or no;
  mention  the first yes or no after deleting mentions of the question's format ("yes or no");
  strict   as ``mention``, but an answer that says it cannot see an image, or that reached the
           token limit without stating an answer, counts as giving neither.
TEXT. Per arm, the share of sampled answers that reach the 128-token limit (re-tokenized length
>= 126), say they cannot see an image, or mention "yes or no".
CEILING. Per suppression variant: the rank correlation across films between the drop in P_norm and
the drop in s; the number of films whose P_norm falls by less than 0.01 and their mean drop in s;
and E4's per-film difference (P_norm drop minus s drop) averaged separately over the films below
and above the median baseline yes-minus-no margin.
E8. When ``clamp_live_mlp.json`` (E8, the same edit added to the live model) is present, the TEXT
shares for its arms, the share of samples giving neither yes nor no under E2's rule for both runs,
and the number of films with no sample giving yes or no.

Usage: python scripts/answer_text_checks.py   (writes results/clt_scale/answer_text_checks.json)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tracecxr.intervention.generation_check import (
    bootstrap_ci,
    effect_verdict,
    parse_answer,
    yes_rate,
)

RESULTS = Path("results/clt_scale")
CLAMPS = ("clamp_scale-1", "clamp_ablate+0", "clamp_scale-2")
TEXT_ARMS = ("base", *CLAMPS, "black")
E8_ARMS = ("base", *CLAMPS)
MAX_NEW, TRUNC_AT = 128, 126

_MD = re.compile(r"[*_#`>]")
_YN = re.compile(r"\b(yes|no)\b")
_FORMAT = re.compile(r"yes\s*(or|/)\s*no|no\s*(or|/)\s*yes|yes-or-no")
_NO_IMAGE = re.compile(
    r"no image|not (been )?provided( an| the)? image|haven't provided|have not provided|"
    r"did not provide|didn't provide|cannot see|can't see|unable to (see|view|access)|"
    r"(don't|do not) have (access to )?(the|an) image|"
    r"does not (show|appear to (be|show)) a chest x-ray|"
    r"image is not (in|provided|available|included|attached)|provide (the|an) image|"
    r"as an ai|text-based")
_STATED = re.compile(r"^\s*[\"']?(yes|no)\b|(answer\s*(is|:)|answer would be|i would say)\s*"
                     r"[\"'\s]*(yes|no)\b")


def _clean(t: str) -> str:
    return _MD.sub(" ", t).lower()


def parse_last(t: str) -> str:
    m = _YN.findall(_clean(t))
    return m[-1].upper() if m else "NONE"


def parse_mention(t: str) -> str:
    m = _YN.search(_FORMAT.sub(" ", _clean(t)))
    return m.group(1).upper() if m else "NONE"


def make_strict(truncated: dict[str, bool]):  # noqa: ANN201
    def parse_strict(t: str) -> str:
        c = _clean(t)
        if _NO_IMAGE.search(c) or (truncated[t] and not _STATED.search(c)):
            return "NONE"
        return parse_mention(t)
    return parse_strict


def verdicts(fp: list[dict], parse) -> dict:  # noqa: ANN001
    out: dict = {"changed": 0}
    rates: dict = {}
    for arm in ("base", *CLAMPS):
        labels = [[parse(t) for t in r["arms"][arm]["samples"]] for r in fp]
        out["changed"] += sum(a != b for r, ls in zip(fp, labels, strict=True)
                              for a, b in zip(r["arms"][arm]["samples_parsed"], ls, strict=True))
        rates[arm] = np.array([yes_rate(ls) for ls in labels])
    for arm in CLAMPS:
        sdrop = rates["base"] - rates[arm]
        for rd in ("raw", "norm"):
            drop = np.array([r["arms"]["base"][rd] - r["arms"][arm][rd] for r in fp])
            out[f"{arm}.{rd}"] = {"diff": bootstrap_ci(drop - sdrop),
                                  "verdict": effect_verdict(drop - sdrop, margin=0.05)}
    return out


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from tracecxr.core.config import MODELS  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    d = json.loads((RESULTS / "readout_sampled_answers.json").read_text())
    fp = [r for r in d["rows"] if r["group"].startswith("fp_")]
    assert d["max_new_tokens"] == MAX_NEW
    # the saved labels are E2's rule, re-derived here as a check
    assert all(parse_answer(t) == p for r in fp for a in TEXT_ARMS
               for t, p in zip(r["arms"][a]["samples"], r["arms"][a]["samples_parsed"],
                               strict=True))
    tok = AutoTokenizer.from_pretrained(MODELS["medgemma"].locator)
    texts = {t for r in fp for a in TEXT_ARMS for t in r["arms"][a]["samples"]}
    truncated = {t: len(tok.encode(t, add_special_tokens=False)) >= TRUNC_AT for t in texts}

    res: dict = {"n_films": len(fp), "parse": {}, "text": {}, "ceiling": {}}
    for name, parse in (("last", parse_last), ("mention", parse_mention),
                        ("strict", make_strict(truncated))):
        res["parse"][name] = verdicts(fp, parse)

    for arm in TEXT_ARMS:
        S = [t for r in fp for t in r["arms"][arm]["samples"]]
        res["text"][arm] = {
            "n": len(S),
            "truncated": float(np.mean([truncated[t] for t in S])),
            "no_image": float(np.mean([bool(_NO_IMAGE.search(_clean(t))) for t in S])),
            "mentions_format": float(np.mean([bool(_FORMAT.search(_clean(t))) for t in S])),
        }

    s = {a: np.array([yes_rate(r["arms"][a]["samples_parsed"]) for r in fp])
         for a in ("base", *CLAMPS)}
    gap = np.array([r["arms"]["base"]["gap"] for r in fp])
    low = gap < np.median(gap)
    for arm in CLAMPS:
        ndrop = np.array([r["arms"]["base"]["norm"] - r["arms"][arm]["norm"] for r in fp])
        sdrop = s["base"] - s[arm]
        tiny = ndrop < 0.01
        diff = ndrop - sdrop
        res["ceiling"][arm] = {
            "spearman": float(spearmanr(ndrop, sdrop).statistic),
            "n_tiny": int(tiny.sum()),
            "sdrop_tiny": float(sdrop[tiny].mean()),
            "diff_low_margin": bootstrap_ci(diff[low]),
            "diff_high_margin": bootstrap_ci(diff[~low]),
            "n_low": int(low.sum()),
        }

    live = RESULTS / "clamp_live_mlp.json"
    if live.exists():
        e8 = json.loads(live.read_text())
        assert e8["max_new_tokens"] == MAX_NEW
        new = {t for r in e8["rows"] for a in E8_ARMS for t in r["arms"][a]["samples"]} - texts
        truncated |= {t: len(tok.encode(t, add_special_tokens=False)) >= TRUNC_AT for t in new}
        res["e8_text"] = {}
        for arm in E8_ARMS:
            S = [t for r in e8["rows"] for t in r["arms"][arm]["samples"]]
            P = [p for r in e8["rows"] for p in r["arms"][arm]["samples_parsed"]]
            res["e8_text"][arm] = {
                "n": len(S),
                "none": float(np.mean([p == "NONE" for p in P])),
                "none_frozen": float(np.mean([p == "NONE" for r in fp
                                              for p in r["arms"][arm]["samples_parsed"]])),
                "truncated": float(np.mean([truncated[t] for t in S])),
                "no_image": float(np.mean([bool(_NO_IMAGE.search(_clean(t))) for t in S])),
                "films_unanswered": int(sum(all(p == "NONE" for p in r["arms"][arm]
                                                ["samples_parsed"]) for r in e8["rows"])),
            }

    out = RESULTS / "answer_text_checks.json"
    out.write_text(json.dumps(res, indent=1))
    for name, v in res["parse"].items():
        print(f"parse {name:8s} changed {v['changed']:5d}  " + "  ".join(
            f"{a[6:]}: norm {v[f'{a}.norm']['diff'][0]:+.3f} {v[f'{a}.norm']['verdict']}"
            f" / raw {v[f'{a}.raw']['diff'][0]:+.3f} {v[f'{a}.raw']['verdict']}" for a in CLAMPS))
    for arm, t in res["text"].items():
        print(f"text {arm:15s} truncated {t['truncated']:.3f}  no image {t['no_image']:.3f}  "
              f"mentions format {t['mentions_format']:.3f}")
    for arm, c in res["ceiling"].items():
        lo, hi = c["diff_low_margin"], c["diff_high_margin"]
        print(f"ceiling {arm:15s} spearman {c['spearman']:.2f}  tiny {c['n_tiny']}/{len(fp)} "
              f"(s drop {c['sdrop_tiny']:.2f})  low-margin {lo[0]:+.3f} [{lo[1]:+.3f},{lo[2]:+.3f}]"
              f"  high-margin {hi[0]:+.3f} [{hi[1]:+.3f},{hi[2]:+.3f}]")
    for arm, t in res.get("e8_text", {}).items():
        print(f"e8 {arm:15s} neither {t['none']:.3f} (frozen {t['none_frozen']:.3f})  truncated "
              f"{t['truncated']:.3f}  no image {t['no_image']:.3f}  "
              f"films unanswered {t['films_unanswered']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
