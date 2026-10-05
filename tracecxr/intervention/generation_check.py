"""Pure helpers for checking a first-token yes/no readout against the text the model generates.

The readouts (``P_raw``, ``P_norm``, answer mass) are all read off one next-token distribution.
Whether they say anything about the model's *answer* depends on what the off-answer mass turns
into once the model keeps writing. :mod:`scripts.readout_generation_check` generates that text;
this module holds the parse rule and the pre-registered verdict so they are unit-tested and
cannot drift between runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

ANSWERS = ("YES", "NO", "NONE")
_MARKDOWN = re.compile(r"[*_#`>]")
_ANSWER = re.compile(r"\b(yes|no)\b")


def parse_answer(text: str) -> str:
    """The pre-registered parse: the first whole-word ``yes`` or ``no``, else ``NONE``.

    Markdown characters are stripped first, so ``**Yes**`` parses as ``YES``. Matching is
    case-insensitive and whole-word, so ``"not"`` and ``"know"`` never match. The rule is
    deliberately simple; outputs it gets wrong are found by reading every greedy output by hand.
    """
    m = _ANSWER.search(_MARKDOWN.sub(" ", text).lower())
    return m.group(1).upper() if m else "NONE"


def call(p: float, threshold: float = 0.5) -> str:
    """A probability readout's yes/no call at ``threshold``."""
    return "YES" if p >= threshold else "NO"


def tally(answers: Iterable[str]) -> dict[str, int]:
    """Counts of ``YES``/``NO``/``NONE`` (every key present)."""
    out = dict.fromkeys(ANSWERS, 0)
    for a in answers:
        out[a] += 1
    return out


def replay_border_blocks(n_records: int, seed: int = 131, nb: int = 8) -> list[int]:
    """The random border block the occlusion audit drew for each of its first ``n_records`` films.

    ``scripts/occlusion_readout_audit.py`` drew one block per banked record, in order, from a
    ``default_rng(seed)`` over the border of the ``nb x nb`` block grid, and did not save it.
    Replaying the same draws recovers it, provided every record's image existed (true for the 20
    banked films).
    """
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    idx = np.arange(nb * nb)
    r, c = idx // nb, idx % nb
    border = idx[(r == 0) | (r == nb - 1) | (c == 0) | (c == nb - 1)]
    return [int(rng.choice(border)) for _ in range(n_records)]


def arm_verdict(norm: Sequence[float], greedy: Sequence[str], *, n_disagree: int = 3,
                n_hedge: int = 5, n_agree: int = 18) -> str:
    """The pre-registered verdict for one arm (see ``scripts/readout_generation_check.py``).

    Args:
        norm: ``P_norm`` per film.
        greedy: parsed greedy answer per film (``YES``/``NO``/``NONE``).

    Returns:
        ``DISAGREES`` if greedy says NO on ``n_disagree`` or more films the ``P_norm`` call
        says YES; else ``HEDGE-DOMINATED`` if ``n_hedge`` or more films parse to NONE; else
        ``VALIDATED`` if greedy matches the ``P_norm`` call on ``n_agree`` or more films; else
        ``MIXED``.
    """
    if len(norm) != len(greedy):
        raise ValueError("norm and greedy must have one entry per film")
    calls = [call(p) for p in norm]
    missed_no = sum(c == "YES" and g == "NO" for c, g in zip(calls, greedy, strict=True))
    if missed_no >= n_disagree:
        return "DISAGREES"
    if sum(g == "NONE" for g in greedy) >= n_hedge:
        return "HEDGE-DOMINATED"
    if sum(c == g for c, g in zip(calls, greedy, strict=True)) >= n_agree:
        return "VALIDATED"
    return "MIXED"


# ---------------------------------------------------------------------------------------------
# Sampled-answer readout: helpers for scripts/readout_sampled_answers.py,
# scripts/clamp_contrastive_answers.py and scripts/readout_eval_answers.py.
# ---------------------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z]+")


def opening_word(text: str) -> str:
    """The first word of a generated answer, lowercased, markdown and punctuation dropped.

    ``"**Yes**, the heart..."`` gives ``"yes"`` and ``"Based on the image"`` gives ``"based"``.
    Returns ``""`` when the text has no letters.
    """
    m = _WORD.search(_MARKDOWN.sub(" ", text).lower())
    return m.group(0) if m else ""


def yes_rate(parsed: Sequence[str], *, given_answer: bool = False) -> float:
    """Fraction of parsed samples that say ``YES``.

    With ``given_answer`` the denominator is the samples that say ``YES`` or ``NO`` (``NONE``
    dropped), the generated-text analogue of the yes-versus-no normalised readout. Returns NaN
    when the denominator is zero.
    """
    n_yes = sum(p == "YES" for p in parsed)
    den = n_yes + sum(p == "NO" for p in parsed) if given_answer else len(parsed)
    return n_yes / den if den else float("nan")


def bootstrap_ci(x: Sequence[float], *, n_boot: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """``(mean, lo, hi)``: the mean of ``x`` and its percentile bootstrap interval over units.

    The unit is whatever one entry of ``x`` is (a radiograph, in every caller), so resampling
    keeps each film's own sampling noise inside the interval. NaN entries are dropped.
    """
    import numpy as np  # noqa: PLC0415

    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(a.mean()), float(lo), float(hi)


def effect_verdict(diff: Sequence[float], *, margin: float = 0.05, **kw: object) -> str:
    """Does a readout's per-film effect match the sampled-answer effect?

    ``diff`` is, per film, (the readout's drop) minus (the drop in the sampled yes rate), both
    taken from base to the intervened arm. With ``(mean, lo, hi)`` its bootstrap interval:
    ``MISREADS`` if the interval excludes 0 and ``|mean| > margin``; ``TRACKS`` if the whole
    interval lies inside ``[-margin, margin]``; ``UNRESOLVED`` otherwise.
    """
    m, lo, hi = bootstrap_ci(diff, **kw)  # type: ignore[arg-type]
    if (lo > 0 or hi < 0) and abs(m) > margin:
        return "MISREADS"
    if lo >= -margin and hi <= margin:
        return "TRACKS"
    return "UNRESOLVED"


def auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Rank AUC with ties averaged (the Mann-Whitney statistic). NaN if one class is empty."""
    import numpy as np  # noqa: PLC0415
    from scipy.stats import rankdata  # noqa: PLC0415

    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auc_bootstrap(labels: Sequence[float], scores: dict[str, Sequence[float]], *,
                  n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, object]:
    """AUC of each score with a film-bootstrap interval, and every pairwise difference's interval.

    All scores are resampled on the SAME film indices in each replicate, so the differences are
    paired. Replicates where one class is empty are skipped.
    """
    import numpy as np  # noqa: PLC0415

    y = np.asarray(labels, dtype=float)
    S = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    rng = np.random.default_rng(seed)
    reps: dict[str, list[float]] = {k: [] for k in S}
    for _ in range(n_boot):
        i = rng.integers(0, y.size, y.size)
        if y[i].sum() in (0, y.size):
            continue
        for k, v in S.items():
            reps[k].append(auc(y[i], v[i]))
    R = {k: np.array(v) for k, v in reps.items()}

    def ci(a):  # noqa: ANN001, ANN202
        lo, hi = np.quantile(a, [alpha / 2, 1 - alpha / 2])
        return [float(lo), float(hi)]

    keys = list(S)
    return {
        "auc": {k: auc(y, S[k]) for k in keys},
        "ci": {k: ci(R[k]) for k in keys},
        "diff": {f"{a}-{b}": [auc(y, S[a]) - auc(y, S[b]), *ci(R[a] - R[b])]
                 for i, a in enumerate(keys) for b in keys[i + 1:]},
    }


def eval_verdict(diffs: Sequence[tuple[float, float, float]], *, margin: float = 0.03) -> str:
    """Pre-registered evaluation verdict over a set of AUC differences ``(est, lo, hi)``.

    ``READOUT CHANGES THE EVALUATION`` if any difference has ``|est| > margin`` with an interval
    that excludes 0; ``INNOCUOUS`` if every interval lies inside ``[-margin, margin]``;
    ``UNRESOLVED`` otherwise.
    """
    if any(abs(e) > margin and (lo > 0 or hi < 0) for e, lo, hi in diffs):
        return "READOUT CHANGES THE EVALUATION"
    if all(lo >= -margin and hi <= margin for _, lo, hi in diffs):
        return "INNOCUOUS"
    return "UNRESOLVED"
