#!/usr/bin/env bash
# Regenerate every number, table and figure from results/ and compile the paper. No GPU needed.
# Run from anywhere: bash paper/build.sh   (set PY to choose the interpreter; default python)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
# Fixed PDF timestamps, so the regenerated figures match the committed ones byte for byte.
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-0}

out=$($PY paper/make_numbers.py 2>&1); echo "$out"
if grep -q WARNING <<<"$out"; then echo "FAILED: a result file is missing or partial"; exit 1; fi
for f in paper/figures/make_fig_split.py paper/figures/make_fig_location.py \
         paper/figures/make_fig2.py; do
  $PY "$f" >/dev/null
done
if git rev-parse --git-dir >/dev/null 2>&1; then
  git diff --quiet -- paper/numbers.tex paper/tables \
    && echo "numbers.tex and tables/ match the committed versions" \
    || echo "NOTE: regenerated numbers differ from the committed ones (git diff paper/)"
  git diff --quiet -- paper/figures \
    || echo "NOTE: figure files differ from the committed ones (a different matplotlib?)"
fi

cd paper
tectonic -X compile main.tex --keep-logs >/dev/null
if grep -q "undefined value" main.log; then echo "FAILED: a \\val key is undefined"; exit 1; fi
if grep -qE "(Citation|Reference) .* undefined" main.log; then
  echo "FAILED: unresolved citation or reference"; exit 1
fi
rm -f main.log
echo "built paper/main.pdf"
