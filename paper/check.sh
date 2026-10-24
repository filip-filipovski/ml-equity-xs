#!/usr/bin/env bash
# Paper gate: page cap, undefined references / citations, leftover placeholders, numbers-macro coverage.
# exit 0 = PASS.
set -uo pipefail
cd "$(dirname "$0")"
CAP=${PAPER_PAGE_CAP:-20}
fail=0
[ -f main.pdf ] || { echo "FAIL: main.pdf missing (run build.sh)"; exit 1; }
pages=$(pdfinfo main.pdf | awk '/^Pages:/{print $2}')
if [ "$pages" -gt "$CAP" ]; then echo "FAIL: $pages pages > cap $CAP"; fail=1; else echo "ok: $pages pages (cap $CAP)"; fi
if grep -q "Reference .* undefined\|Citation .* undefined\|There were undefined" main.log 2>/dev/null; then
  echo "FAIL: undefined references or citations"; grep "undefined" main.log | head -10; fail=1; else echo "ok: references resolved"; fi
if grep -q "Package natbib Warning.*multiply\|LaTeX Warning: Label(s) may have changed" main.log 2>/dev/null; then
  echo "FAIL: labels changed (rerun build)"; fail=1; fi
if grep -rn "TODO\|FIXME\|XXX\|PENDING-FILL" sections/*.tex | grep -v "^[^:]*:[0-9]*:%" ; then echo "FAIL: placeholders left in sections"; fail=1; elif grep -rnF "??" sections/*.tex ; then echo "FAIL: ?? left in sections"; fail=1; else echo "ok: no placeholders"; fi
# Every numbers macro referenced in the sections must be defined in tables/numbers.tex
missing=0
for m in $(grep -oh "\\\\num[A-Z][A-Za-z]*" sections/*.tex | sort -u); do
  grep -qF "newcommand{$m}" tables/numbers.tex || { echo "FAIL: macro $m not generated"; missing=1; }
done
[ $missing -eq 0 ] && echo "ok: all \\num* macros generated"
fail=$((fail|missing))
# Overfull boxes are a warning, not a failure
over=$(grep -c "Overfull \\\\hbox" main.log 2>/dev/null || true); echo "note: $over overfull hboxes"
[ $fail -eq 0 ] && echo "PASS" || echo "FAIL"
exit $fail
