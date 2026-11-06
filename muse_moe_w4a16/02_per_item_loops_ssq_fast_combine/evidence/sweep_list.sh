#!/bin/sh
# Run bench.py for each "label|module|defines|buckets|shapes" line of a list file; one result line each.
#   sh dev/sweep_list.sh dev/sweeps/<name>.txt [reps]
LIST=$1
REPS=${2:-20}
cd "$(dirname "$0")/.." || exit 1
grep -v '^#' "$LIST" | grep -v '^\s*$' | while IFS='|' read -r label mod defs buckets shapes nsmap; do
  sh dev/setv.sh "$mod" || continue
  sh_opt=""
  [ -n "$shapes" ] && sh_opt="--shapes $shapes"
  MOE_DEFINES="$defs" MOE_BUCKETS="$buckets" MOE_NSMAP="$nsmap" python3 dev/bench.py dev/abl.py --reps "$REPS" --no-tests $sh_opt 2>&1 |
    grep -a -E '^(decode|verify|geomean)|rror' |
    awk -v v="$label" 'BEGIN{printf "%-30s: ", v} { if ($1=="geomean") printf "%s", $3; else if ($0 ~ /rror/) printf "ERR %s", $0; else printf "%s:%s(%s) ", $4, $10, $NF} END{print ""}'
done
