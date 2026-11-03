#!/usr/bin/env bash
# README section 6.11 -- the softened prior (v0_soft) and the TVDiag-fidelity switches.
#   bash scripts/fidelity_runs.sh soft         # v0_soft, committed settings, 2 threads
#   bash scripts/fidelity_runs.sh fid0|fid1    # four arms under config.FIDELITY, two halves
# then: python scripts/make_fidelity_report.py
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/run_paper_comparison.py
O=results/fidelity
mkdir -p "$O"
case "${1:-}" in
  soft)
    for d in hotel gaia sockshop; do
      python $R --dataset "$d" --arms v0_soft --seeds 5 --threads 2 --light --out "$O/soft_$d.json"
    done ;;
  fid0|fid1)
    i=${1#fid}
    for d in hotel gaia sockshop; do
      python $R --dataset "$d" --variants fidelity --arms v0_raw,v0_simplex,v3_incident,v0_soft \
        --seeds 5 --threads 1 --light --shard "$i/2" --out "$O/fidelity_${d}_$i.json"
    done ;;
  *) echo "usage: $0 soft|fid0|fid1" >&2; exit 2 ;;
esac
