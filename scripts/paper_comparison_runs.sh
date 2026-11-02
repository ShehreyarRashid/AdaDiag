#!/usr/bin/env bash
# Every run behind results/paper_comparison/, in lanes that can go in parallel (one per
# core). Usage:  bash scripts/paper_comparison_runs.sh <lane>     lanes: main, sweep0 ...
# sweep5, ada, gaia_abl, gaia_drop, gaia_fields, hotel_abl, hotel_drop0, hotel_drop1 -- then: python scripts/make_paper_comparison.py
#
# Thread counts matter for bit-for-bit reproduction (results/ENVIRONMENT.md): `main` uses 2
# threads, so its GAIA/Hotel runs reproduce results/gaia.json and results/hotel.json
# exactly; every other lane uses 1 thread and carries its own TVDiag reference runs.
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/run_paper_comparison.py
O=results/paper_comparison
EMB=${EMB:-/tmp/adadiag_reembedded}     # re-embedded alerts for Fig. 15 (not committed)

case "${1:-}" in
  main)    # Tables 4-5, Figs. 11/13/14/18 -- default settings, 5 seeds
    python $R --dataset hotel    --arms v0_raw,v3_incident            --threads 2
    python $R --dataset gaia     --arms v0_raw,v3_incident            --threads 2
    python $R --dataset sockshop --arms v0_raw,v0_simplex,v3_incident --threads 2 ;;
  sweep[0-5])   # Fig. 16 -- TVDiag, four sweeps, GAIA, 3 seeds (in sixths)
    i=${1#sweep}
    python $R --dataset gaia --variants sweeps --arms v0_raw --seeds 3 --threads 1 --light \
      --shard "$i/6" --out "$O/sweep_gaia_tv$i.json" ;;
  ada)     # Fig. 16(b) -- AdaDiag along the contrastive loss scale
    python $R --dataset gaia --variants default,omega=0.0,omega=0.3,omega=0.5,omega=0.9 \
      --arms v3_incident --seeds 3 --threads 1 --light --out "$O/sweep_gaia_ada.json" ;;
  gaia_abl)   # Table 6 on GAIA (its TVDiag reference runs come from the sweep lanes)
    python $R --dataset gaia --variants no_aug,no_to,no_cm,no_dw --arms v0_raw --seeds 3 \
      --threads 1 --light --out "$O/ablate_gaia.json" ;;
  gaia_drop)  # Table 7 and Fig. 17 on GAIA
    for d in metric trace log trace,log metric,log metric,trace; do
      python $R --dataset gaia --drop "$d" --tag "drop:$d" --arms v0_raw --seeds 3 \
        --threads 1 --light --out "$O/ablate_gaia_drop.json"
    done ;;
  gaia_fields)  # Fig. 15 on GAIA: alert fields removed before embedding
    for v in trace_full trace_no_op trace_no_abtype metric_full metric_no_direction; do
      [ -d "$EMB/gaia_$v" ] || python scripts/reembed_alerts.py --dataset gaia --variant "$v" \
        --out "$EMB/gaia_$v"
      case $v in trace_*) d=metric,log ;; metric_*) d=trace,log ;; esac
      python $R --dataset gaia --data-root "$EMB/gaia_$v" --drop "$d" --tag "emb:$v" \
        --arms v0_raw --seeds 3 --threads 1 --light --out "$O/fig15_gaia.json"
    done ;;
  hotel_abl)   # Table 6 on Hotel Reservation, 5 seeds (with its own TVDiag reference)
    python $R --dataset hotel --variants ablations --arms v0_raw --seeds 5 --threads 1 \
      --light --out "$O/ablate_hotel.json" ;;
  hotel_drop0|hotel_drop1)  # Table 7 and Fig. 17 on Hotel Reservation (seeds split in two)
    i=${1#hotel_drop}
    for d in metric trace log trace,log metric,log metric,trace; do
      python $R --dataset hotel --drop "$d" --tag "drop:$d" --arms v0_raw --seeds 5 \
        --threads 1 --light --shard "$i/2" --out "$O/ablate_hotel_drop$i.json"
    done ;;
  *) echo "usage: $0 main|sweep0..sweep5|ada|gaia_abl|gaia_drop|gaia_fields|hotel_abl|hotel_drop0|hotel_drop1" >&2; exit 2 ;;
esac
