#!/usr/bin/env bash
# Launch training on a shared machine without hogging it.
# Usage:  bash run_train.sh [num_seeds] [iters]
# Each seed is an independent process; env stepping is Python-bound, so
# parallel seeds use cores far better than one process with more envs.
set -e
source venv/bin/activate

SEEDS=${1:-3}
ITERS=${2:-2000}

# Be a good citizen: cap BLAS/torch threads per process. Throughput here is
# dominated by the Catan simulator (pure Python), not matrix math, so this
# costs almost nothing.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs
for s in $(seq 0 $((SEEDS - 1))); do
  nohup nice -n 10 python -m catan_rl.train \
      --iters "$ITERS" --num-envs 8 --rollout-steps 512 \
      --warmup-iters 200 --pool-refresh 50 --save-every 100 \
      --seed "$s" --out "checkpoints/seed$s" \
      > "logs/seed$s.log" 2>&1 &
  echo "seed $s -> PID $! (log: logs/seed$s.log)"
done

echo
echo "Monitor:   tail -f logs/seed0.log"
echo "Stop all:  pkill -u \$USER -f catan_rl.train"
