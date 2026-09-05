#!/usr/bin/env bash
# One-time setup on a shared CSE machine (no sudo needed, everything user-level).
# Run from inside the catan-rl directory:  bash setup_cse.sh
set -e

echo "== Python check =="
python3 --version   # need 3.9+; if the default is ancient, try python3.11 / python3.12

echo "== Creating venv in ./venv =="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "== Installing dependencies (CPU torch: much smaller, and this workload is CPU-bound) =="
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install catanatron==3.2.1 catanatron_gym==4.0.0 gymnasium==0.29.1 numpy matplotlib

echo "== Smoke test =="
python -m catan_rl.train --smoke

echo "== Done. Activate later with: source venv/bin/activate =="
