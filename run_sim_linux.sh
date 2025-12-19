#!/usr/bin/env bash
#
# The MIT License (MIT)
# Copyright © 2025 University of Perugia
#
# Author:
#
# - Francesco Binucci (francesco.binucci@cnit.it)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

set -euo pipefail

# -------------------------------------------------
# Usage:
#   ./run_sims.sh OUTPUT_DIR [SCRIPT] [REALIZATIONS] [ETA...]
#   ./run_sims.sh OUTPUT_DIR [SCRIPT] [REALIZATIONS] "ETA1,ETA2,ETA3"
#
# Examples:
#   ./run_sims.sh results
#   ./run_sims.sh results ./simulators/simulate_lyapunov_strategy_mt.py 10 0.6 0.8 1.0
#   ./run_sims.sh results ./simulators/simulate_lyapunov_strategy_mt.py 20 "0.6,0.8,1.0"
# -------------------------------------------------

folder="${1:-}"
if [[ -z "${folder}" ]]; then
  echo "Usage: $0 OUTPUT_DIR [SCRIPT] [REALIZATIONS] [ETA...]"
  exit 1
fi

script="${2:-./simulators/simulate_lyapunov_strategy_mt.py}"
realizations="${3:-10}"

# Collect ETAs from args 4+; default to 0.8 if none provided
shift || true
shift || true
shift || true

etas=()
if [[ $# -eq 0 ]]; then
  etas=("0.8")
else
  # Allow a single comma-separated string or multiple tokens
  for arg in "$@"; do
    IFS=',' read -r -a parts <<< "$arg"
    for e in "${parts[@]}"; do
      # skip empty entries (e.g., accidental trailing commas)
      [[ -n "${e}" ]] && etas+=("$e")
    done
  done
fi

for eta in "${etas[@]}"; do
  echo "Running: script='${script}' eta=${eta} realizations=${realizations} output='${folder}'"
  python "${script}" \
    --data_output_dir "${folder}" \
    --t_sim 15000 \
    --number_of_realizations "${realizations}" \
    --eta "${eta}" \
    --window_length 10 \
    --alpha 0.14 \
    --V 10 \
    --init_seed 512 \
    --init_sim_index 0
done
