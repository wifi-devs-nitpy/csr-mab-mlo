#!/bin/bash
set -euo pipefail

script_dir="$(dirname "${BASH_SOURCE[0]}")"
LOG_DIR="${script_dir}/logs"
LOG_FILE="$LOG_DIR/run_$(date +"%Y-%m-%d_%H-%M-%S").log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting the tuning"
echo "Starting at $(date)"

start_time=$(date +"%s")
python -u run_all_algorithms.py
end_time=$(date +"%s")
duration=$((end_time - start_time))
echo "Total time taken: ${duration}s"
echo "Completed successfully at $(date)"