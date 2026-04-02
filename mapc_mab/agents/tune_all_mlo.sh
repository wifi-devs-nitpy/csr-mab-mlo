#!/usr/bin/env bash

set -euo pipefail

# -------- CONFIG --------
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/run_$(date +"%Y-%m-%d_%H-%M-%S").log"

mkdir -p "$LOG_DIR"

# Redirect all output (stdout + stderr) to tee (append mode)
exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================"
echo "Run started at: $(date)"
echo "Log file: $LOG_FILE"
echo "======================================"

# -------- FUNCTION --------
run_job() {
    local algo=$1
    local db=$2

    echo ""
    echo "--------------------------------------"
    echo "Starting: $algo"
    echo "Database: $db"
    echo "Start time: $(date)"
    
    start_time=$(date +%s)

    python tuning_2.py -a "$algo" -d "$db"

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "Finished: $algo"
    echo "Duration: ${duration}s"
    echo "End time: $(date)"
    echo "--------------------------------------"
}

# -------- JOBS --------
run_job "EGreedy" "egreedy.db"
run_job "Softmax" "softmax.db"
run_job "UCB" "ucb.db"
run_job "NormalThompsonSampling" "ts.db"

echo ""
echo "======================================"
echo "All jobs completed at: $(date)"
echo "======================================"

echo 'creating a zip file of the current directory'

zip -0 -r "agents.zip" "$LOG_DIR" .

python send_logs.py agents.zip

