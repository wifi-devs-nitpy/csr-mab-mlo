#!/usr/bin/env bash

set -euo pipefail 
# -------- CONFIG --------

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/run_$(date +"%Y-%m-%d_%H-%M-%S").log"

mkdir -p "$LOG_DIR"

# -------- NOTIFICATIONS --------

STATUS_DIR="$LOG_DIR/status"
mkdir -p "$STATUS_DIR"

send_status_email() {
    local subject=$1
    local body=$2
    local attachment=$3
    python send_run_status.py "$attachment" "$subject" "$body"
}

notify_job_complete() {
    local algo=$1
    local db=$2

    local done_file="$STATUS_DIR/done_${algo}_$(date +"%Y-%m-%d_%H-%M-%S").txt"
    {
        echo "Job completed successfully"
        echo "Algorithm: $algo"
        echo "Database: $db"
        echo "Time: $(date)"
    } > "$done_file"

    send_status_email \
        "MAB job completed: $algo" \
        "Job finished successfully for $algo using $db." \
        "$done_file"
}

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

    notify_job_complete "$algo" "$db"
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

echo "Emailing the attachments on successfull completion" 
python send_logs.py agents.zip

echo "System will shut down in 2 minutes..."

shutdown -s -t 120
