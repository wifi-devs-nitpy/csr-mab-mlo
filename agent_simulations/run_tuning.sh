#!/bin/bash

################################################################################
# OPTUNA HYPERPARAMETER TUNING - BASH ORCHESTRATION SCRIPT
# Scenario: Simple scenario5 with 10,000 steps
# Algorithms: UCB, EGreedy, Softmax, NormalThompsonSampling
# Multi-objective: Maximize phase3_mean, Minimize phase3_std
################################################################################

set -o pipefail  # Exit if any command in pipeline fails

# ============================================================================
# CONFIGURATION
# ============================================================================
AGENTS=("UCB" "EGreedy" "Softmax" "NormalThompsonSampling")
N_TRIALS=100
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/arrays/optuna_10k_scenario5"
DB_DIR="${SCRIPT_DIR}/optuna_dbs"
LOGS_DIR="${SCRIPT_DIR}/logs/optuna_tuning"

PYTHON_SCRIPT="${SCRIPT_DIR}/hier_agent_10k_optuna_scenario5.py"

# Create directories
mkdir -p "${DB_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOGS_DIR}"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📋 $*"
}

log_start() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ▶️  $*"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $*" >&2
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $*"
}

print_header() {
    echo ""
    echo "================================================================================"
    echo "$*"
    echo "================================================================================"
    echo ""
}

format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print_header "OPTUNA HYPERPARAMETER TUNING - ALL 4 ALGORITHMS"

log_info "Configuration:"
log_info "  Scenario: scenario5 (d_ap=30, d_sta=2, mcs=11)"
log_info "  Steps per trial: 10,000"
log_info "  Trials per algorithm: ${N_TRIALS}"
log_info "  Optimization: Pareto multi-objective (maximize phase3_mean, minimize phase3_std)"
log_info "  Python script: ${PYTHON_SCRIPT}"
log_info "  Results directory: ${RESULTS_DIR}"
log_info "  Logs directory: ${LOGS_DIR}"

# Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    log_error "Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

# Track overall timing
START_TIME=$(date +%s)
FAILED_AGENTS=()
SUCCESSFUL_AGENTS=()

# ============================================================================
# RUN EACH ALGORITHM
# ============================================================================

for AGENT_NAME in "${AGENTS[@]}"; do
    print_header "TUNING: ${AGENT_NAME}"
    
    DB_PATH="${DB_DIR}/optuna_${AGENT_NAME,,}.db"
    LOG_FILE="${LOGS_DIR}/${AGENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
    
    log_start "Running ${AGENT_NAME} (100 trials × 10k steps)"
    log_info "Log file: ${LOG_FILE}"
    log_info "Database: ${DB_PATH}"
    
    AGENT_START=$(date +%s)
    
    # Run Python script with logging
    if python "${PYTHON_SCRIPT}" \
        -a "${AGENT_NAME}" \
        -d "${DB_PATH}" \
        -n "${N_TRIALS}" \
        -s "${SEED}" \
        > >(tee -a "${LOG_FILE}") \
        2> >(tee -a "${LOG_FILE}" >&2); then
        
        AGENT_END=$(date +%s)
        AGENT_DURATION=$((AGENT_END - AGENT_START))
        
        log_success "${AGENT_NAME} completed in $(format_duration ${AGENT_DURATION})"
        SUCCESSFUL_AGENTS+=("${AGENT_NAME}")
        
        # Verify output files were created
        RESULTS_FILE="${RESULTS_DIR}/best_${AGENT_NAME,,}_params.json"
        PARETO_FILE="${RESULTS_DIR}/pareto_front_${AGENT_NAME,,}.json"
        
        if [ -f "${RESULTS_FILE}" ]; then
            log_info "✓ Results saved: ${RESULTS_FILE}"
        else
            log_warning "Results file not found: ${RESULTS_FILE}"
        fi
        
        if [ -f "${PARETO_FILE}" ]; then
            log_info "✓ Pareto front saved: ${PARETO_FILE}"
            PARETO_SIZE=$(jq '.n_trials' "${PARETO_FILE}" 2>/dev/null || echo "?")
            log_info "  Pareto front size: ${PARETO_SIZE} trials"
        else
            log_warning "Pareto file not found: ${PARETO_FILE}"
        fi
    else
        log_error "${AGENT_NAME} FAILED"
        FAILED_AGENTS+=("${AGENT_NAME}")
        
        # Show last 50 lines of log
        log_error "Last 50 lines of log:"
        tail -50 "${LOG_FILE}" | sed 's/^/  /' >&2
    fi
    
    echo ""
done

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print_header "GENERATING COMPARISON VISUALIZATIONS"

if [ ${#SUCCESSFUL_AGENTS[@]} -gt 0 ]; then
    log_start "Creating comparison plots from results..."
    
    VIZ_SCRIPT="${SCRIPT_DIR}/generate_comparison_plots.py"
    
    # Check if visualization Python script exists, else create inline
    if [ -f "${VIZ_SCRIPT}" ]; then
        python "${VIZ_SCRIPT}" "${RESULTS_DIR}" 2>&1 | tee -a "${LOGS_DIR}/visualization.log"
    else
        # Use Python to generate plots directly
        python3 << 'EOF'
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

results_dir = Path('arrays/optuna_10k_scenario5')

# Load all results
all_results = {}
for agent_file in results_dir.glob('best_*.json'):
    agent_name = agent_file.stem.replace('best_', '').upper()
    if agent_name == 'NORMALTHOMPSONSAMPLING':
        agent_name = 'NormalThompsonSampling'
    try:
        with open(agent_file) as f:
            all_results[agent_name] = json.load(f)
    except:
        pass

if not all_results:
    print("No results found")
    sys.exit(1)

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Performance Metrics Comparison (Pareto-Optimized)', fontsize=14, fontweight='bold')

agents = list(all_results.keys())
phase3_means = [all_results[a]['metrics']['phase3_mean'] for a in agents]
phase3_stds = [all_results[a]['metrics']['phase3_std'] for a in agents]

# Subplot 1: Mean throughput
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
bars = ax.bar(agents, phase3_means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Throughput (Mbps)', fontsize=11)
ax.set_title('Phase 3 Mean Throughput (5k-10k) ↑', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars, phase3_means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{mean:.1f}',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 2: Variance (lower is better)
ax = axes[1]
colors_std = ['green' if std < 30 else 'orange' if std < 50 else 'red' for std in phase3_stds]
bars = ax.bar(agents, phase3_stds, color=colors_std, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Standard Deviation (Mbps)', fontsize=11)
ax.set_title('Phase 3 Variance ↓ (CRITICAL METRIC)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, std in zip(bars, phase3_stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{std:.1f}',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(str(results_dir / 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved comparison plot: {results_dir / 'metrics_comparison.png'}")
plt.close()
EOF
    fi
    
    log_success "Visualizations generated"
else
    log_error "No successful tuning runs - skipping visualizations"
fi

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

print_header "TUNING COMPLETE"

log_info "Summary:"
log_info "  Total time: $(format_duration ${TOTAL_DURATION})"
log_success "Successful: ${#SUCCESSFUL_AGENTS[@]} algorithms (${SUCCESSFUL_AGENTS[*]})"

if [ ${#FAILED_AGENTS[@]} -gt 0 ]; then
    log_error "Failed: ${#FAILED_AGENTS[@]} algorithms (${FAILED_AGENTS[*]})"
fi

log_info ""
log_info "Output locations:"
log_info "  Results:  ${RESULTS_DIR}/"
log_info "  Logs:     ${LOGS_DIR}/"
log_info "  Database: ${DB_DIR}/"

log_info ""
log_info "View results:"
log_info "  JSON summary:    cat ${RESULTS_DIR}/best_UCB_params.json"
log_info "  Comparison plot: ${RESULTS_DIR}/metrics_comparison.png"
log_info "  All logs:        ls -lh ${LOGS_DIR}/"

if [ ${#FAILED_AGENTS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
