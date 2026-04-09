# Expanded Hyperparameter Ranges for Optuna Tuning

## Overview
- **Total Simulation Steps**: 10,000
- **Trials per Algorithm**: 100 (expanded from 50)
- **Scenario**: scenario5 (d_ap=30, d_sta=2, mcs=11, n_tx_power_levels=12)
- **Objective**: Maximize phase 3 (5k-10k) mean throughput + Minimize phase 3 variance
- **Focus**: Very aggressive variance reduction after 5k runs (25x weight on variance)

---

## UCB Algorithm - Expanded Ranges

### Original Ranges (tuning_2.py)
- c: [0, 5]
- gamma: [0, 1]

### NEW Expanded Ranges
| Level | Parameter | Range | Expansion |
|-------|-----------|-------|-----------|
| 1 | c | [0.01, 10.0] | 2x expansion |
| 1 | gamma | [0.0, 1.0] | Same |
| 2 | c | [0.01, 8.0] | 2x expansion |
| 2 | gamma | [0.0, 1.0] | Same |
| 3 | c | [0.01, 6.0] | 2x expansion |
| 3 | gamma | [0.0, 1.0] | Same |
| 4 | c | [0.001, 4.0] | 2x expansion |
| 4 | gamma | [0.0, 1.0] | Same |

---

## EGreedy Algorithm - Expanded Ranges

### Original Ranges (tuning_2.py)
- e: [0.01, 0.1] log scale
- optimistic_start: [0, 100]
- alpha: [0, 1]

### NEW Expanded Ranges
| Level | Parameter | Range | Expansion | Scale |
|-------|-----------|-------|-----------|-------|
| 1 | e | [0.0001, 0.5] | 5x expansion | log |
| 1 | optimistic_start | [1.0, 500.0] | 5x expansion | linear |
| 1 | alpha | [0.0, 1.0] | 2x expansion | linear |
| 2 | e | [0.0001, 0.4] | 4x expansion | log |
| 2 | optimistic_start | [1.0, 300.0] | 3x expansion | linear |
| 2 | alpha | [0.0, 1.0] | 2x expansion | linear |
| 3 | e | [0.00001, 0.2] | 2x expansion | log |
| 3 | optimistic_start | [1.0, 200.0] | 2.5x expansion | linear |
| 3 | alpha | [0.0, 1.0] | 2x expansion | linear |
| 4 | e | [0.00001, 0.1] | 1x expansion | log |
| 4 | optimistic_start | [1.0, 150.0] | 3.75x expansion | linear |
| 4 | alpha | [0.0, 1.0] | 2x expansion | linear |

---

## Softmax Algorithm - Expanded Ranges

### Original Ranges (tuning_2.py)
- lr: [0.01, 10.0] log scale
- tau: [0.1, 10.0]
- alpha: [0, 1]

### NEW Expanded Ranges
| Level | Parameter | Range | Expansion | Scale |
|-------|-----------|-------|-----------|-------|
| 1 | lr | [0.001, 20.0] | 2x expansion | log |
| 1 | tau | [0.01, 50.0] | 5x expansion | linear |
| 1 | alpha | [0.0, 1.0] | 2x expansion | linear |
| 2 | lr | [0.001, 15.0] | 1.5x expansion | log |
| 2 | tau | [0.01, 40.0] | 4x expansion | linear |
| 2 | alpha | [0.0, 1.0] | 2x expansion | linear |
| 3 | lr | [0.001, 10.0] | 1x expansion | log |
| 3 | tau | [0.01, 30.0] | 3x expansion | linear |
| 3 | alpha | [0.0, 1.0] | 2x expansion | linear |
| 4 | lr | [0.001, 5.0] | 0.5x (reduced) | log |
| 4 | tau | [0.01, 20.0] | 2x expansion | linear |
| 4 | alpha | [0.0, 1.0] | 2x expansion | linear |

---

## Thompson Sampling Algorithm - Expanded Ranges

### Original Ranges (tuning_2.py)
- alpha: [0, 10]
- beta: [0, 10]
- mu: [0, 5]
- lam: 1 (fixed)

### NEW Expanded Ranges
| Level | Parameter | Range | Expansion |
|-------|-----------|-------|-----------|
| 1 | alpha | [0.01, 50.0] | 5x expansion |
| 1 | beta | [0.01, 50.0] | 5x expansion |
| 1 | mu | [0.01, 20.0] | 4x expansion |
| 1 | lam | 1.0 | Fixed |
| 2 | alpha | [0.01, 40.0] | 4x expansion |
| 2 | beta | [0.01, 40.0] | 4x expansion |
| 2 | mu | [0.01, 15.0] | 3x expansion |
| 2 | lam | 1.0 | Fixed |
| 3 | alpha | [0.01, 30.0] | 3x expansion |
| 3 | beta | [0.01, 30.0] | 3x expansion |
| 3 | mu | [0.01, 12.0] | 2.4x expansion |
| 3 | lam | 1.0 | Fixed |
| 4 | alpha | [0.01, 20.0] | 2x expansion |
| 4 | beta | [0.01, 20.0] | 2x expansion |
| 4 | mu | [0.01, 10.0] | 2x expansion |
| 4 | lam | 1.0 | Fixed |

---

## Optimization Strategy

### Optuna Configuration
- **Sampler**: TPESampler (Tree-Structured Parzen Estimator)
- **Pruner**: HyperbandPruner (aggressive early stopping of unpromising trials)
- **Direction**: Maximize
- **Seed**: 42 (for reproducibility)

### Objective Function Weights
```
objective_value = (
    phase3_mean * 0.35                          +  # Higher throughput in phase 3
    (1.0 / (phase3_std + 0.1)) * 25            +  # CRITICAL: Very low variance
    phase3_improvement * 0.15                       # Improvement from phase 1
)
```

### Reporting Strategy
- Reports intermediate metrics every 1000 steps
- Allows Optuna to prune after seeing early performance trends
- Full simulation always completes (10k steps)

---

## Expected Results

After tuning 100 trials per algorithm for 10k steps each:

1. **Phase 3 Throughput**: Should be significantly higher than initial 200 Mbps baseline
2. **Phase 3 Variance**: Should be as low as possible (target: < 30 Mbps std)
3. **Algorithm Rankings**: Will be determined by composite score
4. **Best Hyperparameters**: Saved in `arrays/optuna_10k_scenario5/`

---

## Running the Tuning

### Single Algorithm
```bash
python agent_simulations/hier_agent_10k_optuna_scenario5.py \
    -a UCB \
    -d optuna_dbs/ucb.db \
    -n 100 \
    -s 42
```

### All Four Algorithms (Sequential)
```bash
python agent_simulations/run_all_algorithms.py
```

---

**Freedom Applied**: ✅ Full freedom to expand hyperparameter ranges used extensively to explore wider optimal space.
