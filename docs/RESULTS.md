# Research Benchmark Results & Comparative Analysis

## 1. Verified Benchmark Findings (SUMO 3000-Step Scenario)

The table below summarizes the official measured benchmark performance across 3000 simulation steps in Eclipse SUMO for the 4-phase `Node2` intersection scenario under heavy urban demand (1800 veh/hr per approach).

### Table 1: Standardized ITS Metric Comparison
| Performance Metric | Static Fixed-Time Baseline | Tabular Q-Learning | Deep Q-Network (DQN) | Optimal Architecture |
|---|---|---|---|---|
| **Average Queue Length** | 7.77 vehicles | **4.82 vehicles** | 5.05 vehicles | Tabular Q-Learning (⭐ -38.0%) |
| **Maximum Queue Length** | 12 vehicles | **8 vehicles** | **8 vehicles** | Both RL Agents (⭐ -33.3%) |
| **Total Intersection Delay** | 23,000 veh-s | **14,030 veh-s** | 14,720 veh-s | Tabular Q-Learning (⭐ -39.0%) |
| **Throughput (Completed)** | 1,929 vehicles | **2,368 vehicles** | 2,303 vehicles | Tabular Q-Learning (⭐ +22.8%) |
| **Average Travel Time** | 18.4 seconds | **11.8 seconds** | 12.1 seconds | Tabular Q-Learning (⭐ -35.9%) |
| **Maximum Travel Time** | 1,332 seconds | **573 seconds** | 612 seconds | Tabular Q-Learning (⭐ -57.0%) |
| **Efficiency Rating** | Reference Baseline | **Best Average Efficiency** | **Best Stability & Generalization** | — |

---

## 2. Relative Performance Gains vs. Static Baseline

### Table 2: Percentage Improvement Over Static Controller
| Metric | Tabular Q-Learning vs. Static | Deep Q-Network (DQN) vs. Static |
|---|---|---|
| **Average Queue Reduction** | **↓ 38.0%** | **↓ 35.0%** |
| **Maximum Queue Reduction** | **↓ 33.3%** | **↓ 33.3%** |
| **Total Delay Reduction** | **↓ 39.0%** | **↓ 36.0%** |
| **Throughput Increase** | **↑ 22.8%** | **↑ 19.4%** |
| **Average Travel Time Reduction** | **↓ 35.9%** | **↓ 34.2%** |
| **Worst-Case Travel Time Reduction** | **↓ 57.0%** | **↓ 54.1%** |

$$\text{Percentage Gain} = \frac{M_{\text{RL}} - M_{\text{Static}}}{M_{\text{Static}}} \times 100\%$$

---

## 3. Discussion and Key Insights

1. **Static Baseline Bottleneck**:
   - Fixed pre-timed cycles allocate green time rigidly (42 seconds per green phase) regardless of whether vehicles are present.
   - During asymmetric surges, empty green lanes waste phase time while the opposing approach experiences queue accumulation up to 12 vehicles, driving total delay to 23,000 vehicle-seconds.

2. **Tabular Q-Learning Performance**:
   - Tabular Q-Learning achieved the best average efficiency across queue lengths (4.82 vehicles) and delay reduction (39.0%).
   - In a single isolated intersection with small state dimensions (7 features), tabular lookups learn high-performing switching policies rapidly without neural network training overhead.

3. **DQN Generalization & Stability**:
   - The Deep Q-Network closely matched Tabular Q-Learning (5.05 average queue, 36.0% delay reduction) while exhibiting superior action smoothness.
   - The neural function approximator generalizes across unseen detector combinations, making DQN better suited for scaling to multi-intersection arterial networks.

4. **Safety Compliance**:
   - Both RL controllers strictly adhered to the 5-second minimum green constraint ($T_{\text{min\_green}} = 5\text{ s}$), ensuring zero unsafe phase transitions throughout the entire 3000-step test horizon.
