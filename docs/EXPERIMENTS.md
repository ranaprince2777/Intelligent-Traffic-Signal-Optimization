# Experimental Methodology & Protocols — AutoFlow RL

## 1. Experimental Setup

To ensure strict comparability across controllers, all experiments are executed under matched traffic flow conditions and identical simulation horizons.

### 1.1 Scenario Parameters
| Parameter | Value | Description |
|---|---|---|
| **Simulation Engine** | Eclipse SUMO (v1.24.0+) | Microscopic traffic simulation |
| **Simulation Step Length** | $1.0\text{ s}$ | Discrete time resolution |
| **Simulation Horizon** | $3000\text{ steps}$ ($50\text{ minutes}$) | Total evaluation duration |
| **Network Node** | `Node2` | 4-arm signalized intersection |
| **Detectors** | 6 E1/E2 Lane Area Detectors | 3 on Eastbound (`Node1_2_EB_0..2`), 3 on Southbound (`Node2_7_SB_0..2`) |
| **Detector Length** | $10.0\text{ m}$ | Vehicle queue detection zone |
| **Traffic Demand** | $1800\text{ vehicles/hour}$ | Heavy urban congestion regime per approach |
| **Vehicle Types** | Passenger cars, motorcycles, buses, trucks | Diverse kinematic and deceleration profiles |
| **Safety Constraint** | $T_{\text{min\_green}} = 5.0\text{ s}$ | Minimum green time before allowed phase transition |

---

## 2. Controller Hyperparameters

### 2.1 Tabular Q-Learning
- **Learning Rate ($\alpha$)**: $0.1$
- **Discount Factor ($\gamma$)**: $0.9$
- **Exploration Rate ($\epsilon$)**: $0.1$ (during training), $0.0$ (during evaluation)
- **Q-Table Size**: Variable (up to ~200 discrete observed states across 3000 steps)
- **Action Selection**: Epsilon-Greedy with safety lock pre-filtering

### 2.2 Deep Q-Network (DQN)
- **Architecture**: 7 $\rightarrow$ 24 (ReLU) $\rightarrow$ 24 (ReLU) $\rightarrow$ 2 (Linear)
- **Discount Factor ($\gamma$)**: $0.9$
- **Learning Rate**: $0.001$ (Adam Optimizer)
- **Loss Function**: Mean Squared Error (MSE)
- **Replay Buffer Capacity**: $2000$ transitions
- **Minibatch Size**: $32$ transitions
- **Training Frequency**: Gradient descent step every 4 simulation ticks

---

## 3. Evaluation Metrics (ITS Standards)

1. **Average Queue Length ($\bar{Q}$)**:
   $$\bar{Q} = \frac{1}{T} \sum_{t=1}^{T} \sum_{i=1}^{6} q_{i,t}$$
2. **Maximum Queue Length ($Q_{\text{max}}$)**:
   $$Q_{\text{max}} = \max_{t} \sum_{i=1}^{6} q_{i,t}$$
3. **Total Intersection Delay ($D_{\text{total}}$)**:
   $$D_{\text{total}} = \sum_{t=1}^{T} \left( \sum_{i=1}^{6} q_{i,t} \cdot \Delta t \right)$$
4. **Throughput ($N_{\text{completed}}$)**:
   Total count of completed trips arriving at destination edges within the simulation horizon.
5. **Average Travel Time ($\bar{T}_{\text{travel}}$)**:
   $$\bar{T}_{\text{travel}} = \frac{1}{N_{\text{completed}}} \sum_{k=1}^{N_{\text{completed}}} (t_{\text{arrived}, k} - t_{\text{departed}, k})$$
6. **90th Percentile Travel Time ($T_{90}$)**:
   The 90th percentile of vehicle journey times, characterizing worst-case congestion reliability.
7. **Maximum Travel Time ($T_{\text{max}}$)**:
   The single longest recorded trip duration among completed vehicles.

---

## 4. Execution Commands

### Running Individual Controller Trials
```bash
# Static fixed-time baseline
python simulation/Static.py --steps 3000

# Tabular Q-Learning training
python simulation/QL.py --steps 3000

# Deep Q-Network training
python simulation/DQN.py --steps 3000

# Tabular Q-Learning evaluation (epsilon = 0)
python simulation/QL.py --steps 3000 --eval

# Deep Q-Network evaluation (greedy inference)
python simulation/DQN.py --steps 3000 --eval
```

### Running Consolidated Benchmark Suite
```bash
python simulation/run_experiments.py --steps 3000
```
This automatically generates:
- `simulation/results/metrics_comparison.json`
- `simulation/results/metrics_comparison.csv`
