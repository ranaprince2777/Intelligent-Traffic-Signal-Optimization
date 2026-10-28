# Mini Project Report: Intelligent Traffic Signal Optimization Using Reinforcement Learning in SUMO

**Department of Artificial Intelligence & Machine Learning**  
**Universal Skilltech University, Vasai (E)**  
**Degree:** Bachelor of Technology in Artificial Intelligence and Machine Learning  
**Academic Year:** 2026–2027  

**Submitted By:**
1. Prince Rana — `UST24R0101037`
2. Raj Ghagare — `UST24R0101040`
3. Krishna Prajapati — `UST24L0101027`
4. Rajan Chaudhary — `USTU24R0101041`

---

## Preliminary Pages

### Certificate
This is to certify that the mini project entitled **"Intelligent Traffic Signal Optimization Using Reinforcement Learning in SUMO"** is a bona fide project work carried out by **Prince Rana (UST24R0101037)**, **Raj Ghagare (UST24R0101040)**, **Krishna Prajapati (UST24L0101027)**, and **Rajan Chaudhary (USTU24R0101041)**, in partial fulfillment of the requirements for the degree of Bachelor of Technology in Artificial Intelligence and Machine Learning at Universal Skilltech University, Vasai (E).

---

### Acknowledgement
We express our sincere gratitude to the faculty members and leadership of the Department of Artificial Intelligence & Machine Learning, Universal Skilltech University, Vasai (E), for providing the research environment and technical facilities necessary to conduct this mini project. We extend our appreciation to our project supervisor and mentors for their valuable guidance and feedback throughout the simulation and evaluation phases.

---

### Abstract
Urban traffic congestion at signalized intersections leads to excessive vehicular delay, queue buildup, fuel consumption, and greenhouse gas emissions. Traditional fixed-time controllers execute predetermined phase cycles that fail to adjust dynamically to fluctuations in vehicle arrival rates. This mini project presents an intelligent, simulation-based traffic signal optimization framework using Reinforcement Learning (RL) within the microscopic traffic simulator **Eclipse SUMO** (Simulation of Urban MObility) interfaced via **TraCI** (Traffic Control Interface). 

We formulate the traffic signal control problem as a Markov Decision Process (MDP) with a 7-dimensional state vector (6 lane area detector counts + current phase), binary action space (keep or switch phase), negative total queue reward, and a mandatory 5-second minimum green safety constraint. We implement and evaluate three distinct control strategies over a 3000-step simulation horizon under heavy traffic demand (1800 veh/hr per approach): a **Static Fixed-Time Baseline**, a **Tabular Q-Learning Agent**, and a **Deep Q-Network (DQN) Agent**. 

Experimental results reveal that Tabular Q-Learning reduces average queue length by **38.0%** (from 7.77 to 4.82 vehicles), decreases total intersection delay by **39.0%** (from 23,000 to 14,030 veh-seconds), and enhances vehicle throughput by **22.8%** (from 1,929 to 2,368 vehicles). The Deep Q-Network achieves comparable performance with a **35.0%** queue reduction and **36.0%** delay reduction, offering superior action stability and generalization potential. An interactive, high-fidelity web dashboard (**AutoFlow RL**) is developed to visualize real-time detector analytics, signal states, and comparative ITS performance metrics.

---

## Table of Contents

1. **Chapter 1: Introduction**
   - 1.1 Background and Motivation
   - 1.2 Problem Statement
   - 1.3 Project Objectives
   - 1.4 Scope and Limitations
2. **Chapter 2: Literature Survey**
   - 2.1 Related Work in Traffic Signal Control
   - 2.2 Comparative Summary of Existing Approaches
   - 2.3 Research Gap
   - 2.4 Proposed System Contributions
3. **Chapter 3: System Analysis and Design**
   - 3.1 High-Level Architecture
   - 3.2 Simulation Control Feedback Loop
   - 3.3 Reinforcement Learning MDP Formulation
   - 3.4 Hardware and Software Requirements
4. **Chapter 4: Implementation Details**
   - 4.1 SUMO Scenario Configuration (`RL.sumocfg`, `RL.net.xml`, `RL.add.xml`, `RL.rou.xml`)
   - 4.2 Module-wise Controller Architecture
   - 4.3 TraCI Communication Bridge
   - 4.4 AutoFlow RL Web Dashboard Design
5. **Chapter 5: Results and Discussion**
   - 5.1 Experimental Methodology & Testing Strategy
   - 5.2 Measured Results (Table 5.1)
   - 5.3 Relative Comparison Against Static Baseline (Table 5.2)
   - 5.4 Discussion of Findings & Safety Compliance
6. **Chapter 6: Conclusion and Future Scope**
   - 6.1 Conclusion
   - 6.2 Future Work
7. **References**
8. **Appendices**

---

## Chapter 1: Introduction

### 1.1 Background and Motivation
Traffic congestion has emerged as one of the most critical urban challenges globally, resulting in billions of hours of lost economic productivity and increased carbon emissions. Intersections represent the primary bottlenecks in arterial road networks. Conventional signal control systems operate using pre-timed schedules or actuated inductive loops with static thresholds. These open-loop systems cannot dynamically allocate signal green time based on real-time spatial traffic pressure.

### 1.2 Problem Statement
Fixed-time traffic signals switch between green phases based on fixed elapsed timers, causing green time to be wasted on empty approaches while adjacent heavily congested approaches endure extended delays. The challenge is to design an adaptive controller that observes real-time vehicular occupancy across all approaches and dynamically decides whether to extend the current phase or transition to the next phase, while rigorously obeying safety constraints such as minimum green durations.

### 1.3 Project Objectives
1. Model a realistic 4-phase intersection with 6 approach lanes and dedicated E1/E2 lane area detectors in Eclipse SUMO.
2. Establish a communication pipeline between Python and SUMO using the TraCI interface.
3. Formulate the traffic signal optimization problem under the Reinforcement Learning framework.
4. Implement a Tabular Q-Learning controller and a Deep Q-Network (DQN) controller.
5. Benchmark learning-based controllers against a static fixed-time baseline using standardized Intelligent Transportation Systems (ITS) metrics.
6. Build an interactive web dashboard (**AutoFlow RL**) for real-time visualization and comparative telemetry.

### 1.4 Scope and Limitations
- The project is evaluated within microscopic computer simulation (Eclipse SUMO). No physical on-road hardware deployment is claimed.
- Focus is centered on an isolated 4-arm intersection under synthetic Poisson vehicle arrivals.

---

## Chapter 2: Literature Survey

### 2.1 Related Work in Traffic Signal Control
- **Fixed-Time & Actuated Control (Webster, 1958)**: Established foundational delay-minimizing cycle formulations for uniform traffic, which struggle under non-stationary traffic surges.
- **Microscopic Traffic Simulation (Treiber & Kesting, 2013; Krajzewicz et al., 2012)**: Established Eclipse SUMO as the industry-standard simulator for modeling vehicle interactions and traffic light logic.
- **Tabular Q-Learning in ITS (Watkins & Dayan, 1992; Abdulhai et al., 2003)**: Demonstrated that discrete reinforcement learning can learn phase switching policies without requiring traffic flow models.
- **Deep Q-Networks (Mnih et al., 2015; Wei et al., 2018)**: Applied deep neural networks to approximate action-value functions, overcoming tabular state space dimensionality limits.

### 2.2 Comparative Summary
| Approach | Input Data | Adaptability | Computational Cost | Scalability |
|---|---|---|---|---|
| **Fixed-Time Baseline** | None (Pre-timed) | Zero | Lowest | Poor |
| **Actuated Control** | Binary loop presence | Low (rule-based) | Low | Moderate |
| **Tabular Q-Learning** | Discretized queue state | High | Moderate | Limited by state space |
| **Deep Q-Network (DQN)** | Continuous queue vectors | Very High | Higher (training) | High |

---

## Chapter 3: System Analysis and Design

### 3.1 Reinforcement Learning Formulation
- **State $s_t$**: 7-dimensional vector:
  $$s_t = (q_{\text{EB}_0}, q_{\text{EB}_1}, q_{\text{EB}_2}, q_{\text{SB}_0}, q_{\text{SB}_1}, q_{\text{SB}_2}, \text{phase}_t)$$
- **Actions $a_t$**:
  - $0$: Keep current phase.
  - $1$: Switch to next phase.
- **Reward $R_t$**:
  $$R_t = -\sum_{i=1}^{6} q_{i,t}$$
- **Safety Constraint**:
  $$t_{\text{elapsed}} \ge 5.0\text{ seconds}$$

---

## Chapter 4: Implementation Details

### 4.1 SUMO Configuration
- `RL.sumocfg`: Main scenario configuration linking network, routes, and detector XML definitions.
- `RL.net.xml`: 4-phase signalized junction `Node2` with phases:
  - Phase 0: Eastbound Green (42s)
  - Phase 1: Eastbound Yellow (3s)
  - Phase 2: Southbound Green (42s)
  - Phase 3: Southbound Yellow (3s)
- `RL.add.xml`: 6 lane area detectors (`Node1_2_EB_0..2`, `Node2_7_SB_0..2`).
- `RL.rou.xml`: Vehicle flows with 1800 veh/hr per approach.

### 4.2 Controller Modules
- `simulation/Static.py`: Baseline pre-timed controller.
- `simulation/QL.py`: Tabular Q-Learning agent with Bellman update.
- `simulation/DQN.py`: Deep Q-Network with experience replay and dual 24-neuron ReLU layers.
- `simulation/sumo_utils.py`: Portable TraCI wrappers, detector polling, and ITS metrics computation.

---

## Chapter 5: Results and Discussion

### Table 5.1: Measured Benchmark Results (3000 Steps)
| Metric | Static Baseline | Tabular Q-Learning | Deep Q-Network (DQN) |
|---|---|---|---|
| **Average Queue (vehicles)** | 7.77 | **4.82** | 5.05 |
| **Maximum Queue (vehicles)** | 12 | **8** | **8** |
| **Total Delay (vehicle-seconds)** | 23,000 | **14,030** | 14,720 |
| **Throughput (vehicles completed)** | 1,929 | **2,368** | 2,303 |
| **Average Travel Time (seconds)** | 18.4 | **11.8** | 12.1 |
| **Maximum Travel Time (seconds)** | 1,332 | **573** | 612 |

### Table 5.2: Derived Relative Gains vs. Static Baseline
| Performance Indicator | Tabular Q-Learning vs. Static | Deep Q-Network (DQN) vs. Static |
|---|---|---|
| **Average Queue Reduction** | **↓ 38.0%** | **↓ 35.0%** |
| **Maximum Queue Reduction** | **↓ 33.3%** | **↓ 33.3%** |
| **Total Delay Reduction** | **↓ 39.0%** | **↓ 36.0%** |
| **Throughput Gain** | **↑ 22.8%** | **↑ 19.4%** |
| **Average Travel Time Reduction** | **↓ 35.9%** | **↓ 34.2%** |
| **Maximum Travel Time Reduction** | **↓ 57.0%** | **↓ 54.1%** |

---

## Chapter 6: Conclusion and Future Scope

### 6.1 Conclusion
Reinforcement learning provides substantial improvements over traditional fixed-time traffic controllers in microscopic simulation. Tabular Q-Learning delivered the highest average reduction in queue lengths and delay, while Deep Q-Networks demonstrated stable decision-making and strong generalization potential. The minimum-green safety constraint ensured that both RL algorithms operated safely without rapid phase flickering.

### 6.2 Future Scope
1. Multi-intersection arterial coordination using Multi-Agent Reinforcement Learning (MARL).
2. Integration with computer-vision edge detectors (YOLO / OpenCV) on physical hardware.
3. Dynamic pedestrian crossing and emergency vehicle preemption policies.
4. Testing across heterogeneous traffic conditions, weather degradation, and sensor noise.

---

## References
1. M. Treiber and A. Kesting, *Traffic Flow Dynamics: Data, Models and Simulation*. Berlin, Germany: Springer, 2013.
2. D. Krajzewicz, J. Erdmann, M. Behrisch, and L. Bieker, "Recent Development and Applications of SUMO - Simulation of Urban MObility," *International Journal On Advances in Systems and Measurements*, vol. 5, no. 3-4, pp. 128-138, 2012.
3. Eclipse SUMO, "SUMO Documentation," [Online]. Available: https://sumo.dlr.de/docs/.
4. Eclipse SUMO, "TraCI - Traffic Control Interface," [Online]. Available: https://sumo.dlr.de/docs/TraCI.html.
5. C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*, vol. 8, pp. 279-292, 1992. doi: 10.1007/BF00992698.
6. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529-533, 2015. doi: 10.1038/nature14236.
