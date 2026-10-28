# 🚦 Intelligent Traffic Signal Optimization Using Reinforcement Learning in SUMO
### AutoFlow RL: Adaptive Traffic Control & Real-Time Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-v1.24%2B-green.svg)](https://eclipse.dev/sumo/)
[![Vite](https://img.shields.io/badge/Vite-6.4.5-purple.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-29%20Passed-brightgreen.svg)]()

**Department of Artificial Intelligence & Machine Learning**  
**Universal Skilltech University, Vasai (E)**  
**Contributors:** Prince Rana (`UST24R0101037`), Raj Ghagare (`UST24R0101040`), Krishna Prajapati (`UST24L0101027`), Rajan Chaudhary (`USTU24R0101041`)  
**GitHub Repository:** [ranaprince2777/Intelligent-Traffic-Signal-Optimization](https://github.com/ranaprince2777/Intelligent-Traffic-Signal-Optimization)

---

## 📌 1. Project Title & Overview
**Intelligent Traffic Signal Optimization** is a simulation-based autonomous signal control framework built inside **Eclipse SUMO** (Simulation of Urban MObility) and controlled via Python using the **TraCI** (Traffic Control Interface) API. 

The project addresses intersection bottlenecks by comparing three signal control strategies under heavy urban traffic demand:
1. **Static Fixed-Time Baseline Controller**
2. **Tabular Q-Learning Reinforcement Learning Agent**
3. **Deep Q-Network (DQN) Neural Reinforcement Learning Agent**

The project includes **AutoFlow RL**, an interactive web dashboard built with Vite, HTML5 Canvas, and modern CSS for real-time visualization of detector readings, traffic light state machines, and comparative ITS performance telemetry.

> **Simulation Disclosure:** This project is implemented, validated, and evaluated exclusively in microscopic computer simulation using Eclipse SUMO. No live-road hardware deployment or camera-based real-world vehicle tracking is claimed.

---

## ⚠️ 2. Problem Statement
Fixed-time traffic controllers operate on rigid, pre-scheduled timing cycles (e.g. 42 seconds of green per approach) regardless of actual vehicle queues. During asymmetric traffic surges, empty lanes retain green lights while heavily congested opposing lanes accumulate severe queues. This open-loop scheduling leads to:
- Excessive intersection delay and fuel wastage.
- Severe vehicle queue spillbacks across approach corridors.
- Reduced overall intersection throughput.

---

## 🎯 3. Objectives
- Model an authentic 4-phase intersection with 6 approach lanes and dedicated E1/E2 lane area detectors in Eclipse SUMO.
- Establish an automated Python TraCI bidirectional control interface.
- Implement discrete Tabular Q-Learning and Deep Q-Network (DQN) controllers with strict **minimum-green safety guards (5s)**.
- Benchmark reinforcement learning models against fixed-time control using standardized Intelligent Transportation Systems (ITS) metrics.
- Deliver an interactive web dashboard (**AutoFlow RL**) for live simulation, telemetry streaming, and comparative academic benchmarks.

---

## 🚀 4. Key Features
- **7-Dimensional State Representation**: Combines 6 lane-area detector queue counts with the active signal phase.
- **Safety-First Actuation**: Hardcoded 5-second minimum green constraint prevents rapid phase flickering.
- **Dual RL Architectures**: Tabular Q-Learning for rapid tabular convergence + Deep Q-Network for neural generalization.
- **Ultra-Fast Pure-NumPy Fallback**: DQN runs high-speed vectorized forward/backward passes even when TensorFlow is not installed.
- **AutoFlow RL Web Dashboard**: High-performance 60 FPS HTML5 Canvas animation, real-time telemetry charts, and academic tables.
- **Modular Test Suite**: 29 automated pytest unit and integration tests covering XML configurations, TraCI safety, and RL updates.

---

## 🏗️ 5. System Architecture
```
                  ┌────────────────────────────────────────┐
                  │    Eclipse SUMO Microscopic Sim       │
                  │  (RL.sumocfg, RL.net.xml, RL.add.xml)  │
                  └──────────────────┬─────────────────────┘
                                     │ TraCI TCP / API
                                     ▼
                  ┌────────────────────────────────────────┐
                  │     AutoFlow RL TraCI Bridge           │
                  │  - State Extractor (6 Detectors+Phase) │
                  │  - Reward: R = - sum(Queues)           │
                  │  - Safety Guard: Min-Green Lock >= 5s  │
                  └──────────────────┬─────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│   Static Baseline  │    │ Tabular Q-Learning │    │    Deep Q-Network  │
│ Fixed 42s/3s Cycle │    │ Bellman Q(s, a)    │    │ 24x24 MLP + Buffer │
└──────────┬─────────┘    └──────────┬─────────┘    └──────────┬─────────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │ Export Telemetry & Metrics
                                     ▼
                  ┌────────────────────────────────────────┐
                  │     AutoFlow RL Web Dashboard          │
                  │  - Live Canvas Traffic Emulation       │
                  │  - Real-Time Queue & Reward Charts     │
                  │  - Table 5.1 & 5.2 Research Analytics  │
                  └────────────────────────────────────────┘
```

---

## 💻 6. Technology Stack
- **Simulation**: Eclipse SUMO (v1.24+), TraCI API, Netedit
- **Machine Learning**: Python 3.10+, NumPy, Pandas, TensorFlow / Keras (with pure-NumPy fallback)
- **Frontend / Visualization**: Vite 6, HTML5 Canvas, Vanilla ES Modules, CSS3 Glassmorphism
- **Testing & Quality Assurance**: pytest 9.1+, automated XML schema verifiers

---

## 📂 7. Project Folder Structure
```
Traffic_Signal_Optimization/
├── .gitignore                     # Comprehensive git ignore rules
├── index.html                     # AutoFlow RL dashboard web application
├── package.json                   # Vite frontend configuration
├── pytest.ini                     # Pytest environment setup
├── README.md                      # Primary project documentation
├── requirements.txt               # Python package dependencies
├── Static.py                      # Root entrypoint for static baseline controller
├── QL.py                          # Root entrypoint for Tabular Q-Learning
├── DQN.py                         # Root entrypoint for Deep Q-Network
│
├── simulation/                    # Simulation models, configs, and utilities
│   ├── DQN.py                     # DQN controller implementation
│   ├── QL.py                      # Tabular Q-Learning implementation
│   ├── Static.py                  # Static fixed-time baseline implementation
│   ├── run_experiments.py         # Multi-model benchmark suite runner
│   ├── traci_bridge.py            # Local HTTP telemetry bridge for web dashboard
│   ├── sumo_utils.py              # Shared TraCI utilities, safety lock, ITS metrics
│   ├── RL.sumocfg                 # Main SUMO configuration file
│   ├── RL.net.xml                 # Road network and 4-phase traffic light logic
│   ├── RL.add.xml                 # 6 lane area detector definitions
│   ├── RL.rou.xml                 # Vehicle flow routes and demand specifications
│   └── results/                   # Persisted benchmark metrics and telemetry
│       ├── verified_benchmark_report.json # Official Table 5.1/5.2 benchmarks
│       ├── metrics_comparison.json        # Latest benchmark output
│       └── metrics_comparison.csv         # Consolidated CSV comparison table
│
├── src/                           # AutoFlow RL Dashboard source code
│   ├── main.js                    # Web application controller and bridge polling
│   ├── sim/                       # Simulation and rendering modules
│   │   ├── trafficEngine.js       # Client-side high-fidelity simulation engine
│   │   ├── renderer.js            # 60 FPS HTML5 canvas renderer
│   │   └── charts.js              # Real-time streaming canvas chart engine
│   └── styles/
│       └── main.css               # Modern responsive glassmorphic stylesheet
│
├── tests/                         # Comprehensive automated test suite
│   ├── test_sumo_config.py        # XML schema, traffic light, and detector tests
│   ├── test_sumo_utils.py         # State extraction, reward, and safety lock tests
│   ├── test_static_controller.py  # Static baseline timing tests
│   ├── test_q_learning.py         # Bellman update and Q-table persistence tests
│   ├── test_dqn.py                # Neural network forward/backward and replay tests
│   ├── test_bridge.py             # TraCI HTTP bridge server API tests
│   └── test_benchmarks.py         # Table 5.1/5.2 metric math verification
│
├── report/                        # College mini project documentation
│   ├── Traffic_Signal_Optimization_Mini_Project_Report.pdf # 44-Page final report
│   └── REPORT_SOURCE.md           # Editable source document for the report
│
└── docs/                          # Detailed engineering documentation
    ├── ARCHITECTURE.md            # In-depth system and RL design
    ├── EXPERIMENTS.md             # Experimental protocol and hyperparameters
    ├── TESTING.md                 # QA test catalog and verification results
    └── RESULTS.md                 # Statistical findings and discussion
```

---

## ⚙️ 8. Hardware & Software Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+
- **Processor**: Intel Core i3 / AMD Ryzen 3 or higher
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk Space**: 500 MB free disk space
- **Python**: 3.10, 3.11, or 3.12
- **Node.js**: v18.0+ or v20.0+ (for AutoFlow RL dashboard)

---

## 📥 9. SUMO Installation Instructions
1. **Windows**:
   - Download the official installer from [SUMO Download Page](https://eclipse.dev/sumo/).
   - Install to standard directory: `C:\Program Files (x86)\Eclipse\Sumo`.
   - Set environment variable in PowerShell:
     ```powershell
     [System.Environment]::SetEnvironmentVariable('SUMO_HOME', 'C:\Program Files (x86)\Eclipse\Sumo', 'User')
     ```
2. **Ubuntu / Debian**:
   ```bash
   sudo apt-get update
   sudo apt-get install sumo sumo-tools sumo-gui
   echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc
   source ~/.bashrc
   ```

---

## 🐍 10. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/ranaprince2777/Intelligent-Traffic-Signal-Optimization.git
cd Intelligent-Traffic-Signal-Optimization

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🌐 11. Frontend Installation & Dashboard Startup
```bash
# Install frontend dependencies
npm install

# Start development server
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser. The dashboard automatically runs in **high-fidelity client-side emulation mode** out of the box, with zero external dependencies required.

---

## 🚦 12. Running the Static Baseline Controller
```bash
# Run 3000 steps headless
python simulation/Static.py --steps 3000

# Run with SUMO GUI visualization
python simulation/Static.py --steps 3000 --gui
```

---

## 🧠 13. Running Tabular Q-Learning
```bash
# Train Q-Learning agent (3000 steps)
python simulation/QL.py --steps 3000

# Evaluate trained Q-table with zero exploration (epsilon = 0)
python simulation/QL.py --steps 3000 --eval
```

---

## ⚡ 14. Running Deep Q-Network (DQN)
```bash
# Train DQN agent with experience replay
python simulation/DQN.py --steps 3000

# Evaluate trained DQN policy
python simulation/DQN.py --steps 3000 --eval
```

---

## 📊 15. Benchmark Suite & Verified Academic Results
Run the entire comparative benchmark suite in a single command:
```bash
python simulation/run_experiments.py --steps 3000
```

### Official Verified Benchmark Findings (SUMO 3000-Step Horizon)
*As documented in Table 5.1 & Table 5.2 of the college project report:*

| Metric | Static Baseline | Tabular Q-Learning | Deep Q-Network (DQN) | Best Performing Model |
|---|---|---|---|---|
| **Average Queue Length** | 7.77 veh | **4.82 veh** | 5.05 veh | Tabular Q-Learning (**-38.0%**) |
| **Maximum Queue Length** | 12 veh | **8 veh** | **8 veh** | Both RL Agents (**-33.3%**) |
| **Total Intersection Delay** | 23,000 veh-s | **14,030 veh-s** | 14,720 veh-s | Tabular Q-Learning (**-39.0%**) |
| **Vehicle Throughput** | 1,929 veh | **2,368 veh** | 2,303 veh | Tabular Q-Learning (**+22.8%**) |
| **Average Travel Time** | 18.4 s | **11.8 s** | 12.1 s | Tabular Q-Learning (**-35.9%**) |
| **Maximum Travel Time** | 1,332 s | **573 s** | 612 s | Tabular Q-Learning (**-57.0%**) |

---

## 🛠️ 16. Troubleshooting
- **SUMO_HOME not found**: Verify that the `SUMO_HOME` environment variable is set. The scripts automatically detect standard Windows and Linux directories and fall back to MockTraCI if SUMO is not installed.
- **Port 5173 occupied**: Start Vite on another port: `npm run dev -- --port 3000`.
- **Python UnicodeEncodeError**: Ensure your terminal uses UTF-8: `$env:PYTHONIOENCODING = "utf-8"`.

---

## ⚠️ 17. Limitations
- **Simulation Boundary**: Evaluated exclusively in Eclipse SUMO software simulation.
- **Isolated Intersection**: Models a single 4-arm junction; arterial network effects are not captured.
- **Poisson Arrivals**: Uses synthetic stationary traffic arrival rates rather than empirical loop detector feeds.

---

## 🔮 18. Future Scope
- Multi-intersection coordination using Multi-Agent Reinforcement Learning (MARL).
- CCTV camera vehicle counting via YOLOv8 edge object detection.
- Emergency vehicle preemption and pedestrian crosswalk request integration.

---

## 📚 19. Research References
1. M. Treiber and A. Kesting, *Traffic Flow Dynamics: Data, Models and Simulation*. Berlin, Germany: Springer, 2013.
2. D. Krajzewicz, J. Erdmann, M. Behrisch, and L. Bieker, "Recent Development and Applications of SUMO - Simulation of Urban MObility," *International Journal On Advances in Systems and Measurements*, vol. 5, no. 3-4, pp. 128-138, 2012.
3. Eclipse SUMO, "SUMO Documentation," [Online]. Available: https://sumo.dlr.de/docs/.
4. Eclipse SUMO, "TraCI - Traffic Control Interface," [Online]. Available: https://sumo.dlr.de/docs/TraCI.html.
5. C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*, vol. 8, pp. 279-292, 1992. doi: 10.1007/BF00992698.
6. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529-533, 2015. doi: 10.1038/nature14236.

---

## 🔗 20. GitHub Repository Information
- **Repository**: [https://github.com/ranaprince2777/Intelligent-Traffic-Signal-Optimization](https://github.com/ranaprince2777/Intelligent-Traffic-Signal-Optimization)
- **Author**: Prince Rana (`ranaprince2777`)
- **Institution**: Universal Skilltech University, Vasai (E)
