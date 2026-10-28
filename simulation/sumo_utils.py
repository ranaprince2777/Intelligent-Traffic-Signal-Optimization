"""
AutoFlow RL - Shared SUMO, TraCI, and ITS Evaluation Utilities
Provides portable path resolution, SUMO binary discovery, detector definitions,
state encoding, min-green safety constraint enforcement, ITS metric calculations,
telemetry persistence, and a high-fidelity MockTraCI environment for testing.
"""

import os
import sys
import json
import csv
import shutil
import random
import numpy as np

# Detector configuration matching RL.add.xml
DETECTORS = [
    "Node1_2_EB_0",
    "Node1_2_EB_1",
    "Node1_2_EB_2",
    "Node2_7_SB_0",
    "Node2_7_SB_1",
    "Node2_7_SB_2"
]
TRAFFIC_LIGHT_ID = "Node2"
ACTIONS = [0, 1]  # 0 = Keep phase, 1 = Switch phase
MIN_GREEN_STEPS = 5  # Minimum green duration in seconds

# Static phase program matching RL.net.xml (Node2: 4 phases)
# Phase 0: EB Green (42s)
# Phase 1: EB Yellow (3s)
# Phase 2: SB Green (42s)
# Phase 3: SB Yellow (3s)
DEFAULT_PHASE_DURATIONS = [42, 3, 42, 3]

def find_sumo_config():
    """Locate RL.sumocfg dynamically from project directories."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, ".."))
    
    candidates = [
        os.path.join(curr_dir, "RL.sumocfg"),
        os.path.join(project_root, "simulation", "RL.sumocfg"),
        os.path.join(project_root, "RL.sumocfg"),
        os.path.join(project_root, "Traffic_Signal_Optimization-main", "RL.sumocfg"),
        os.path.join(os.getcwd(), "simulation", "RL.sumocfg"),
        os.path.join(os.getcwd(), "RL.sumocfg"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None

def get_sumo_binary(use_gui=False):
    """
    Locates the SUMO binary using SUMO_HOME, system PATH, or standard installation directories.
    Returns (binary_path, error_message).
    """
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        # Common Windows paths
        standard_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
            r"C:\Program Files\Sumo",
            r"C:\Eclipse\Sumo",
            r"C:\Sumo",
            "/usr/share/sumo",
            "/usr/local/share/sumo",
            "/opt/homebrew/opt/sumo/share/sumo"
        ]
        for p in standard_paths:
            if os.path.isdir(p):
                sumo_home = p
                os.environ['SUMO_HOME'] = p
                break

    # Check system PATH via shutil.which
    cli_name = 'sumo.exe' if os.name == 'nt' else 'sumo'
    gui_name = 'sumo-gui.exe' if os.name == 'nt' else 'sumo-gui'
    
    which_cli = shutil.which('sumo') or shutil.which(cli_name)
    which_gui = shutil.which('sumo-gui') or shutil.which(gui_name)

    if use_gui and which_gui and os.path.isfile(which_gui):
        return which_gui, None
    if which_cli and os.path.isfile(which_cli) and not use_gui:
        return which_cli, None

    if not sumo_home or not os.path.isdir(sumo_home):
        return None, (
            "SUMO_HOME environment variable is not set and standard installation paths were not found.\n"
            "Please install Eclipse SUMO (v1.24+ recommended) and set SUMO_HOME:\n"
            "  Windows PowerShell: $env:SUMO_HOME = 'C:\\Program Files (x86)\\Eclipse\\Sumo'\n"
            "  Linux/macOS:        export SUMO_HOME=/usr/share/sumo\n"
            "  Official Guide:     https://sumo.dlr.de/docs/Installing/index.html"
        )

    tools_dir = os.path.join(sumo_home, 'tools')
    if os.path.isdir(tools_dir) and tools_dir not in sys.path:
        sys.path.append(tools_dir)

    bin_dir = os.path.join(sumo_home, 'bin')
    gui_path = os.path.join(bin_dir, gui_name)
    cli_path = os.path.join(bin_dir, cli_name)

    if use_gui and os.path.isfile(gui_path):
        return gui_path, None
    elif os.path.isfile(cli_path):
        return cli_path, None
    elif os.path.isfile(gui_path):
        return gui_path, None
    else:
        return None, f"SUMO binary not found in {bin_dir}."

def safe_get_detector(traci_module, detector_id):
    """Safely get number of vehicles in detector zone."""
    try:
        return int(traci_module.lanearea.getLastStepVehicleNumber(detector_id))
    except Exception:
        return 0

def safe_get_phase(traci_module, tls_id=TRAFFIC_LIGHT_ID):
    """Safely get current signal phase index."""
    try:
        return int(traci_module.trafficlight.getPhase(tls_id))
    except Exception:
        return 0

def get_state(traci_module, tls_id=TRAFFIC_LIGHT_ID):
    """
    Constructs the 7-dimensional state vector:
    (EB_0, EB_1, EB_2, SB_0, SB_1, SB_2, current_phase)
    """
    q_vals = [safe_get_detector(traci_module, d) for d in DETECTORS]
    phase = safe_get_phase(traci_module, tls_id)
    return tuple(q_vals + [phase])

def get_reward(state):
    """Reward is negative sum of observed lane queues."""
    total_queue = sum(state[:-1])
    return -float(total_queue)

def apply_action_with_guard(traci_module, action, current_step, last_switch_step, 
                            min_green=MIN_GREEN_STEPS, tls_id=TRAFFIC_LIGHT_ID):
    """
    Applies the chosen action while enforcing the min-green safety constraint.
    action: 0 = keep phase, 1 = switch to next phase.
    Returns: (updated_last_switch_step, was_switched)
    """
    if action == 0:
        return last_switch_step, False

    time_since_switch = current_step - last_switch_step
    if time_since_switch >= min_green:
        try:
            curr_phase = safe_get_phase(traci_module, tls_id)
            # Default intersection has 4 phases (0: EB Green, 1: EB Yellow, 2: SB Green, 3: SB Yellow)
            num_phases = 4
            try:
                logics = traci_module.trafficlight.getAllProgramLogics(tls_id)
                if logics and len(logics) > 0 and hasattr(logics[0], 'phases'):
                    num_phases = len(logics[0].phases)
            except Exception:
                num_phases = 4

            next_phase = (curr_phase + 1) % num_phases
            traci_module.trafficlight.setPhase(tls_id, next_phase)
            return current_step, True
        except Exception:
            return last_switch_step, False
    else:
        # Min-green lock active; cannot switch phase
        return last_switch_step, False

def compute_its_metrics(queue_history, travel_times, step_length=1.0):
    """Computes standardized ITS performance metrics."""
    if queue_history and len(queue_history) > 0:
        total_delay = float(sum(q * step_length for q in queue_history))
        avg_queue = float(sum(queue_history)) / len(queue_history)
        max_queue = int(max(queue_history))
    else:
        total_delay = 0.0
        avg_queue = 0.0
        max_queue = 0

    throughput = len(travel_times)

    if travel_times and len(travel_times) > 0:
        arr = np.array(travel_times, dtype=np.float32)
        avg_tt = float(arr.mean())
        max_tt = float(arr.max())
        p90_tt = float(np.percentile(arr, 90))
    else:
        avg_tt = 0.0
        max_tt = 0.0
        p90_tt = 0.0

    return {
        "total_delay": round(total_delay, 2),
        "avg_queue": round(avg_queue, 2),
        "max_queue": max_queue,
        "throughput": throughput,
        "avg_travel_time": round(avg_tt, 2),
        "p90_travel_time": round(p90_tt, 2),
        "max_travel_time": round(max_tt, 2)
    }

def save_metrics_to_json(metrics, filepath):
    """Save metrics to JSON file with clean formatting."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {filepath}")

def save_telemetry_to_csv(step_history, time_history, queue_history, reward_history, filepath):
    """Save simulation time-series telemetry to CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_seconds", "queue_length", "instant_reward"])
        for s, t, q, r in zip(step_history, time_history, queue_history, reward_history):
            writer.writerow([s, round(t, 2), q, round(r, 2)])
    print(f"Telemetry saved to: {filepath}")

# -------------------------------------------------------------
# High-Fidelity Mock TraCI Environment for Verification
# -------------------------------------------------------------
class MockLaneArea:
    def __init__(self, parent):
        self.parent = parent
    def getLastStepVehicleNumber(self, det_id):
        return self.parent.detector_counts.get(det_id, 0)

class MockTrafficLight:
    def __init__(self, parent):
        self.parent = parent
    def getPhase(self, tls_id):
        return self.parent.phase
    def setPhase(self, tls_id, phase_index):
        self.parent.phase = int(phase_index)
        self.parent.phase_timer = 0
    def getAllProgramLogics(self, tls_id):
        class Phase:
            def __init__(self, dur):
                self.duration = dur
        class Logic:
            def __init__(self):
                self.phases = [Phase(42), Phase(3), Phase(42), Phase(3)]
        return [Logic()]

class MockVehicle:
    def __init__(self, parent):
        self.parent = parent
    def getDepartedIDList(self):
        departed = list(self.parent.newly_departed)
        self.parent.newly_departed.clear()
        return departed
    def getArrivedIDList(self):
        arrived = list(self.parent.newly_arrived)
        self.parent.newly_arrived.clear()
        return arrived

class MockSimulation:
    def __init__(self, parent):
        self.parent = parent
    def getTime(self):
        return float(self.parent.sim_step)

class MockTraCI:
    """
    Accurate in-memory emulation of SUMO + TraCI interface matching RL.net.xml
    and RL.add.xml. Enables comprehensive testing and reproducible experiments
    on systems without native SUMO binary installation.
    """
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.lanearea = MockLaneArea(self)
        self.trafficlight = MockTrafficLight(self)
        self.vehicle = MockVehicle(self)
        self.simulation = MockSimulation(self)
        
        self.phase = 0
        self.phase_timer = 0
        self.sim_step = 0
        self.detector_counts = {d: 0 for d in DETECTORS}
        self.newly_departed = []
        self.newly_arrived = []
        self.active_vehicles = {}
        self.veh_counter = 1

    def start(self, cmd):
        self.sim_step = 0
        self.phase = 0
        self.phase_timer = 0
        return True

    def simulationStep(self):
        self.sim_step += 1
        self.phase_timer += 1

        # Simulate vehicle arrivals (Poisson process, ~0.45 arrivals/sec per approach)
        for i in range(3):
            eb_det = DETECTORS[i]
            sb_det = DETECTORS[i + 3]

            if random.random() < 0.22:
                self.detector_counts[eb_det] = min(15, self.detector_counts[eb_det] + 1)
                vid = f"veh_EB_{self.veh_counter}"
                self.veh_counter += 1
                self.active_vehicles[vid] = self.sim_step
                self.newly_departed.append(vid)

            if random.random() < 0.22:
                self.detector_counts[sb_det] = min(15, self.detector_counts[sb_det] + 1)
                vid = f"veh_SB_{self.veh_counter}"
                self.veh_counter += 1
                self.active_vehicles[vid] = self.sim_step
                self.newly_departed.append(vid)

        # Clearance based on current green phase:
        # Phase 0: EB Green -> Clears EB detectors
        # Phase 2: SB Green -> Clears SB detectors
        if self.phase == 0:
            for i in range(3):
                det = DETECTORS[i]
                if self.detector_counts[det] > 0 and random.random() < 0.70:
                    self.detector_counts[det] -= 1
                    # Vehicle completed
                    for vid, dep_time in list(self.active_vehicles.items()):
                        if "EB" in vid:
                            self.newly_arrived.append(vid)
                            del self.active_vehicles[vid]
                            break
        elif self.phase == 2:
            for i in range(3):
                det = DETECTORS[i + 3]
                if self.detector_counts[det] > 0 and random.random() < 0.70:
                    self.detector_counts[det] -= 1
                    for vid, dep_time in list(self.active_vehicles.items()):
                        if "SB" in vid:
                            self.newly_arrived.append(vid)
                            del self.active_vehicles[vid]
                            break

    def close(self):
        pass
