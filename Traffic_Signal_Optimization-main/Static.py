"""
AutoFlow RL - Static / Fixed-Time Traffic Signal Controller Baseline
Runs a deterministic, pre-timed signal cycle in SUMO without adaptive feedback.
Serves as the benchmark baseline against Tabular Q-Learning and Deep Q-Networks.

Phase Configuration:
- Phase 0: Eastbound Green  (42s)
- Phase 1: Eastbound Yellow (3s)
- Phase 2: Southbound Green (42s)
- Phase 3: Southbound Yellow(3s)
Total cycle time: 90s
"""

import os
import sys
import argparse
import numpy as np

# Add parent and simulation directory to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, ".."))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from simulation.sumo_utils import (
    find_sumo_config, get_sumo_binary, safe_get_detector,
    safe_get_phase, get_state, get_reward, compute_its_metrics,
    save_metrics_to_json, save_telemetry_to_csv, DETECTORS,
    TRAFFIC_LIGHT_ID, DEFAULT_PHASE_DURATIONS, MockTraCI
)

def run_static_controller(total_steps=3000, step_length=1.0, use_gui=False, use_mock=False):
    """
    Executes the static/fixed-time baseline controller.
    """
    print("\n=======================================================")
    print("[STATIC] RUNNING STATIC / FIXED-TIME BASELINE CONTROLLER")
    print("=======================================================")
    print(f"Total Steps:   {total_steps}")
    print(f"Step Length:   {step_length}s")
    print(f"Phase Cycle:   {DEFAULT_PHASE_DURATIONS} (Total: {sum(DEFAULT_PHASE_DURATIONS)}s)")

    if use_mock:
        print("Mode: In-Memory MockTraCI (Standalone Verification)")
        traci = MockTraCI(seed=42)
        sumo_config_path = "mock_config"
    else:
        sumo_binary, error = get_sumo_binary(use_gui=use_gui)
        if error:
            print(f"\n[WARNING] {error}")
            print("Falling back to verified MockTraCI simulation for demonstration...")
            traci = MockTraCI(seed=42)
            sumo_config_path = "mock_config"
        else:
            sumo_config_path = find_sumo_config()
            if not sumo_config_path:
                print("[ERROR] RL.sumocfg could not be found.")
                return None

            try:
                import traci
            except ImportError:
                print("[WARNING] TraCI module not found in Python environment. Using MockTraCI.")
                traci = MockTraCI(seed=42)

    sumo_cmd = [
        sumo_binary if not isinstance(traci, MockTraCI) else "sumo",
        "-c", sumo_config_path,
        "--step-length", str(step_length),
        "--lateral-resolution", "0",
        "--no-step-log", "true",
        "--time-to-teleport", "-1"
    ]

    step_history = []
    time_history = []
    queue_history = []
    reward_history = []
    depart_times = {}
    travel_times = []

    try:
        traci.start(sumo_cmd)
        print("Connected to simulation environment. Running baseline steps...")

        current_phase_idx = 0
        phase_time_elapsed = 0

        for step in range(1, total_steps + 1):
            # Fixed-time phase switching logic
            phase_time_elapsed += step_length
            current_target_duration = DEFAULT_PHASE_DURATIONS[current_phase_idx]

            if phase_time_elapsed >= current_target_duration:
                current_phase_idx = (current_phase_idx + 1) % len(DEFAULT_PHASE_DURATIONS)
                phase_time_elapsed = 0
                try:
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, current_phase_idx)
                except Exception:
                    pass

            # Advance simulation by one step
            traci.simulationStep()
            sim_time = step * step_length

            # State & Reward calculation
            state = get_state(traci, TRAFFIC_LIGHT_ID)
            total_queue = sum(state[:-1])
            reward = get_reward(state)

            # Record telemetry
            step_history.append(step)
            time_history.append(sim_time)
            queue_history.append(total_queue)
            reward_history.append(reward)

            # Track vehicle departures and arrivals for throughput and travel time
            try:
                for vid in traci.vehicle.getDepartedIDList():
                    depart_times[vid] = sim_time
                for vid in traci.vehicle.getArrivedIDList():
                    if vid in depart_times:
                        tt = sim_time - depart_times[vid]
                        travel_times.append(tt)
                        del depart_times[vid]
            except Exception:
                pass

            if step % 500 == 0 or step == total_steps:
                avg_q_so_far = sum(queue_history) / len(queue_history)
                print(f"Step {step:4d}/{total_steps} | Instant Queue: {total_queue:2d} | Avg Queue: {avg_q_so_far:5.2f} | Completed: {len(travel_times)}")

    except Exception as e:
        print(f"[ERROR] Simulation interrupted: {e}")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    # Compute standardized ITS metrics
    metrics = compute_its_metrics(queue_history, travel_times, step_length)
    metrics["controller"] = "Static Fixed-Time Baseline"
    metrics["steps"] = total_steps

    print("\n-------------------------------------------------------")
    print("[METRICS] STATIC BASELINE RESULTS SUMMARY:")
    print(f"  Average Queue:      {metrics['avg_queue']} vehicles")
    print(f"  Maximum Queue:      {metrics['max_queue']} vehicles")
    print(f"  Total Delay:        {metrics['total_delay']} vehicle-seconds")
    print(f"  Throughput:         {metrics['throughput']} vehicles completed")
    print(f"  Average Travel Time:{metrics['avg_travel_time']} s")
    print(f"  P90 Travel Time:    {metrics['p90_travel_time']} s")
    print(f"  Max Travel Time:    {metrics['max_travel_time']} s")
    print("-------------------------------------------------------")

    # Persist outputs
    out_dir = os.path.join(curr_dir, "results")
    json_path = os.path.join(out_dir, "static_metrics.json")
    csv_path = os.path.join(out_dir, "static_telemetry.csv")
    save_metrics_to_json(metrics, json_path)
    save_telemetry_to_csv(step_history, time_history, queue_history, reward_history, csv_path)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static Fixed-Time Signal Controller")
    parser.add_argument("--steps", type=int, default=3000, help="Total simulation steps (default: 3000)")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI instead of headless")
    parser.add_argument("--mock", action="store_true", help="Force mock environment for testing")
    args = parser.parse_args()

    run_static_controller(total_steps=args.steps, use_gui=args.gui, use_mock=args.mock)
