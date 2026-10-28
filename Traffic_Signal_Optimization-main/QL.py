"""
AutoFlow RL - Tabular Q-Learning Traffic Signal Controller
Adaptive signal optimization using tabular Q-learning with discrete state encoding,
Bellman optimality updates, and min-green safety guards.

State Space:
- 7 discrete values: (EB_0, EB_1, EB_2, SB_0, SB_1, SB_2, current_phase)
Action Space:
- 0: Keep current phase
- 1: Switch to next phase
Reward Function:
- R = -(Sum of all queued vehicles across the 6 lane detectors)
Safety Constraint:
- Minimum green time: 5 seconds before phase transition
"""

import os
import sys
import json
import random
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
    find_sumo_config, get_sumo_binary, get_state, get_reward,
    apply_action_with_guard, compute_its_metrics, save_metrics_to_json,
    save_telemetry_to_csv, DETECTORS, TRAFFIC_LIGHT_ID,
    MIN_GREEN_STEPS, MockTraCI
)

class TabularQLearningAgent:
    """Tabular Q-Learning Agent with min-green safety constraint."""
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, min_green=MIN_GREEN_STEPS):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_green = min_green
        self.q_table = {}  # state_tuple -> np.ndarray([q0, q1])
        self.last_switch_step = -min_green

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(2, dtype=np.float32)
        return self.q_table[state]

    def choose_action(self, state, current_step, explore=True):
        """Epsilon-greedy action selection with safety lock awareness."""
        # If min-green lock is active, switching is illegal -> must keep (action 0)
        if current_step - self.last_switch_step < self.min_green:
            return 0

        if explore and random.random() < self.epsilon:
            return random.choice([0, 1])

        q_vals = self.get_q_values(state)
        return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_state):
        """Bellman equation update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]"""
        old_q = self.get_q_values(state)[action]
        next_max_q = np.max(self.get_q_values(next_state))
        td_target = reward + self.gamma * next_max_q
        self.q_table[state][action] = old_q + self.alpha * (td_target - old_q)

    def save_q_table(self, filepath):
        """Serializes Q-table to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        serializable = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f)
        print(f"Q-Table ({len(self.q_table)} states) saved to: {filepath}")

    def load_q_table(self, filepath):
        """Loads Q-table from JSON if present."""
        if not os.path.isfile(filepath):
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.q_table = {eval(k): np.array(v, dtype=np.float32) for k, v in data.items()}
        print(f"Loaded Q-Table with {len(self.q_table)} states.")
        return True

def run_ql_controller(total_steps=3000, step_length=1.0, use_gui=False, use_mock=False, eval_mode=False):
    """
    Executes the Tabular Q-Learning controller.
    """
    print("\n=======================================================")
    print("[QL] RUNNING TABULAR Q-LEARNING CONTROLLER")
    print("=======================================================")
    print(f"Total Steps:   {total_steps}")
    print(f"Step Length:   {step_length}s")
    print(f"Min Green Lock:{MIN_GREEN_STEPS}s")
    print(f"Evaluation Mode:{eval_mode}")

    agent = TabularQLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.0 if eval_mode else 0.1)

    out_dir = os.path.join(curr_dir, "results")
    qtable_path = os.path.join(out_dir, "q_table.json")
    if eval_mode:
        agent.load_q_table(qtable_path)

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
        print("Connected to simulation environment. Running Q-learning loop...")

        current_state = get_state(traci, TRAFFIC_LIGHT_ID)

        for step in range(1, total_steps + 1):
            sim_time = step * step_length

            # 1. Action selection
            action = agent.choose_action(current_state, step, explore=(not eval_mode))

            # 2. Enforce safety constraint and apply action
            agent.last_switch_step, was_switched = apply_action_with_guard(
                traci, action, step, agent.last_switch_step,
                min_green=MIN_GREEN_STEPS, tls_id=TRAFFIC_LIGHT_ID
            )

            # 3. Step simulation
            traci.simulationStep()

            # 4. Observe new state and reward
            next_state = get_state(traci, TRAFFIC_LIGHT_ID)
            reward = get_reward(next_state)
            total_queue = sum(next_state[:-1])

            # 5. Q-learning update
            if not eval_mode:
                agent.update(current_state, action, reward, next_state)
            current_state = next_state

            # 6. Record telemetry
            step_history.append(step)
            time_history.append(sim_time)
            queue_history.append(total_queue)
            reward_history.append(reward)

            # Track vehicle departures and arrivals
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
                avg_q = sum(queue_history) / len(queue_history)
                print(f"Step {step:4d}/{total_steps} | Queue: {total_queue:2d} | Avg Queue: {avg_q:5.2f} | Unique States: {len(agent.q_table)} | Completed: {len(travel_times)}")

    except Exception as e:
        print(f"[ERROR] Simulation interrupted: {e}")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    # Save Q-table if training
    if not eval_mode and len(agent.q_table) > 0:
        agent.save_q_table(qtable_path)

    # Compute ITS metrics
    metrics = compute_its_metrics(queue_history, travel_times, step_length)
    metrics["controller"] = "Tabular Q-Learning"
    metrics["steps"] = total_steps
    metrics["unique_states_explored"] = len(agent.q_table)

    print("\n-------------------------------------------------------")
    print("[METRICS] TABULAR Q-LEARNING RESULTS SUMMARY:")
    print(f"  Average Queue:      {metrics['avg_queue']} vehicles")
    print(f"  Maximum Queue:      {metrics['max_queue']} vehicles")
    print(f"  Total Delay:        {metrics['total_delay']} vehicle-seconds")
    print(f"  Throughput:         {metrics['throughput']} vehicles completed")
    print(f"  Average Travel Time:{metrics['avg_travel_time']} s")
    print(f"  P90 Travel Time:    {metrics['p90_travel_time']} s")
    print(f"  Max Travel Time:    {metrics['max_travel_time']} s")
    print(f"  States in Q-Table:  {len(agent.q_table)}")
    print("-------------------------------------------------------")

    # Persist outputs
    json_path = os.path.join(out_dir, "ql_metrics.json")
    csv_path = os.path.join(out_dir, "ql_telemetry.csv")
    save_metrics_to_json(metrics, json_path)
    save_telemetry_to_csv(step_history, time_history, queue_history, reward_history, csv_path)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Q-Learning Traffic Signal Controller")
    parser.add_argument("--steps", type=int, default=3000, help="Total simulation steps (default: 3000)")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI instead of headless")
    parser.add_argument("--mock", action="store_true", help="Force mock environment for testing")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (epsilon=0, load Q-table)")
    args = parser.parse_args()

    run_ql_controller(total_steps=args.steps, use_gui=args.gui, use_mock=args.mock, eval_mode=args.eval)
