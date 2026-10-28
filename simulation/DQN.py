"""
AutoFlow RL - Deep Q-Network (DQN) Traffic Signal Controller
Approximates Q(s, a) using a multi-layer perceptron neural network with experience replay,
min-green safety constraints, and automated fallback to pure-NumPy when TensorFlow is not installed.

Network Architecture:
- Input Layer:  7 features (6 lane detector counts + traffic light phase)
- Hidden Layer 1: 24 units (ReLU activation)
- Hidden Layer 2: 24 units (ReLU activation)
- Output Layer:   2 units (Linear Q-values for Action 0: Keep, Action 1: Switch)

Safety Constraint:
- Minimum green time: 5 seconds before phase transition
"""

import os
import sys
import json
import random
import argparse
from collections import deque
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

# -------------------------------------------------------------
# Pure-NumPy Neural Network Engine (Portable Fallback)
# -------------------------------------------------------------
class NumpyDQNModel:
    """Portable 2-layer MLP for Q-function approximation using pure NumPy."""
    def __init__(self, state_size=7, action_size=2, hidden_size=24, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr

        # He initialization for ReLU layers
        self.W1 = np.random.randn(state_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros((1, hidden_size), dtype=np.float32)

        self.W2 = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size), dtype=np.float32)

        self.W3 = np.random.randn(hidden_size, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, action_size), dtype=np.float32)

    def forward(self, x):
        """Forward pass with ReLU activations."""
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)

        self.out = np.dot(self.a2, self.W3) + self.b3
        return self.out

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    def train_on_batch(self, x, target_q):
        """Backpropagation with Mean Squared Error loss."""
        batch_size = x.shape[0]
        preds = self.forward(x)
        d_out = (preds - target_q) / batch_size  # gradient of MSE w.r.t output

        dW3 = np.dot(self.a2.T, d_out)
        db3 = np.sum(d_out, axis=0, keepdims=True)

        da2 = np.dot(d_out, self.W3.T)
        dz2 = da2 * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # SGD parameter update with learning rate
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def save_weights(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        weights = {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W3": self.W3.tolist(), "b3": self.b3.tolist()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weights, f)
        print(f"DQN weights saved to: {filepath}")

    def load_weights(self, filepath):
        if not os.path.isfile(filepath):
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            weights = json.load(f)
        self.W1 = np.array(weights["W1"], dtype=np.float32)
        self.b1 = np.array(weights["b1"], dtype=np.float32)
        self.W2 = np.array(weights["W2"], dtype=np.float32)
        self.b2 = np.array(weights["b2"], dtype=np.float32)
        self.W3 = np.array(weights["W3"], dtype=np.float32)
        self.b3 = np.array(weights["b3"], dtype=np.float32)
        print(f"Loaded DQN weights from: {filepath}")
        return True

# -------------------------------------------------------------
# Deep Q-Network Agent with Experience Replay
# -------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_size=7, action_size=2, gamma=0.9, epsilon=0.1, lr=0.001, min_green=MIN_GREEN_STEPS):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_green = min_green
        self.last_switch_step = -min_green

        self.memory = deque(maxlen=2000)
        self.batch_size = 32

        # Check if TensorFlow is available, otherwise use NumpyDQNModel
        self.use_tf = False
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            self.model = keras.Sequential([
                layers.Input(shape=(state_size,)),
                layers.Dense(24, activation='relu'),
                layers.Dense(24, activation='relu'),
                layers.Dense(action_size, activation='linear')
            ])
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
            self.use_tf = True
            print("DQN Backend: TensorFlow / Keras")
        except Exception:
            self.model = NumpyDQNModel(state_size, action_size, hidden_size=24, lr=lr)
            self.use_tf = False
            print("DQN Backend: Pure-NumPy (Ultra-Fast Lightweight Engine)")

    def choose_action(self, state, current_step, explore=True):
        """Epsilon-greedy action selection with safety lock."""
        # Min-green lock: illegal to switch before 5 seconds
        if current_step - self.last_switch_step < self.min_green:
            return 0

        if explore and random.random() < self.epsilon:
            return random.choice([0, 1])

        state_arr = np.array(state, dtype=np.float32).reshape(1, -1)
        if self.use_tf:
            q_values = self.model.predict(state_arr, verbose=0)[0]
        else:
            q_values = self.model.predict(state_arr)[0]
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state):
        self.memory.append((
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32)
        ))

    def replay(self):
        """Train neural network on batch sampled from experience replay."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = [m[1] for m in minibatch]
        rewards = [m[2] for m in minibatch]
        next_states = np.array([m[3] for m in minibatch])

        if self.use_tf:
            current_qs = self.model.predict(states, verbose=0)
            next_qs = self.model.predict(next_states, verbose=0)
            for i in range(self.batch_size):
                target = rewards[i] + self.gamma * np.max(next_qs[i])
                current_qs[i][actions[i]] = target
            self.model.fit(states, current_qs, verbose=0, batch_size=self.batch_size)
        else:
            current_qs = self.model.predict(states)
            next_qs = self.model.predict(next_states)
            for i in range(self.batch_size):
                target = rewards[i] + self.gamma * np.max(next_qs[i])
                current_qs[i][actions[i]] = target
            self.model.train_on_batch(states, current_qs)

    def save(self, filepath):
        if self.use_tf:
            self.model.save(filepath)
        else:
            self.model.save_weights(filepath)

    def load(self, filepath):
        if self.use_tf:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(filepath)
        else:
            self.model.load_weights(filepath)

def run_dqn_controller(total_steps=3000, step_length=1.0, use_gui=False, use_mock=False, eval_mode=False):
    """
    Executes the Deep Q-Network controller.
    """
    print("\n=======================================================")
    print("[DQN] RUNNING DEEP Q-NETWORK (DQN) CONTROLLER")
    print("=======================================================")
    print(f"Total Steps:   {total_steps}")
    print(f"Step Length:   {step_length}s")
    print(f"Min Green Lock:{MIN_GREEN_STEPS}s")
    print(f"Evaluation Mode:{eval_mode}")

    agent = DQNAgent(gamma=0.9, epsilon=0.0 if eval_mode else 0.1)

    out_dir = os.path.join(curr_dir, "results")
    weights_path = os.path.join(out_dir, "dqn_weights.json")
    if eval_mode and os.path.isfile(weights_path):
        agent.load(weights_path)

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
        print("Connected to simulation environment. Running DQN training loop...")

        current_state = get_state(traci, TRAFFIC_LIGHT_ID)

        for step in range(1, total_steps + 1):
            sim_time = step * step_length

            # 1. Action selection via DQN inference
            action = agent.choose_action(current_state, step, explore=(not eval_mode))

            # 2. Enforce safety constraint and apply action
            agent.last_switch_step, was_switched = apply_action_with_guard(
                traci, action, step, agent.last_switch_step,
                min_green=MIN_GREEN_STEPS, tls_id=TRAFFIC_LIGHT_ID
            )

            # 3. Advance simulation
            traci.simulationStep()

            # 4. Observe new state and reward
            next_state = get_state(traci, TRAFFIC_LIGHT_ID)
            reward = get_reward(next_state)
            total_queue = sum(next_state[:-1])

            # 5. Experience replay & learning
            if not eval_mode:
                agent.remember(current_state, action, reward, next_state)
                if step % 4 == 0:  # Train periodically to maintain high simulation speed
                    agent.replay()
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
                print(f"Step {step:4d}/{total_steps} | Queue: {total_queue:2d} | Avg Queue: {avg_q:5.2f} | Buffer: {len(agent.memory)} | Completed: {len(travel_times)}")

    except Exception as e:
        print(f"[ERROR] Simulation interrupted: {e}")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    # Save weights
    if not eval_mode:
        agent.save(weights_path)

    # Compute ITS metrics
    metrics = compute_its_metrics(queue_history, travel_times, step_length)
    metrics["controller"] = "Deep Q-Network (DQN)"
    metrics["steps"] = total_steps
    metrics["replay_buffer_size"] = len(agent.memory)

    print("\n-------------------------------------------------------")
    print("[METRICS] DEEP Q-NETWORK (DQN) RESULTS SUMMARY:")
    print(f"  Average Queue:      {metrics['avg_queue']} vehicles")
    print(f"  Maximum Queue:      {metrics['max_queue']} vehicles")
    print(f"  Total Delay:        {metrics['total_delay']} vehicle-seconds")
    print(f"  Throughput:         {metrics['throughput']} vehicles completed")
    print(f"  Average Travel Time:{metrics['avg_travel_time']} s")
    print(f"  P90 Travel Time:    {metrics['p90_travel_time']} s")
    print(f"  Max Travel Time:    {metrics['max_travel_time']} s")
    print("-------------------------------------------------------")

    # Persist outputs
    json_path = os.path.join(out_dir, "dqn_metrics.json")
    csv_path = os.path.join(out_dir, "dqn_telemetry.csv")
    save_metrics_to_json(metrics, json_path)
    save_telemetry_to_csv(step_history, time_history, queue_history, reward_history, csv_path)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Q-Network (DQN) Traffic Signal Controller")
    parser.add_argument("--steps", type=int, default=3000, help="Total simulation steps (default: 3000)")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI instead of headless")
    parser.add_argument("--mock", action="store_true", help="Force mock environment for testing")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    args = parser.parse_args()

    run_dqn_controller(total_steps=args.steps, use_gui=args.gui, use_mock=args.mock, eval_mode=args.eval)
