"""
AutoFlow RL - TraCI HTTP Telemetry Bridge
Exposes a lightweight local HTTP server allowing the AutoFlow RL web dashboard
to connect to real-time SUMO TraCI simulation steps when running locally.
"""

import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, ".."))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulation.sumo_utils import (
    find_sumo_config, get_sumo_binary, get_state, get_reward,
    apply_action_with_guard, compute_its_metrics, DETECTORS,
    TRAFFIC_LIGHT_ID, MIN_GREEN_STEPS, MockTraCI
)

class TraCISimulationState:
    def __init__(self):
        self.traci = None
        self.is_mock = True
        self.step = 0
        self.last_switch_step = -MIN_GREEN_STEPS
        self.queue_history = []
        self.travel_times = []
        self.depart_times = {}
        self.init_simulation()

    def init_simulation(self):
        binary, err = get_sumo_binary()
        cfg = find_sumo_config()
        if binary and cfg:
            try:
                import traci
                traci.start([binary, "-c", cfg, "--no-step-log", "true"])
                self.traci = traci
                self.is_mock = False
                print("[BRIDGE] Connected to native Eclipse SUMO binary via TraCI.")
                return
            except Exception as e:
                print(f"[BRIDGE] Falling back to MockTraCI: {e}")

        self.traci = MockTraCI(seed=42)
        self.is_mock = True
        print("[BRIDGE] Running high-fidelity MockTraCI in-memory simulator.")

    def step_sim(self, action=0):
        self.step += 1
        self.last_switch_step, switched = apply_action_with_guard(
            self.traci, action, self.step, self.last_switch_step, min_green=MIN_GREEN_STEPS
        )
        self.traci.simulationStep()
        state = get_state(self.traci, TRAFFIC_LIGHT_ID)
        reward = get_reward(state)
        total_q = sum(state[:-1])
        self.queue_history.append(total_q)

        detector_map = {d: int(state[i]) for i, d in enumerate(DETECTORS)}
        return {
            "step": self.step,
            "phase": int(state[-1]),
            "reward": reward,
            "total_queue": total_q,
            "detectors": detector_map,
            "action_taken": action,
            "switched": switched,
            "is_mock": self.is_mock
        }

sim_instance = TraCISimulationState()

class BridgeHTTPHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            resp = {
                "status": "online",
                "is_mock": sim_instance.is_mock,
                "step": sim_instance.step,
                "detectors": DETECTORS,
                "traffic_light": TRAFFIC_LIGHT_ID
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/step":
            query = urllib.parse.parse_qs(parsed.query)
            action = int(query.get("action", [0])[0])
            result = sim_instance.step_sim(action)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_bridge_server(port=8000):
    server = HTTPServer(("0.0.0.0", port), BridgeHTTPHandler)
    print(f"[BRIDGE] AutoFlow RL TraCI HTTP Bridge running at http://localhost:{port}")
    print("[BRIDGE] Press Ctrl+C to terminate.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if sim_instance.traci:
            try:
                sim_instance.traci.close()
            except Exception:
                pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_bridge_server(port)
