"""
AutoFlow RL - Static Fixed-Time Baseline Controller Entrypoint
Runs deterministic signal timing baseline.
Delegates to simulation.Static.run_static_controller.
"""

import sys
import argparse
from simulation.Static import run_static_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static Fixed-Time Signal Controller")
    parser.add_argument("--steps", type=int, default=3000, help="Total simulation steps (default: 3000)")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI instead of headless")
    parser.add_argument("--mock", action="store_true", help="Force mock environment for testing")
    args = parser.parse_args()

    run_static_controller(total_steps=args.steps, use_gui=args.gui, use_mock=args.mock)
