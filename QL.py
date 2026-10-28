"""
AutoFlow RL - Tabular Q-Learning Controller Entrypoint
Runs reinforcement learning tabular Q-learning agent.
Delegates to simulation.QL.run_ql_controller.
"""

import sys
import argparse
from simulation.QL import run_ql_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Q-Learning Signal Controller")
    parser.add_argument("--steps", type=int, default=3000, help="Total simulation steps (default: 3000)")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI instead of headless")
    parser.add_argument("--mock", action="store_true", help="Force mock environment for testing")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    args = parser.parse_args()

    run_ql_controller(total_steps=args.steps, use_gui=args.gui, use_mock=args.mock, eval_mode=args.eval)
