"""
AutoFlow RL - Deep Q-Network (DQN) Controller Entrypoint
Runs neural reinforcement learning traffic signal agent.
Delegates to simulation.DQN.run_dqn_controller.
"""

import sys
import argparse
from simulation.DQN import run_dqn_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Q-Network (DQN) Signal Controller")
    parser.add_argument("--steps", type=int, default=3000, help="Total simulation steps (default: 3000)")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI instead of headless")
    parser.add_argument("--mock", action="store_true", help="Force mock environment for testing")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    args = parser.parse_args()

    run_dqn_controller(total_steps=args.steps, use_gui=args.gui, use_mock=args.mock, eval_mode=args.eval)
