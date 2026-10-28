"""
AutoFlow RL - Unified Experiment & Benchmark Harness
Runs Static, Tabular Q-Learning, and Deep Q-Network controllers under identical conditions,
aggregates standardized ITS metrics, computes percentage improvements, and exports
structured comparison tables to CSV, JSON, and Markdown.
"""

import os
import sys
import json
import csv
import argparse
import pandas as pd

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, ".."))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from simulation.Static import run_static_controller
from simulation.QL import run_ql_controller
from simulation.DQN import run_dqn_controller

def compute_relative_gain(val, baseline, lower_is_better=True):
    """Computes relative percentage difference against baseline."""
    if baseline == 0:
        return 0.0
    diff = ((val - baseline) / baseline) * 100.0
    return round(diff, 2)

def run_all_experiments(steps=3000, step_length=1.0, use_gui=False, use_mock=False):
    print("\n=================================================================")
    print("[RUN] AUTOFLOW RL: MULTI-MODEL COMPARATIVE EXPERIMENT SUITE")
    print("=================================================================")
    print(f"Simulation Steps per Model: {steps}")
    print(f"Step Length:                {step_length}s")
    print(f"Force Mock Environment:     {use_mock}\n")

    # 1. Run Static Controller Baseline
    print("\n[1/3] Executing Static Baseline...")
    static_metrics = run_static_controller(total_steps=steps, step_length=step_length, use_gui=use_gui, use_mock=use_mock)

    # 2. Run Tabular Q-Learning Controller
    print("\n[2/3] Executing Tabular Q-Learning Controller...")
    ql_metrics = run_ql_controller(total_steps=steps, step_length=step_length, use_gui=use_gui, use_mock=use_mock)

    # 3. Run Deep Q-Network (DQN) Controller
    print("\n[3/3] Executing Deep Q-Network (DQN) Controller...")
    dqn_metrics = run_dqn_controller(total_steps=steps, step_length=step_length, use_gui=use_gui, use_mock=use_mock)

    # Aggregated Comparison Table
    results = {
        "static": static_metrics,
        "ql": ql_metrics,
        "dqn": dqn_metrics,
        "relative_comparison": {
            "ql_vs_static": {
                "avg_queue_diff_pct": compute_relative_gain(ql_metrics["avg_queue"], static_metrics["avg_queue"]),
                "max_queue_diff_pct": compute_relative_gain(ql_metrics["max_queue"], static_metrics["max_queue"]),
                "total_delay_diff_pct": compute_relative_gain(ql_metrics["total_delay"], static_metrics["total_delay"]),
                "throughput_diff_pct": compute_relative_gain(ql_metrics["throughput"], static_metrics["throughput"], lower_is_better=False),
                "avg_travel_time_diff_pct": compute_relative_gain(ql_metrics["avg_travel_time"], static_metrics["avg_travel_time"]),
                "max_travel_time_diff_pct": compute_relative_gain(ql_metrics["max_travel_time"], static_metrics["max_travel_time"]),
            },
            "dqn_vs_static": {
                "avg_queue_diff_pct": compute_relative_gain(dqn_metrics["avg_queue"], static_metrics["avg_queue"]),
                "max_queue_diff_pct": compute_relative_gain(dqn_metrics["max_queue"], static_metrics["max_queue"]),
                "total_delay_diff_pct": compute_relative_gain(dqn_metrics["total_delay"], static_metrics["total_delay"]),
                "throughput_diff_pct": compute_relative_gain(dqn_metrics["throughput"], static_metrics["throughput"], lower_is_better=False),
                "avg_travel_time_diff_pct": compute_relative_gain(dqn_metrics["avg_travel_time"], static_metrics["avg_travel_time"]),
                "max_travel_time_diff_pct": compute_relative_gain(dqn_metrics["max_travel_time"], static_metrics["max_travel_time"]),
            }
        }
    }

    out_dir = os.path.join(curr_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    # Save comparison JSON
    json_path = os.path.join(out_dir, "metrics_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save comparison CSV
    csv_path = os.path.join(out_dir, "metrics_comparison.csv")
    rows = [
        ["Metric", "Static Baseline", "Tabular Q-Learning", "DQN", "QL vs Static (%)", "DQN vs Static (%)"],
        ["Average Queue (veh)", static_metrics["avg_queue"], ql_metrics["avg_queue"], dqn_metrics["avg_queue"],
         f"{results['relative_comparison']['ql_vs_static']['avg_queue_diff_pct']:+}%",
         f"{results['relative_comparison']['dqn_vs_static']['avg_queue_diff_pct']:+}%"],
        ["Maximum Queue (veh)", static_metrics["max_queue"], ql_metrics["max_queue"], dqn_metrics["max_queue"],
         f"{results['relative_comparison']['ql_vs_static']['max_queue_diff_pct']:+}%",
         f"{results['relative_comparison']['dqn_vs_static']['max_queue_diff_pct']:+}%"],
        ["Total Delay (veh-s)", static_metrics["total_delay"], ql_metrics["total_delay"], dqn_metrics["total_delay"],
         f"{results['relative_comparison']['ql_vs_static']['total_delay_diff_pct']:+}%",
         f"{results['relative_comparison']['dqn_vs_static']['total_delay_diff_pct']:+}%"],
        ["Throughput (veh)", static_metrics["throughput"], ql_metrics["throughput"], dqn_metrics["throughput"],
         f"{results['relative_comparison']['ql_vs_static']['throughput_diff_pct']:+}%",
         f"{results['relative_comparison']['dqn_vs_static']['throughput_diff_pct']:+}%"],
        ["Average Travel Time (s)", static_metrics["avg_travel_time"], ql_metrics["avg_travel_time"], dqn_metrics["avg_travel_time"],
         f"{results['relative_comparison']['ql_vs_static']['avg_travel_time_diff_pct']:+}%",
         f"{results['relative_comparison']['dqn_vs_static']['avg_travel_time_diff_pct']:+}%"],
        ["Max Travel Time (s)", static_metrics["max_travel_time"], ql_metrics["max_travel_time"], dqn_metrics["max_travel_time"],
         f"{results['relative_comparison']['ql_vs_static']['max_travel_time_diff_pct']:+}%",
         f"{results['relative_comparison']['dqn_vs_static']['max_travel_time_diff_pct']:+}%"],
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("\n=================================================================")
    print("[RESULTS] CONSOLIDATED EXPERIMENTAL COMPARISON RESULTS:")
    print("=================================================================")
    df = pd.DataFrame(rows[1:], columns=rows[0])
    print(df.to_string(index=False))
    print(f"\nSaved structured comparison to: {json_path}")
    print(f"Saved structured CSV to:        {csv_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Simulation Benchmark Runner")
    parser.add_argument("--steps", type=int, default=3000, help="Simulation steps per model")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--mock", action="store_true", help="Force mock environment")
    args = parser.parse_args()

    run_all_experiments(steps=args.steps, use_gui=args.gui, use_mock=args.mock)
