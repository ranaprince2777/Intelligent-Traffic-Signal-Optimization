"""
Tests for Benchmark Data Consistency and Relative Performance Metrics
Ensures verified benchmark report matches Table 5.1 & Table 5.2 of the project report and README.
"""

import os
import json
import pytest

def test_verified_benchmarks_structure():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    bench_file = os.path.join(curr_dir, "..", "simulation", "results", "verified_benchmark_report.json")
    assert os.path.isfile(bench_file), f"Verified benchmark file not found: {bench_file}"

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data["models"]
    assert "static_baseline" in models
    assert "tabular_q_learning" in models
    assert "deep_q_network" in models

    # Check Static Baseline values
    sb = models["static_baseline"]
    assert sb["avg_queue"] == 7.77
    assert sb["max_queue"] == 12
    assert sb["total_delay"] == 23000.0
    assert sb["throughput"] == 1929
    assert sb["avg_travel_time"] == 18.4
    assert sb["max_travel_time"] == 1332.0

    # Check Tabular Q-Learning values
    ql = models["tabular_q_learning"]
    assert ql["avg_queue"] == 4.82
    assert ql["max_queue"] == 8
    assert ql["total_delay"] == 14030.0
    assert ql["throughput"] == 2368
    assert ql["avg_travel_time"] == 11.8
    assert ql["max_travel_time"] == 573.0

    # Check Deep Q-Network values
    dqn = models["deep_q_network"]
    assert dqn["avg_queue"] == 5.05
    assert dqn["max_queue"] == 8
    assert dqn["total_delay"] == 14720.0
    assert dqn["throughput"] == 2303
    assert dqn["avg_travel_time"] == 12.1
    assert dqn["max_travel_time"] == 612.0

def test_relative_improvement_math():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    bench_file = os.path.join(curr_dir, "..", "simulation", "results", "verified_benchmark_report.json")
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sb = data["models"]["static_baseline"]
    ql = data["models"]["tabular_q_learning"]
    dqn = data["models"]["deep_q_network"]

    # Q-Learning queue reduction: (7.77 - 4.82) / 7.77 * 100 = 37.966% ~ 38.0%
    ql_q_red = round(((sb["avg_queue"] - ql["avg_queue"]) / sb["avg_queue"]) * 100, 1)
    assert ql_q_red == 38.0

    # Q-Learning delay reduction: (23000 - 14030) / 23000 * 100 = 39.0%
    ql_delay_red = round(((sb["total_delay"] - ql["total_delay"]) / sb["total_delay"]) * 100, 1)
    assert ql_delay_red == 39.0

    # Q-Learning throughput gain: (2368 - 1929) / 1929 * 100 = 22.757% ~ 22.8%
    ql_tp_gain = round(((ql["throughput"] - sb["throughput"]) / sb["throughput"]) * 100, 1)
    assert ql_tp_gain == 22.8

    # DQN queue reduction: (7.77 - 5.05) / 7.77 * 100 = 35.0%
    dqn_q_red = round(((sb["avg_queue"] - dqn["avg_queue"]) / sb["avg_queue"]) * 100, 1)
    assert dqn_q_red == 35.0

    # DQN delay reduction: (23000 - 14720) / 23000 * 100 = 36.0%
    dqn_delay_red = round(((sb["total_delay"] - dqn["total_delay"]) / sb["total_delay"]) * 100, 1)
    assert dqn_delay_red == 36.0
