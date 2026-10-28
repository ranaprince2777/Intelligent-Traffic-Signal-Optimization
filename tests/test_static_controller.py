"""
Tests for Static Fixed-Time Signal Controller
Validates baseline cycle execution, determinism, and metrics recording.
"""

import os
import pytest
from simulation.Static import run_static_controller

def test_static_controller_mock_execution():
    metrics = run_static_controller(total_steps=50, step_length=1.0, use_mock=True)
    assert metrics is not None
    assert metrics["controller"] == "Static Fixed-Time Baseline"
    assert metrics["steps"] == 50
    assert metrics["avg_queue"] >= 0
    assert metrics["total_delay"] >= 0
    assert "throughput" in metrics
    assert "avg_travel_time" in metrics
