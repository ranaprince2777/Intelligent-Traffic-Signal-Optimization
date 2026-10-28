"""
Tests for sumo_utils.py
Validates path resolution, binary lookup error handling, state extraction,
reward calculation, min-green safety guard, and ITS metrics computation.
"""

import pytest
import numpy as np
from simulation.sumo_utils import (
    find_sumo_config, get_sumo_binary, get_state, get_reward,
    apply_action_with_guard, compute_its_metrics, MockTraCI,
    DETECTORS, TRAFFIC_LIGHT_ID, MIN_GREEN_STEPS
)

def test_find_sumo_config():
    cfg = find_sumo_config()
    assert cfg is not None
    assert cfg.endswith("RL.sumocfg")

def test_get_sumo_binary_error_handling(monkeypatch):
    # Temporarily clear SUMO_HOME and PATH to test error message
    monkeypatch.delenv("SUMO_HOME", raising=False)
    monkeypatch.setenv("PATH", "")
    binary, err = get_sumo_binary()
    assert binary is None
    assert "SUMO_HOME" in err
    assert "https://sumo.dlr.de/docs/Installing/index.html" in err

def test_state_vector_construction():
    mock = MockTraCI()
    state = get_state(mock, TRAFFIC_LIGHT_ID)
    assert isinstance(state, tuple)
    assert len(state) == 7  # 6 detectors + 1 phase index
    assert all(isinstance(v, int) for v in state)

def test_reward_calculation():
    # State with queues: [2, 1, 0, 3, 1, 1, 0] -> sum = 8 -> reward = -8.0
    state = (2, 1, 0, 3, 1, 1, 0)
    reward = get_reward(state)
    assert reward == -8.0

    zero_state = (0, 0, 0, 0, 0, 0, 2)
    assert get_reward(zero_state) == 0.0

def test_min_green_safety_guard():
    mock = MockTraCI()
    mock.phase = 0
    last_switch = 0
    min_green = 5

    # 1. Action 0: Keep phase -> should not switch
    new_last, switched = apply_action_with_guard(mock, action=0, current_step=3,
                                                  last_switch_step=last_switch, min_green=min_green)
    assert not switched
    assert new_last == 0
    assert mock.phase == 0

    # 2. Action 1: Switch attempted at step 3 (< 5s since switch) -> BLOCKED by min-green guard
    new_last, switched = apply_action_with_guard(mock, action=1, current_step=3,
                                                  last_switch_step=last_switch, min_green=min_green)
    assert not switched
    assert new_last == 0
    assert mock.phase == 0

    # 3. Action 1: Switch attempted at step 5 (>= 5s since switch) -> ALLOWED
    new_last, switched = apply_action_with_guard(mock, action=1, current_step=5,
                                                  last_switch_step=last_switch, min_green=min_green)
    assert switched
    assert new_last == 5
    assert mock.phase == 1  # Transitioned from 0 to 1

def test_compute_its_metrics_normal():
    queues = [5, 4, 3, 6, 2]
    travel_times = [10.0, 15.0, 20.0, 25.0]
    metrics = compute_its_metrics(queues, travel_times, step_length=1.0)

    assert metrics["total_delay"] == 20.0
    assert metrics["avg_queue"] == 4.0
    assert metrics["max_queue"] == 6
    assert metrics["throughput"] == 4
    assert metrics["avg_travel_time"] == 17.5
    assert metrics["max_travel_time"] == 25.0

def test_compute_its_metrics_empty():
    metrics = compute_its_metrics([], [], step_length=1.0)
    assert metrics["total_delay"] == 0.0
    assert metrics["avg_queue"] == 0.0
    assert metrics["max_queue"] == 0
    assert metrics["throughput"] == 0
    assert metrics["avg_travel_time"] == 0.0
