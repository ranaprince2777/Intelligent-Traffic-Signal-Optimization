"""
Tests for Tabular Q-Learning Controller
Validates Q-table initialization, Bellman update equation, epsilon-greedy action selection,
safety lock integration, and state persistence.
"""

import os
import numpy as np
import pytest
from simulation.QL import TabularQLearningAgent, run_ql_controller

def test_q_agent_initialization():
    agent = TabularQLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, min_green=5)
    assert len(agent.q_table) == 0
    assert agent.alpha == 0.1
    assert agent.gamma == 0.9

def test_q_table_unseen_state_default():
    agent = TabularQLearningAgent()
    state = (1, 2, 0, 1, 0, 0, 0)
    q_vals = agent.get_q_values(state)
    assert isinstance(q_vals, np.ndarray)
    assert q_vals.shape == (2,)
    assert np.all(q_vals == 0.0)

def test_bellman_update_calculation():
    agent = TabularQLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.0)
    s = (1, 1, 1, 0, 0, 0, 0)
    s_next = (1, 0, 0, 0, 0, 0, 0)
    a = 1
    r = -3.0

    # Initialize next state Q-values
    agent.q_table[s_next] = np.array([4.0, 2.0], dtype=np.float32)  # max is 4.0
    agent.q_table[s] = np.array([0.0, 0.0], dtype=np.float32)       # old Q(s,1) = 0.0

    # Update: Q(s,1) <- 0.0 + 0.1 * [-3.0 + 0.9 * 4.0 - 0.0]
    # td_target = -3.0 + 3.6 = 0.6
    # Q(s,1) = 0.0 + 0.1 * (0.6 - 0.0) = 0.06
    agent.update(s, a, r, s_next)

    assert pytest.approx(agent.get_q_values(s)[1], rel=1e-4) == 0.06

def test_min_green_lock_in_choose_action():
    agent = TabularQLearningAgent(min_green=5)
    agent.last_switch_step = 10
    state = (5, 5, 5, 0, 0, 0, 0)

    # Step 12: only 2 seconds since switch -> must force action 0 (keep)
    action = agent.choose_action(state, current_step=12, explore=True)
    assert action == 0

    # Step 14: only 4 seconds since switch -> still locked
    action = agent.choose_action(state, current_step=14, explore=True)
    assert action == 0

def test_q_table_serialization(tmp_path):
    agent = TabularQLearningAgent()
    s1 = (1, 0, 0, 0, 0, 0, 0)
    s2 = (2, 2, 2, 1, 1, 1, 2)
    agent.q_table[s1] = np.array([1.5, -2.0], dtype=np.float32)
    agent.q_table[s2] = np.array([0.0, 3.5], dtype=np.float32)

    save_path = tmp_path / "test_q_table.json"
    agent.save_q_table(str(save_path))
    assert os.path.isfile(save_path)

    new_agent = TabularQLearningAgent()
    loaded = new_agent.load_q_table(str(save_path))
    assert loaded is True
    assert len(new_agent.q_table) == 2
    assert np.allclose(new_agent.get_q_values(s1), [1.5, -2.0])
    assert np.allclose(new_agent.get_q_values(s2), [0.0, 3.5])

def test_run_ql_controller_mock_execution():
    metrics = run_ql_controller(total_steps=50, step_length=1.0, use_mock=True)
    assert metrics is not None
    assert metrics["controller"] == "Tabular Q-Learning"
    assert metrics["steps"] == 50
    assert metrics["avg_queue"] >= 0
    assert metrics["total_delay"] >= 0
