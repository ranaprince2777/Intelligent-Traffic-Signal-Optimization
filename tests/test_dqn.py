"""
Tests for Deep Q-Network (DQN) Controller
Validates network architecture, forward pass shape, experience replay,
Q-target calculation, weights persistence, and mock execution.
"""

import os
import numpy as np
import pytest
from simulation.DQN import NumpyDQNModel, DQNAgent, run_dqn_controller

def test_numpy_dqn_forward_pass_shape():
    model = NumpyDQNModel(state_size=7, action_size=2, hidden_size=24)
    dummy_input = np.random.randn(4, 7).astype(np.float32)
    output = model.forward(dummy_input)

    assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"
    assert not np.isnan(output).any(), "Output contains NaN values"

def test_numpy_dqn_backprop_loss_decrease():
    model = NumpyDQNModel(state_size=7, action_size=2, hidden_size=24, lr=0.01)
    x = np.random.randn(8, 7).astype(np.float32)
    target = np.random.randn(8, 2).astype(np.float32)

    # Initial MSE loss
    init_preds = model.forward(x)
    init_loss = np.mean((init_preds - target) ** 2)

    # Train for 20 iterations
    for _ in range(20):
        model.train_on_batch(x, target)

    final_preds = model.forward(x)
    final_loss = np.mean((final_preds - target) ** 2)

    assert final_loss < init_loss, f"Loss did not decrease: {init_loss} -> {final_loss}"

def test_dqn_agent_replay_buffer():
    agent = DQNAgent(state_size=7, action_size=2)
    assert len(agent.memory) == 0

    s = (1, 0, 0, 0, 0, 0, 0)
    s_next = (2, 0, 0, 0, 0, 0, 0)
    agent.remember(s, 1, -2.0, s_next)

    assert len(agent.memory) == 1

def test_dqn_agent_min_green_guard():
    agent = DQNAgent(min_green=5)
    agent.last_switch_step = 20
    state = (4, 3, 2, 1, 0, 0, 1)

    # Step 22 (only 2s passed) -> must force keep (action 0)
    act = agent.choose_action(state, current_step=22, explore=True)
    assert act == 0

def test_dqn_weights_persistence(tmp_path):
    model = NumpyDQNModel()
    save_path = tmp_path / "test_dqn_weights.json"
    model.save_weights(str(save_path))
    assert os.path.isfile(save_path)

    new_model = NumpyDQNModel()
    loaded = new_model.load_weights(str(save_path))
    assert loaded is True
    assert np.allclose(model.W1, new_model.W1)
    assert np.allclose(model.W2, new_model.W2)
    assert np.allclose(model.W3, new_model.W3)

def test_run_dqn_controller_mock_execution():
    metrics = run_dqn_controller(total_steps=50, step_length=1.0, use_mock=True)
    assert metrics is not None
    assert metrics["controller"] == "Deep Q-Network (DQN)"
    assert metrics["steps"] == 50
    assert metrics["avg_queue"] >= 0
    assert metrics["total_delay"] >= 0
