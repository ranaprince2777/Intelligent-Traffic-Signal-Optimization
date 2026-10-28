"""
Tests for TraCI HTTP Bridge logic
"""

import pytest
from simulation.traci_bridge import TraCISimulationState

def test_bridge_simulation_state_initialization():
    state = TraCISimulationState()
    assert state.step == 0
    assert state.traci is not None

def test_bridge_step_execution():
    state = TraCISimulationState()
    res = state.step_sim(action=0)
    assert res["step"] == 1
    assert "phase" in res
    assert "reward" in res
    assert "detectors" in res
    assert len(res["detectors"]) == 6
    assert res["action_taken"] == 0
