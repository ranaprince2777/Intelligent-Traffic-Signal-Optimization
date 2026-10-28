"""
Tests for SUMO Scenario XML Configuration Files
Validates RL.sumocfg, RL.net.xml, RL.add.xml, and RL.rou.xml.
"""

import os
import xml.etree.ElementTree as ET
import pytest

from simulation.sumo_utils import find_sumo_config, DETECTORS, TRAFFIC_LIGHT_ID

def test_sumo_config_file_exists():
    cfg = find_sumo_config()
    assert cfg is not None, "RL.sumocfg should be discoverable by find_sumo_config()"
    assert os.path.isfile(cfg), f"Config file does not exist: {cfg}"

def test_sumo_config_xml_structure():
    cfg = find_sumo_config()
    tree = ET.parse(cfg)
    root = tree.getroot()
    assert root.tag == "sumoConfiguration", f"Root tag must be sumoConfiguration, got {root.tag}"
    
    input_tag = root.find("input")
    assert input_tag is not None, "Config must contain an <input> section"
    assert input_tag.find("net-file") is not None, "<input> must contain <net-file>"
    assert input_tag.find("route-files") is not None, "<input> must contain <route-files>"
    assert input_tag.find("additional-files") is not None, "<input> must contain <additional-files>"

def test_network_traffic_lights_and_phases():
    cfg = find_sumo_config()
    cfg_dir = os.path.dirname(cfg)
    net_path = os.path.join(cfg_dir, "RL.net.xml")
    assert os.path.isfile(net_path), f"RL.net.xml not found at {net_path}"

    tree = ET.parse(net_path)
    root = tree.getroot()

    tl_logics = root.findall("tlLogic")
    assert len(tl_logics) > 0, "Network must contain at least one tlLogic"

    node2_tl = None
    for tl in tl_logics:
        if tl.get("id") == TRAFFIC_LIGHT_ID:
            node2_tl = tl
            break

    assert node2_tl is not None, f"Traffic light '{TRAFFIC_LIGHT_ID}' not found in RL.net.xml"
    phases = node2_tl.findall("phase")
    assert len(phases) == 4, f"Expected 4 phases for {TRAFFIC_LIGHT_ID}, found {len(phases)}"

    # Validate phase durations match 42s Green, 3s Yellow, 42s Green, 3s Yellow
    durations = [int(p.get("duration")) for p in phases]
    assert durations == [42, 3, 42, 3], f"Unexpected phase durations: {durations}"

def test_detectors_configuration():
    cfg = find_sumo_config()
    cfg_dir = os.path.dirname(cfg)
    add_path = os.path.join(cfg_dir, "RL.add.xml")
    assert os.path.isfile(add_path), f"RL.add.xml not found at {add_path}"

    tree = ET.parse(add_path)
    root = tree.getroot()

    found_detectors = []
    for det in root.findall("laneAreaDetector"):
        found_detectors.append(det.get("id"))

    for expected_det in DETECTORS:
        assert expected_det in found_detectors, f"Expected detector '{expected_det}' not defined in RL.add.xml"

    assert len(found_detectors) == len(DETECTORS), f"Expected {len(DETECTORS)} detectors, found {len(found_detectors)}"

def test_route_flows():
    cfg = find_sumo_config()
    cfg_dir = os.path.dirname(cfg)
    rou_path = os.path.join(cfg_dir, "RL.rou.xml")
    assert os.path.isfile(rou_path), f"RL.rou.xml not found at {rou_path}"

    tree = ET.parse(rou_path)
    root = tree.getroot()

    flows = root.findall("flow")
    assert len(flows) >= 3, f"Expected at least 3 traffic flows, found {len(flows)}"
    flow_ids = [f.get("id") for f in flows]
    for fid in ["f_0", "f_1", "f_2"]:
        assert fid in flow_ids, f"Flow '{fid}' not found in RL.rou.xml"
