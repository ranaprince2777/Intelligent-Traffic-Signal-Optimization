/**
 * AutoFlow RL - Main Application Logic
 * Ties together the simulation engine, canvas renderer, telemetry charts,
 * tab navigation, and interactive controls.
 */

import { TrafficEngine } from './sim/trafficEngine.js';
import { IntersectionRenderer } from './sim/renderer.js';
import { StreamCharts } from './sim/charts.js';

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Simulator, Renderer, and Charts
  const canvas = document.getElementById('intersectionCanvas');
  const engine = new TrafficEngine();
  const renderer = new IntersectionRenderer(canvas);
  const charts = new StreamCharts('queueChartCanvas', 'rewardChartCanvas');

  // DOM Elements - Metrics
  const kpiAvgQueue = document.getElementById('kpiAvgQueue');
  const kpiMaxQueue = document.getElementById('kpiMaxQueue');
  const kpiTotalDelay = document.getElementById('kpiTotalDelay');
  const kpiThroughput = document.getElementById('kpiThroughput');
  const kpiPhaseName = document.getElementById('kpiPhaseName');
  const kpiPhaseElapsed = document.getElementById('kpiPhaseElapsed');

  // DOM Elements - Canvas HUD
  const hudModelName = document.getElementById('hudModelName');
  const hudSimClock = document.getElementById('hudSimClock');
  const hudRewardVal = document.getElementById('hudRewardVal');
  const hudLightRed = document.getElementById('hudLightRed');
  const hudLightYellow = document.getElementById('hudLightYellow');
  const hudLightGreen = document.getElementById('hudLightGreen');
  const hudPhaseLabel = document.getElementById('hudPhaseLabel');
  const hudPhaseTimer = document.getElementById('hudPhaseTimer');

  // DOM Elements - State Vector Inspector
  const sv_eb0 = document.getElementById('sv_eb0');
  const sv_eb1 = document.getElementById('sv_eb1');
  const sv_eb2 = document.getElementById('sv_eb2');
  const sv_sb0 = document.getElementById('sv_sb0');
  const sv_sb1 = document.getElementById('sv_sb1');
  const sv_sb2 = document.getElementById('sv_sb2');
  const sv_phase = document.getElementById('sv_phase');

  // DOM Elements - Action Q-Value Inspector
  const qValKeep = document.getElementById('qValKeep');
  const qValSwitch = document.getElementById('qValSwitch');
  const qBarKeep = document.getElementById('qBarKeep');
  const qBarSwitch = document.getElementById('qBarSwitch');
  const agentActionText = document.getElementById('agentActionText');
  const minGreenLockStatus = document.getElementById('minGreenLockStatus');
  const modelDescBox = document.getElementById('modelDescBox');

  // DOM Elements - Controls
  const btnPlayPause = document.getElementById('btnPlayPause');
  const playPauseIcon = document.getElementById('playPauseIcon');
  const playPauseText = document.getElementById('playPauseText');
  const btnStep = document.getElementById('btnStep');
  const btnReset = document.getElementById('btnReset');
  const btnEmergency = document.getElementById('btnEmergency');
  const speedRange = document.getElementById('speedRange');
  const speedVal = document.getElementById('speedVal');
  const densitySelect = document.getElementById('densitySelect');
  const imbalanceRange = document.getElementById('imbalanceRange');
  const imbalanceVal = document.getElementById('imbalanceVal');

  // DOM Elements - Models
  const btnModelStatic = document.getElementById('btnModelStatic');
  const btnModelQL = document.getElementById('btnModelQL');
  const btnModelDQN = document.getElementById('btnModelDQN');

  // Navigation Tabs
  const navTabBtns = document.querySelectorAll('.nav-tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  navTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTabId = btn.getAttribute('data-tab');
      navTabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(targetTabId);
      if (targetContent) {
        targetContent.classList.add('active');
      }

      // Re-trigger chart resize if switching to charts
      if (targetTabId === 'tab-charts') {
        setTimeout(() => charts.setupResize(), 50);
      }
    });
  });

  // Model Selection
  function updateModelSelection(model) {
    engine.setModel(model);
    [btnModelStatic, btnModelQL, btnModelDQN].forEach(b => b.classList.remove('active'));

    if (model === 'static') {
      btnModelStatic.classList.add('active');
      hudModelName.textContent = 'Static Timer (Fixed 30s)';
      hudModelName.style.color = '#f59e0b';
      modelDescBox.innerHTML = `
        <strong style="color: var(--accent-amber);">Static Timer Controller:</strong>
        Operates on rigid, predetermined 30s green intervals. Completely ignores queue length buildup, leading to severe congestion (~23,000 veh-seconds of delay) and spillback.
      `;
    } else if (model === 'ql') {
      btnModelQL.classList.add('active');
      hudModelName.textContent = 'Tabular Q-Learning';
      hudModelName.style.color = '#10b981';
      modelDescBox.innerHTML = `
        <strong style="color: var(--accent-green);">Tabular Q-Learning Agent:</strong>
        Uses a discretized state-action Q-table learned via Q-learning updates: <code>Q(s,a) ← Q + α[R + γ max Q - Q]</code>. Rapidly adapts to queue imbalances, achieving a 38% queue reduction and 22% throughput gain.
      `;
    } else {
      btnModelDQN.classList.add('active');
      hudModelName.textContent = 'Deep Q-Network (DQN)';
      hudModelName.style.color = '#06b6d4';
      modelDescBox.innerHTML = `
        <strong style="color: var(--primary);">Deep Q-Network (DQN):</strong>
        Employs a multi-layer neural network with 2 hidden layers (24 units each, ReLU). Generalizes complex detector inputs to optimize overall intersection delay, stabilizing worst-case traffic bursts.
      `;
    }
  }

  btnModelStatic.addEventListener('click', () => updateModelSelection('static'));
  btnModelQL.addEventListener('click', () => updateModelSelection('ql'));
  btnModelDQN.addEventListener('click', () => updateModelSelection('dqn'));

  // Simulation Controls
  btnPlayPause.addEventListener('click', () => {
    engine.isRunning = !engine.isRunning;
    if (engine.isRunning) {
      playPauseIcon.textContent = '⏸';
      playPauseText.textContent = 'Pause';
    } else {
      playPauseIcon.textContent = '▶';
      playPauseText.textContent = 'Resume';
    }
  });

  btnStep.addEventListener('click', () => {
    engine.update(0.5);
    updateUI();
  });

  btnReset.addEventListener('click', () => {
    engine.reset();
    updateUI();
  });

  btnEmergency.addEventListener('click', () => {
    engine.spawnEmergencyVehicle();
  });

  speedRange.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    engine.speedMultiplier = val;
    speedVal.textContent = `${val.toFixed(1)}x`;
  });

  function applyTrafficDemand() {
    const density = densitySelect.value;
    const imbalance = parseFloat(imbalanceRange.value);
    engine.setTrafficDemand(density, imbalance);

    if (imbalance === 0.5) {
      imbalanceVal.textContent = 'Balanced';
    } else if (imbalance > 0.5) {
      imbalanceVal.textContent = `EB +${Math.round((imbalance - 0.5) * 200)}%`;
    } else {
      imbalanceVal.textContent = `SB +${Math.round((0.5 - imbalance) * 200)}%`;
    }
  }

  densitySelect.addEventListener('change', applyTrafficDemand);
  imbalanceRange.addEventListener('input', applyTrafficDemand);

  // Lightbox Modal for Research Gallery
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImg');
  const modalTitle = document.getElementById('modalTitle');
  const modalDesc = document.getElementById('modalDesc');
  const modalCloseBtn = document.getElementById('modalCloseBtn');

  document.querySelectorAll('.gallery-card').forEach(card => {
    card.addEventListener('click', () => {
      const src = card.getAttribute('data-img');
      const title = card.getAttribute('data-title');
      const desc = card.getAttribute('data-desc');

      modalImg.src = src;
      modalTitle.textContent = title;
      modalDesc.textContent = desc;
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
    });
  });

  modalCloseBtn.addEventListener('click', () => {
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
  });

  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
    }
  });

  // Main UI Update Function
  function updateUI() {
    const metrics = engine.getMetrics();

    // 1. KPI Cards
    kpiAvgQueue.textContent = metrics.avgQueue;
    kpiMaxQueue.textContent = metrics.maxQueue;
    kpiTotalDelay.textContent = metrics.totalDelay.toLocaleString();
    kpiThroughput.textContent = metrics.throughput;

    const phaseNames = ['EB GREEN', 'EB YELLOW', 'SB GREEN', 'SB YELLOW'];
    const phaseColors = ['#10b981', '#f59e0b', '#10b981', '#f59e0b'];
    kpiPhaseName.textContent = phaseNames[metrics.currentPhase];
    kpiPhaseName.style.color = phaseColors[metrics.currentPhase];
    kpiPhaseElapsed.textContent = `Elapsed: ${metrics.phaseTime}s (Min: 5s)`;

    // 2. Canvas HUD
    hudSimClock.textContent = `${metrics.simTime}s`;
    hudRewardVal.textContent = metrics.currentReward;

    // Traffic Light HUD in Overlay
    hudLightRed.classList.remove('active');
    hudLightYellow.classList.remove('active');
    hudLightGreen.classList.remove('active');

    if (metrics.currentPhase === 0) {
      hudLightGreen.classList.add('active');
      hudPhaseLabel.textContent = 'Phase 0: EB Green';
    } else if (metrics.currentPhase === 1) {
      hudLightYellow.classList.add('active');
      hudPhaseLabel.textContent = 'Phase 1: EB Yellow';
    } else if (metrics.currentPhase === 2) {
      hudLightGreen.classList.add('active');
      hudPhaseLabel.textContent = 'Phase 2: SB Green';
    } else {
      hudLightYellow.classList.add('active');
      hudPhaseLabel.textContent = 'Phase 3: SB Yellow';
    }
    hudPhaseTimer.textContent = `Hold: ${metrics.phaseTime}s`;

    // 3. State Vector
    const s = metrics.state;
    sv_eb0.textContent = s[0];
    sv_eb1.textContent = s[1];
    sv_eb2.textContent = s[2];
    sv_sb0.textContent = s[3];
    sv_sb1.textContent = s[4];
    sv_sb2.textContent = s[5];
    sv_phase.textContent = s[6];

    // Highlight state vector items
    [sv_eb0, sv_eb1, sv_eb2].forEach(el => {
      const val = parseInt(el.textContent, 10);
      el.parentElement.classList.toggle('active', val > 0);
    });
    [sv_sb0, sv_sb1, sv_sb2].forEach(el => {
      const val = parseInt(el.textContent, 10);
      el.parentElement.classList.toggle('active', val > 0);
    });

    // 4. Action Q-Values
    const [q0, q1] = metrics.qValues;
    qValKeep.textContent = `Q: ${q0 >= -900 ? q0.toFixed(2) : 'LOCKED'}`;
    qValSwitch.textContent = `Q: ${q1 >= -900 ? q1.toFixed(2) : 'LOCKED'}`;

    // Normalize progress bar lengths
    if (q0 >= -900 && q1 >= -900) {
      const minQ = Math.min(q0, q1);
      const maxQ = Math.max(q0, q1);
      const span = Math.max(1, maxQ - minQ);
      const pctKeep = Math.max(15, Math.min(85, ((q0 - minQ) / span) * 70 + 15));
      const pctSwitch = Math.max(15, Math.min(85, ((q1 - minQ) / span) * 70 + 15));

      qBarKeep.style.width = `${pctKeep}%`;
      qBarSwitch.style.width = `${pctSwitch}%`;
    } else {
      qBarKeep.style.width = '75%';
      qBarSwitch.style.width = '20%';
    }

    if (metrics.chosenAction === 0) {
      qBarKeep.classList.add('winner');
      qBarSwitch.classList.remove('winner');
      agentActionText.textContent = '0 (KEEP PHASE)';
      agentActionText.style.color = '#10b981';
    } else {
      qBarKeep.classList.remove('winner');
      qBarSwitch.classList.add('winner');
      agentActionText.textContent = '1 (SWITCH PHASE)';
      agentActionText.style.color = '#06b6d4';
    }

    const timeSinceSwitch = parseFloat(metrics.simTime) - engine.lastSwitchTime;
    if (timeSinceSwitch < 5.0) {
      minGreenLockStatus.textContent = `Locked (${(5.0 - timeSinceSwitch).toFixed(1)}s remaining)`;
      minGreenLockStatus.style.color = '#f59e0b';
    } else {
      minGreenLockStatus.textContent = 'Unlocked (Hold > 5s)';
      minGreenLockStatus.style.color = '#10b981';
    }
  }

  // Animation Loop (60 FPS)
  let lastTime = performance.now();
  let uiUpdateTimer = 0;

  function loop(currentTime) {
    const deltaMs = currentTime - lastTime;
    lastTime = currentTime;
    const dt = Math.min(0.1, deltaMs / 1000);

    // Update simulation
    engine.update(dt);

    // Render Canvas
    renderer.render(engine);

    // Render Telemetry Charts
    charts.render(engine);

    // Throttle UI text / numbers updates to ~10 Hz for buttery smooth performance
    uiUpdateTimer += dt;
    if (uiUpdateTimer >= 0.1) {
      uiUpdateTimer = 0;
      updateUI();
    }

    requestAnimationFrame(loop);
  }

  // Check for local Python TraCI bridge (simulation/traci_bridge.py)
  const bridgeStatusDot = document.getElementById('bridgeStatusDot');
  const bridgeStatusText = document.getElementById('bridgeStatusText');

  async function checkTraCIBridge() {
    try {
      const res = await fetch('http://localhost:8000/api/status', { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        const data = await res.json();
        if (bridgeStatusDot && bridgeStatusText) {
          bridgeStatusDot.style.background = '#10b981';
          bridgeStatusDot.style.boxShadow = '0 0 8px #10b981';
          bridgeStatusText.textContent = data.is_mock 
            ? 'Mode: TraCI Bridge Connected (MockTraCI)' 
            : 'Mode: TraCI Bridge Connected (Native SUMO)';
          bridgeStatusText.style.color = '#34d399';
        }
      }
    } catch (e) {
      if (bridgeStatusDot && bridgeStatusText) {
        bridgeStatusDot.style.background = '#94a3b8';
        bridgeStatusDot.style.boxShadow = 'none';
        bridgeStatusText.textContent = 'Mode: High-Fidelity Client Emulation (TraCI Standalone)';
        bridgeStatusText.style.color = '#cbd5e1';
      }
    }
  }

  // Periodic check every 5 seconds
  checkTraCIBridge();
  setInterval(checkTraCIBridge, 5000);

  // Start the simulation loop
  requestAnimationFrame(loop);
});
