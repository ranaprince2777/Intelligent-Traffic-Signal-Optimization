/**
 * AutoFlow RL - Intersection Canvas Renderer
 * High-performance 2D canvas drawing with glassmorphic styling,
 * lane detector highlighting, vehicle physics visualization, and traffic light bloom.
 */

export class IntersectionRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = canvas.width;
    this.height = canvas.height;

    // Resize observer
    this.setupResize();
  }

  setupResize() {
    const resize = () => {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.width = rect.width;
      this.height = rect.height;
      this.ctx.scale(dpr, dpr);
    };
    window.addEventListener('resize', resize);
    setTimeout(resize, 50);
  }

  render(engine) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    // Center coordinates
    const cx = w / 2;
    const cy = h / 2;

    // Lane and road dimensions
    const laneWidth = 26;
    const numLanesPerDir = 3;
    const halfRoadWidth = laneWidth * numLanesPerDir; // 78px each side, total road 156px
    const fullRoadWidth = halfRoadWidth * 2;

    // 1. Draw Background Grid & Terrain
    this.drawTerrain(ctx, w, h, cx, cy);

    // 2. Draw Roads (Asphalt + borders)
    this.drawRoads(ctx, w, h, cx, cy, halfRoadWidth);

    // 3. Draw Lane Markings, Dividers, Crosswalks, and Stop Lines
    this.drawLaneMarkings(ctx, w, h, cx, cy, laneWidth, halfRoadWidth);

    // 4. Draw E1 Lane Area Detectors (e2_0 to e2_5)
    this.drawDetectors(ctx, cx, cy, laneWidth, halfRoadWidth, engine.detectorCounts);

    // 5. Draw Vehicles
    this.drawVehicles(ctx, cx, cy, laneWidth, halfRoadWidth, engine.vehicles);

    // 6. Draw Traffic Light Signal Gantries
    this.drawTrafficLights(ctx, cx, cy, halfRoadWidth, engine.currentPhase, engine.phaseTime);
  }

  drawTerrain(ctx, w, h, cx, cy) {
    // Subtle cyber grid
    ctx.fillStyle = '#080d1a';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
    ctx.lineWidth = 1;
    const gridSize = 40;

    ctx.beginPath();
    for (let x = 0; x < w; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    // Corner decorative sidewalk zones
    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.fillRect(0, 0, cx - halfRoad(78), cy - halfRoad(78));
    ctx.fillRect(cx + halfRoad(78), 0, w - (cx + halfRoad(78)), cy - halfRoad(78));
    ctx.fillRect(0, cy + halfRoad(78), cx - halfRoad(78), h - (cy + halfRoad(78)));
    ctx.fillRect(cx + halfRoad(78), cy + halfRoad(78), w - (cx + halfRoad(78)), h - (cy + halfRoad(78)));
  }

  drawRoads(ctx, w, h, cx, cy, hrw) {
    // Road Asphalt Fill
    ctx.fillStyle = '#111827';
    // Horizontal (East-West) road
    ctx.fillRect(0, cy - hrw, w, hrw * 2);
    // Vertical (North-South) road
    ctx.fillRect(cx - hrw, 0, hrw * 2, h);

    // Road Outer Curbs / Borders
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.6)';
    ctx.lineWidth = 2;

    // NW corner curb
    ctx.beginPath();
    ctx.moveTo(0, cy - hrw);
    ctx.lineTo(cx - hrw, cy - hrw);
    ctx.lineTo(cx - hrw, 0);
    ctx.stroke();

    // NE corner curb
    ctx.beginPath();
    ctx.moveTo(w, cy - hrw);
    ctx.lineTo(cx + hrw, cy - hrw);
    ctx.lineTo(cx + hrw, 0);
    ctx.stroke();

    // SW corner curb
    ctx.beginPath();
    ctx.moveTo(0, cy + hrw);
    ctx.lineTo(cx - hrw, cy + hrw);
    ctx.lineTo(cx - hrw, h);
    ctx.stroke();

    // SE corner curb
    ctx.beginPath();
    ctx.moveTo(w, cy + hrw);
    ctx.lineTo(cx + hrw, cy + hrw);
    ctx.lineTo(cx + hrw, h);
    ctx.stroke();
  }

  drawLaneMarkings(ctx, w, h, cx, cy, lw, hrw) {
    // Center divider (double solid amber lines)
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    // West approach
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(cx - hrw, cy);
    ctx.stroke();
    // East approach
    ctx.beginPath();
    ctx.moveTo(cx + hrw, cy);
    ctx.lineTo(w, cy);
    ctx.stroke();
    // North approach
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, cy - hrw);
    ctx.stroke();
    // South approach
    ctx.beginPath();
    ctx.moveTo(cx, cy + hrw);
    ctx.lineTo(cx, h);
    ctx.stroke();

    // Dashed white lane dividers
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 12]);

    for (let i = 1; i < 3; i++) {
      // East-West lanes (Eastbound side: cy to cy + hrw)
      ctx.beginPath();
      ctx.moveTo(0, cy + i * lw);
      ctx.lineTo(cx - hrw, cy + i * lw);
      ctx.moveTo(cx + hrw, cy + i * lw);
      ctx.lineTo(w, cy + i * lw);
      ctx.stroke();

      // East-West lanes (Westbound side: cy - hrw to cy)
      ctx.beginPath();
      ctx.moveTo(0, cy - i * lw);
      ctx.lineTo(cx - hrw, cy - i * lw);
      ctx.moveTo(cx + hrw, cy - i * lw);
      ctx.lineTo(w, cy - i * lw);
      ctx.stroke();

      // North-South lanes (Southbound side: cx to cx + hrw)
      ctx.beginPath();
      ctx.moveTo(cx + i * lw, 0);
      ctx.lineTo(cx + i * lw, cy - hrw);
      ctx.moveTo(cx + i * lw, cy + hrw);
      ctx.lineTo(cx + i * lw, h);
      ctx.stroke();

      // North-South lanes (Northbound side: cx - hrw to cx)
      ctx.beginPath();
      ctx.moveTo(cx - i * lw, 0);
      ctx.lineTo(cx - i * lw, cy - hrw);
      ctx.moveTo(cx - i * lw, cy + hrw);
      ctx.lineTo(cx - i * lw, h);
      ctx.stroke();
    }
    ctx.setLineDash([]); // Reset dash

    // Solid White Stop Lines
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;

    // Eastbound Stop Line (at cx - hrw, for y from cy to cy + hrw)
    ctx.beginPath();
    ctx.moveTo(cx - hrw, cy);
    ctx.lineTo(cx - hrw, cy + hrw);
    ctx.stroke();

    // Southbound Stop Line (at cy - hrw, for x from cx to cx + hrw)
    ctx.beginPath();
    ctx.moveTo(cx, cy - hrw);
    ctx.lineTo(cx + hrw, cy - hrw);
    ctx.stroke();

    // Pedestrian Crosswalk stripes
    this.drawCrosswalk(ctx, cx - hrw - 18, cy, 14, hrw, 'vertical');
    this.drawCrosswalk(ctx, cx, cy - hrw - 18, hrw, 14, 'horizontal');
  }

  drawCrosswalk(ctx, x, y, width, length, orient) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    const stripeCount = 6;
    if (orient === 'vertical') {
      const step = length / stripeCount;
      for (let i = 0; i < stripeCount; i++) {
        ctx.fillRect(x, y + i * step + 2, width, step - 4);
      }
    } else {
      const step = width / stripeCount;
      for (let i = 0; i < stripeCount; i++) {
        ctx.fillRect(x + i * step + 2, y, step - 4, length);
      }
    }
  }

  drawDetectors(ctx, cx, cy, lw, hrw, detectorCounts) {
    // Detector zone lengths: 40px long, placed before stop line
    const detLength = 65;
    const detOffset = 15; // from stop line

    // Eastbound Detectors: Node1_2_EB_0 (Lane 0), Node1_2_EB_1 (Lane 1), Node1_2_EB_2 (Lane 2)
    const ebKeys = ['Node1_2_EB_0', 'Node1_2_EB_1', 'Node1_2_EB_2'];
    for (let lane = 0; lane < 3; lane++) {
      const key = ebKeys[lane];
      const count = detectorCounts[key] || 0;
      const x = cx - hrw - detOffset - detLength;
      const y = cy + lane * lw + 3;
      const w = detLength;
      const h = lw - 6;

      this.renderDetectorBox(ctx, x, y, w, h, key, count, 'EB');
    }

    // Southbound Detectors: Node2_7_SB_0 (Lane 0), Node2_7_SB_1 (Lane 1), Node2_7_SB_2 (Lane 2)
    const sbKeys = ['Node2_7_SB_0', 'Node2_7_SB_1', 'Node2_7_SB_2'];
    for (let lane = 0; lane < 3; lane++) {
      const key = sbKeys[lane];
      const count = detectorCounts[key] || 0;
      const x = cx + lane * lw + 3;
      const y = cy - hrw - detOffset - detLength;
      const w = lw - 6;
      const h = detLength;

      this.renderDetectorBox(ctx, x, y, w, h, key, count, 'SB');
    }
  }

  renderDetectorBox(ctx, x, y, w, h, id, count, dir) {
    const isOccupied = count > 0;
    
    // Background glow
    if (isOccupied) {
      ctx.fillStyle = 'rgba(6, 182, 212, 0.25)';
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1.5;
      ctx.shadowColor = 'rgba(6, 182, 212, 0.6)';
      ctx.shadowBlur = 10;
    } else {
      ctx.fillStyle = 'rgba(30, 41, 59, 0.3)';
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
      ctx.lineWidth = 1;
      ctx.shadowBlur = 0;
    }

    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur = 0; // reset

    // Detector ID & Count Label
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = isOccupied ? '#38bdf8' : 'rgba(148, 163, 184, 0.7)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const label = `${id.split('_').slice(-2).join('_')} : ${count}`;
    if (dir === 'EB') {
      ctx.fillText(label, x + w / 2, y + h / 2);
    } else {
      ctx.save();
      ctx.translate(x + w / 2, y + h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }
  }

  drawVehicles(ctx, cx, cy, lw, hrw, vehicles) {
    const stopLineDist = 180;

    for (const v of vehicles) {
      let vx = 0;
      let vy = 0;
      let angle = 0;

      if (v.direction === 'EB') {
        // Eastbound: starts from West (x=0) moving East (+x)
        // Stop line is at cx - hrw
        const stopX = cx - hrw;
        vx = stopX - (stopLineDist - v.pos);
        vy = cy + v.lane * lw + lw / 2;
        angle = 0;
      } else {
        // Southbound: starts from North (y=0) moving South (+y)
        // Stop line is at cy - hrw
        const stopY = cy - hrw;
        vx = cx + v.lane * lw + lw / 2;
        vy = stopY - (stopLineDist - v.pos);
        angle = Math.PI / 2;
      }

      this.renderSingleVehicle(ctx, vx, vy, angle, v);
    }
  }

  renderSingleVehicle(ctx, x, y, angle, v) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    const carLen = v.length * 4.2; // approx 19 - 27px
    const carWid = 14;

    // Shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.fillRect(-carLen / 2 + 1, -carWid / 2 + 2, carLen, carWid);

    // Car Body
    ctx.fillStyle = v.color;
    ctx.beginPath();
    ctx.roundRect(-carLen / 2, -carWid / 2, carLen, carWid, 3);
    ctx.fill();

    // Windshield & Roof
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(-carLen / 4, -carWid / 2 + 2.5, carLen / 2, carWid - 5);

    // Headlights (Front is +x)
    ctx.fillStyle = '#fef08a';
    ctx.fillRect(carLen / 2 - 2, -carWid / 2 + 1.5, 2, 3);
    ctx.fillRect(carLen / 2 - 2, carWid / 2 - 4.5, 2, 3);

    // Brake Lights (Red if stopped or slowing down)
    ctx.fillStyle = v.speed < 1.0 ? '#ef4444' : '#7f1d1d';
    if (v.speed < 1.0) {
      ctx.shadowColor = '#ef4444';
      ctx.shadowBlur = 6;
    }
    ctx.fillRect(-carLen / 2, -carWid / 2 + 1.5, 2, 3);
    ctx.fillRect(-carLen / 2, carWid / 2 - 4.5, 2, 3);
    ctx.shadowBlur = 0;

    // Emergency Vehicle Flasher
    if (v.isEmergency) {
      const flash = Math.floor(Date.now() / 150) % 2 === 0;
      ctx.fillStyle = flash ? '#ef4444' : '#3b82f6';
      ctx.shadowColor = flash ? '#ef4444' : '#3b82f6';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(0, 0, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    ctx.restore();
  }

  drawTrafficLights(ctx, cx, cy, hrw, phase, phaseTime) {
    // Eastbound Signal Head (Controls Eastbound Approach, placed at NW or SW curb)
    const ebX = cx - hrw - 12;
    const ebY = cy + hrw + 12;
    // EB Light State: Phase 0 = Green, Phase 1 = Yellow, Phase 2/3 = Red
    let ebColor = 'red';
    if (phase === 0) ebColor = 'green';
    else if (phase === 1) ebColor = 'yellow';

    this.renderSignalPost(ctx, ebX, ebY, ebColor, 'EB', phaseTime);

    // Southbound Signal Head (Controls Southbound Approach, placed at NE curb)
    const sbX = cx + hrw + 12;
    const sbY = cy - hrw - 12;
    // SB Light State: Phase 2 = Green, Phase 3 = Yellow, Phase 0/1 = Red
    let sbColor = 'red';
    if (phase === 2) sbColor = 'green';
    else if (phase === 3) sbColor = 'yellow';

    this.renderSignalPost(ctx, sbX, sbY, sbColor, 'SB', phaseTime);
  }

  renderSignalPost(ctx, x, y, state, label, timer) {
    const boxW = 22;
    const boxH = 50;

    // Housing Box
    ctx.fillStyle = '#0f172a';
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(x - boxW / 2, y - boxH / 2, boxW, boxH, 5);
    ctx.fill();
    ctx.stroke();

    // Red Lamp
    const redActive = state === 'red';
    this.renderLamp(ctx, x, y - 14, redActive ? '#ef4444' : '#3a1a1a', redActive);

    // Yellow Lamp
    const yellowActive = state === 'yellow';
    this.renderLamp(ctx, x, y, yellowActive ? '#f59e0b' : '#3a2e15', yellowActive);

    // Green Lamp
    const greenActive = state === 'green';
    this.renderLamp(ctx, x, y + 14, greenActive ? '#10b981' : '#143527', greenActive);

    // Approach Badge
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.fillText(`${label}`, x, y + boxH / 2 + 14);
  }

  renderLamp(ctx, lx, ly, color, active) {
    ctx.fillStyle = color;
    if (active) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
    }
    ctx.beginPath();
    ctx.arc(lx, ly, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

function halfRoad(val) {
  return val;
}
