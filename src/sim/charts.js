/**
 * AutoFlow RL - Real-Time Canvas Charts
 * Lightweight, zero-dependency, ultra-smooth charting engine for live metrics streaming.
 */

export class StreamCharts {
  constructor(queueCanvasId, rewardCanvasId) {
    this.queueCanvas = document.getElementById(queueCanvasId);
    this.queueCtx = this.queueCanvas ? this.queueCanvas.getContext('2d') : null;

    this.rewardCanvas = document.getElementById(rewardCanvasId);
    this.rewardCtx = this.rewardCanvas ? this.rewardCanvas.getContext('2d') : null;

    this.setupResize();
  }

  setupResize() {
    const resize = () => {
      [this.queueCanvas, this.rewardCanvas].forEach(c => {
        if (!c) return;
        const rect = c.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        c.width = rect.width * dpr;
        c.height = rect.height * dpr;
        const ctx = c.getContext('2d');
        ctx.scale(dpr, dpr);
      });
    };
    window.addEventListener('resize', resize);
    setTimeout(resize, 60);
  }

  render(engine) {
    if (this.queueCanvas && this.queueCtx) {
      this.drawQueueChart(this.queueCtx, this.queueCanvas, engine);
    }
    if (this.rewardCanvas && this.rewardCtx) {
      this.drawRewardChart(this.rewardCtx, this.rewardCanvas, engine);
    }
  }

  drawQueueChart(ctx, canvas, engine) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.clearRect(0, 0, w, h);

    const padLeft = 40;
    const padRight = 15;
    const padTop = 20;
    const padBottom = 25;
    const chartW = w - padLeft - padRight;
    const chartH = h - padTop - padBottom;

    const data = engine.telemetryHistory.currentQueue;
    const staticData = engine.telemetryHistory.queueStatic;
    const qlData = engine.telemetryHistory.queueQL;
    const dqnData = engine.telemetryHistory.queueDQN;

    if (!data || data.length < 2) {
      this.drawPlaceholder(ctx, w, h, 'Gathering real-time detector data...');
      return;
    }

    // Grid lines and Y-axis scale (0 to 15 vehicles)
    const maxY = 15;
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
    ctx.lineWidth = 1;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';

    for (let step = 0; step <= maxY; step += 5) {
      const y = padTop + chartH - (step / maxY) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();
      ctx.fillText(`${step}`, padLeft - 6, y + 3);
    }

    // X-axis baseline
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.5)';
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop + chartH);
    ctx.lineTo(w - padRight, padTop + chartH);
    ctx.stroke();

    // Draw Static Baseline Line (Amber dashed)
    this.drawLine(ctx, staticData, padLeft, padTop, chartW, chartH, maxY, '#f59e0b', [4, 4], 1.5);

    // Draw Q-Learning Line (Emerald green)
    this.drawLine(ctx, qlData, padLeft, padTop, chartW, chartH, maxY, '#10b981', [2, 3], 1.5);

    // Draw Active Simulation Run Line (Cyan glowing solid)
    const activeColor = engine.modelType === 'static' ? '#f59e0b' : (engine.modelType === 'ql' ? '#10b981' : '#06b6d4');
    this.drawFilledLine(ctx, data, padLeft, padTop, chartW, chartH, maxY, activeColor);
  }

  drawRewardChart(ctx, canvas, engine) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.clearRect(0, 0, w, h);

    const padLeft = 45;
    const padRight = 15;
    const padTop = 20;
    const padBottom = 25;
    const chartW = w - padLeft - padRight;
    const chartH = h - padTop - padBottom;

    const data = engine.telemetryHistory.reward;

    if (!data || data.length < 2) {
      this.drawPlaceholder(ctx, w, h, 'Gathering RL reward signals R = -(Queue)...');
      return;
    }

    // Reward runs from -15 to 0
    const minY = -15;
    const maxY = 0;
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
    ctx.lineWidth = 1;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';

    for (let r = minY; r <= maxY; r += 5) {
      const y = padTop + ((r - maxY) / (minY - maxY)) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();
      ctx.fillText(`${r}`, padLeft - 6, y + 3);
    }

    // Zero Line
    const zeroY = padTop;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, zeroY);
    ctx.lineTo(w - padRight, zeroY);
    ctx.stroke();

    // Reward Plot (Purple/Cyan gradient)
    this.drawRewardLine(ctx, data, padLeft, padTop, chartW, chartH, minY, maxY, '#8b5cf6');
  }

  drawLine(ctx, points, x0, y0, cw, ch, maxY, color, dash = [], lineW = 2) {
    if (!points || points.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.setLineDash(dash);

    const stepX = cw / (points.length - 1);
    ctx.beginPath();

    for (let i = 0; i < points.length; i++) {
      const val = Math.max(0, Math.min(maxY, points[i]));
      const x = x0 + i * stepX;
      const y = y0 + ch - (val / maxY) * ch;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawFilledLine(ctx, points, x0, y0, cw, ch, maxY, color) {
    if (!points || points.length < 2) return;
    const stepX = cw / (points.length - 1);

    // Gradient fill under curve
    const grad = ctx.createLinearGradient(0, y0, 0, y0 + ch);
    grad.addColorStop(0, color.replace(')', ', 0.35)').replace('rgb', 'rgba').replace('#06b6d4', 'rgba(6, 182, 212, 0.35)'));
    grad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(x0, y0 + ch);

    for (let i = 0; i < points.length; i++) {
      const val = Math.max(0, Math.min(maxY, points[i]));
      const x = x0 + i * stepX;
      const y = y0 + ch - (val / maxY) * ch;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(x0 + (points.length - 1) * stepX, y0 + ch);
    ctx.closePath();
    ctx.fill();

    // Outline
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();

    for (let i = 0; i < points.length; i++) {
      const val = Math.max(0, Math.min(maxY, points[i]));
      const x = x0 + i * stepX;
      const y = y0 + ch - (val / maxY) * ch;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Latest point pulse
    const lastVal = points[points.length - 1];
    const lastX = x0 + (points.length - 1) * stepX;
    const lastY = y0 + ch - (lastVal / maxY) * ch;
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  drawRewardLine(ctx, points, x0, y0, cw, ch, minY, maxY, color) {
    if (!points || points.length < 2) return;
    const stepX = cw / (points.length - 1);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.beginPath();

    for (let i = 0; i < points.length; i++) {
      const val = points[i];
      const norm = (val - maxY) / (minY - maxY);
      const x = x0 + i * stepX;
      const y = y0 + norm * ch;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  drawPlaceholder(ctx, w, h, text) {
    ctx.font = '12px "Outfit", sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText(text, w / 2, h / 2);
  }
}
