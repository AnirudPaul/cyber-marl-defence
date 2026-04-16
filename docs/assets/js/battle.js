/* ============================================================
   battle.js — AI vs AI Simulation
   ASTRA: DDPG Attacker vs Dueling-DQN Defender
   Pure JS — no ML libraries needed.
   ============================================================ */

(function () {

  /* ── Canvas setup ── */
  const canvas = document.getElementById('battle-canvas');
  const ctx    = canvas.getContext('2d');
  const W      = canvas.width  = 560;
  const H      = canvas.height = 280;

  /* ── State ── */
  let running   = false;
  let speed     = 1;        // 1 = normal, 2 = fast, 3 = x3
  let step      = 0;
  let frameRef  = null;

  /* ── Stats (mirroring real RESULTS.md numbers, animated toward them) ── */
  const TARGET = {
    defCleanF1:    0.7608,
    defRobustF1:   0.7610,
    pgdCleanF1:    0.4998,
    pgdRobustF1:   0.4999,
    atkSuccessEarly: 0.68,  // attacker starts strong
    atkSuccessFinal: 0.12,  // defender learns to resist
  };

  let state = {
    atkSuccess: 0.68,
    defAccuracy: 0.42,
    cleanF1: 0.42,
    robustF1: 0.41,
    episode: 0,
    atkReward: 0,
    defReward: 0,
    phase: 'Warmup',        // Warmup → Attack → Adapt → Stable
  };

  /* ── Packets ── */
  const PACKET_TYPES = ['BENIGN','DDoS','DoS','Port Scan','Backdoor'];
  const COLORS = {
    BENIGN:     '#10b981',
    DDoS:       '#ef4444',
    DoS:        '#f59e0b',
    'Port Scan':'#a78bfa',
    Backdoor:   '#00d4ff',
  };

  let packets = [];

  function makePacket() {
    const type  = PACKET_TYPES[Math.floor(Math.random() * PACKET_TYPES.length)];
    const isAdv = Math.random() < 0.35;   // 35% are adversarial
    return {
      x: -30,
      y: 60 + Math.random() * (H - 120),
      r: 7 + Math.random() * 4,
      speed: 1.2 + Math.random() * 1.2,
      type,
      isAdv,
      perturbed: false,
      classified: false,
      correct: false,
      alpha: 1,
      label: null,
      glitch: 0,
    };
  }

  /* ── Zones ── */
  const ZONES = {
    atkX: 130,   // Attacker agent x-position
    defX: W - 130, // Defender agent x-position
    atkW: 80,
    defW: 80,
  };

  /* ── Draw helpers ── */
  function drawAgent(x, label, color, energy, isAtk) {
    // Outer glow
    const grd = ctx.createRadialGradient(x, H/2, 10, x, H/2, 55);
    grd.addColorStop(0, color + '30');
    grd.addColorStop(1, 'transparent');
    ctx.fillStyle = grd;
    ctx.fillRect(x - 60, H/2 - 60, 120, 120);

    // Hex shape (simplified as circle)
    ctx.save();
    ctx.beginPath();
    const sides = 6;
    const R = 32;
    for (let i = 0; i < sides; i++) {
      const angle = (i * 2 * Math.PI / sides) - Math.PI / 2;
      const px = x + R * Math.cos(angle);
      const py = H/2 + R * Math.sin(angle);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = color + '18';
    ctx.fill();
    ctx.restore();

    // Icon text
    ctx.fillStyle = color;
    ctx.font = 'bold 18px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(isAtk ? '⚔' : '🛡', x, H/2);

    // Label
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.fillStyle = color;
    ctx.fillText(label, x, H/2 + 48);

    // Energy bar
    const bw = 60, bh = 5, bx = x - bw/2, by = H/2 + 58;
    ctx.fillStyle = '#1e2d4a';
    ctx.fillRect(bx, by, bw, bh);
    ctx.fillStyle = color;
    ctx.fillRect(bx, by, bw * energy, bh);
  }

  function drawPacket(p) {
    ctx.save();
    ctx.globalAlpha = p.alpha;

    const baseColor = COLORS[p.type] || '#64748b';
    const color     = p.perturbed ? '#ff6b6b' : baseColor;

    // Glitch effect on adversarial packets
    let dx = 0;
    if (p.glitch > 0) {
      dx = (Math.random() - 0.5) * p.glitch * 4;
      p.glitch *= 0.85;
    }

    // Packet body
    ctx.beginPath();
    ctx.arc(p.x + dx, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = color + (p.perturbed ? 'cc' : '99');
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = p.perturbed ? 2 : 1;
    ctx.stroke();

    // ADV marker
    if (p.isAdv && !p.classified) {
      ctx.font = '8px monospace';
      ctx.fillStyle = '#ef4444';
      ctx.textAlign = 'center';
      ctx.fillText('ADV', p.x + dx, p.y - p.r - 3);
    }

    // Classification label
    if (p.label) {
      ctx.font = 'bold 8px monospace';
      ctx.fillStyle = p.correct ? '#10b981' : '#ef4444';
      ctx.textAlign = 'center';
      ctx.fillText(p.label, p.x + dx, p.y + p.r + 10);
    }

    ctx.restore();
  }

  function drawArrow(x1, y1, x2, y2, color, text) {
    ctx.save();
    ctx.strokeStyle = color + '60';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
    if (text) {
      ctx.font = '9px monospace';
      ctx.fillStyle = color + 'aa';
      ctx.textAlign = 'center';
      ctx.fillText(text, (x1 + x2) / 2, (y1 + y2) / 2 - 6);
    }
  }

  function drawGrid() {
    ctx.save();
    ctx.strokeStyle = '#1e2d4a44';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    ctx.restore();
  }

  function drawLane() {
    // Flow lane
    const gradient = ctx.createLinearGradient(0, 0, W, 0);
    gradient.addColorStop(0,   'rgba(0,212,255,0.03)');
    gradient.addColorStop(0.5, 'rgba(0,212,255,0.06)');
    gradient.addColorStop(1,   'rgba(0,212,255,0.03)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, H/2 - 60, W, 120);
  }

  /* ── Phase labels ── */
  function getPhase() {
    if (step < 80)  return { name: 'Warmup',  color: '#64748b' };
    if (step < 200) return { name: 'Attack ↑', color: '#ef4444' };
    if (step < 400) return { name: 'Adapting', color: '#f59e0b' };
    return           { name: 'Stable',  color: '#10b981' };
  }

  /* ── Main render ── */
  function render() {
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, W, H);

    drawGrid();
    drawLane();

    const phase = getPhase();

    // Phase label top-center
    ctx.font = 'bold 10px monospace';
    ctx.fillStyle = phase.color;
    ctx.textAlign = 'center';
    ctx.fillText(`[ ${phase.name} — Ep ${state.episode} ]`, W/2, 18);

    // Attacker energy: rises early, then falls as defender adapts
    const atkEnergy = Math.max(0.05, Math.min(1, state.atkSuccess));
    const defEnergy = Math.max(0.1, Math.min(1, state.defAccuracy));

    drawAgent(ZONES.atkX, 'DDPG Attacker', '#ef4444', atkEnergy, true);
    drawAgent(ZONES.defX, 'DQN Defender',  '#10b981', defEnergy, false);

    // Feedback arrow between agents
    if (step > 80) {
      drawArrow(ZONES.defX - 40, H/2 - 15, ZONES.atkX + 40, H/2 - 15,
                '#7c3aed', 'reward signal');
    }

    // Packets
    packets.forEach(drawPacket);

    // Step watermark
    ctx.font = '9px monospace';
    ctx.fillStyle = '#1e2d4a';
    ctx.textAlign = 'right';
    ctx.fillText(`t=${step}`, W - 8, H - 6);
  }

  /* ── Simulation tick ── */
  function lerp(a, b, t) { return a + (b - a) * t; }

  let spawnTimer = 0;

  function tick() {
    step++;
    spawnTimer++;

    // Spawn packets
    if (spawnTimer > Math.max(4, 10 - speed * 2)) {
      packets.push(makePacket());
      spawnTimer = 0;
    }

    // Update episodes
    if (step % 50 === 0) state.episode++;

    // Phase-based stat evolution
    const t = Math.min(1, Math.max(0, (step - 80) / 300));

    // Attacker success: starts high, drops as defender adapts
    const targetAtk = lerp(TARGET.atkSuccessEarly, TARGET.atkSuccessFinal, t);
    state.atkSuccess = lerp(state.atkSuccess, targetAtk + (Math.random() - 0.5) * 0.03, 0.04);

    // Defender accuracy: starts low, rises
    const targetDef = lerp(0.42, TARGET.defRobustF1, t);
    state.defAccuracy = lerp(state.defAccuracy, targetDef + (Math.random() - 0.5) * 0.02, 0.04);

    // Clean F1
    state.cleanF1  = lerp(state.cleanF1, TARGET.defCleanF1, 0.01);
    state.robustF1 = state.defAccuracy;

    // RL rewards (noisy)
    state.atkReward = (state.atkSuccess - 0.5) * 2 + (Math.random() - 0.5) * 0.3;
    state.defReward = (state.defAccuracy - 0.3) * 3 + (Math.random() - 0.5) * 0.2;

    // Move and interact packets
    packets = packets.filter(p => p.alpha > 0.05);

    packets.forEach(p => {
      p.x += p.speed * speed;

      // Attacker zone: perturb adversarial packets
      if (!p.perturbed && p.isAdv && Math.abs(p.x - ZONES.atkX) < 20) {
        p.perturbed = true;
        p.glitch = 3;
        logEvent('atk', `Attacker perturbs ${p.type} packet`);
      }

      // Defender zone: classify
      if (!p.classified && Math.abs(p.x - ZONES.defX) < 20) {
        p.classified = true;
        // Probability of correct classification
        const pCorrect = p.perturbed
          ? state.defAccuracy         // harder to classify perturbed
          : 0.96 + Math.random() * 0.03;

        p.correct = Math.random() < pCorrect;
        p.label   = p.correct ? '✓' : '✗';
        logEvent('def', `Defender: ${p.correct ? 'DETECTED' : 'EVADED'} (${p.type})`);
      }

      // Fade out at right edge
      if (p.x > W - 20) p.alpha -= 0.06;
    });

    // Update UI stats
    updateStats();
  }

  /* ── Log ── */
  const logEl = document.getElementById('battle-log');
  let logLines = [];

  function logEvent(who, msg) {
    const cls  = who === 'atk' ? 'log-atk' : 'log-def';
    const time = String(step).padStart(4, '0');
    logLines.push(`<span class="${cls}">[t=${time}] ${msg}</span>`);
    if (logLines.length > 60) logLines.shift();
    logEl.innerHTML = logLines.join('<br>');
    logEl.scrollTop = logEl.scrollHeight;
  }

  function logInfo(msg) {
    logLines.push(`<span class="log-info">${msg}</span>`);
    if (logLines.length > 60) logLines.shift();
    logEl.innerHTML = logLines.join('<br>');
    logEl.scrollTop = logEl.scrollHeight;
  }

  /* ── Stats DOM update ── */
  function pct(v) { return (v * 100).toFixed(1) + '%'; }
  function f(v)   { return v.toFixed(4); }

  function setBar(id, val) {
    const el = document.getElementById(id);
    if (el) el.style.width = Math.round(val * 100) + '%';
  }
  function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  function updateStats() {
    setVal('stat-atk-success',  pct(state.atkSuccess));
    setVal('stat-def-accuracy', pct(state.defAccuracy));
    setVal('stat-clean-f1',     f(state.cleanF1));
    setVal('stat-robust-f1',    f(state.robustF1));
    setVal('stat-atk-reward',   state.atkReward.toFixed(3));
    setVal('stat-def-reward',   state.defReward.toFixed(3));
    setVal('stat-episode',      state.episode);
    setVal('stat-step',         step);

    setBar('bar-atk',  state.atkSuccess);
    setBar('bar-def',  state.defAccuracy);
    setBar('bar-clean', state.cleanF1);
    setBar('bar-rob',   state.robustF1);

    const phase = getPhase();
    const phaseBadge = document.getElementById('phase-badge');
    if (phaseBadge) {
      phaseBadge.textContent = phase.name;
      phaseBadge.style.color = phase.color;
    }
  }

  /* ── Animation loop ── */
  let lastTime = 0;
  const MS_PER_TICK = { 1: 50, 2: 25, 3: 10 };

  function loop(ts) {
    if (!running) return;
    const dt = ts - lastTime;
    const interval = MS_PER_TICK[speed] || 50;
    if (dt >= interval) {
      lastTime = ts;
      tick();
      render();
    }
    frameRef = requestAnimationFrame(loop);
  }

  /* ── Controls ── */
  function start() {
    if (running) return;
    running = true;
    logInfo('=== Simulation started ===');
    frameRef = requestAnimationFrame(loop);
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-pause').disabled = false;
  }

  function pause() {
    running = false;
    if (frameRef) cancelAnimationFrame(frameRef);
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-pause').disabled = true;
    logInfo('--- Paused ---');
  }

  function reset() {
    pause();
    step = 0;
    spawnTimer = 0;
    packets = [];
    logLines = [];
    logEl.innerHTML = '';
    state = { atkSuccess: 0.68, defAccuracy: 0.42, cleanF1: 0.42, robustF1: 0.41,
               episode: 0, atkReward: 0, defReward: 0, phase: 'Warmup' };
    updateStats();
    render();
    logInfo('=== Reset. Press Start to begin. ===');
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-pause').disabled = true;
  }

  function setSpeed(s) {
    speed = s;
    document.querySelectorAll('.speed-btn').forEach(b => {
      b.classList.toggle('active', parseInt(b.dataset.speed) === s);
    });
  }

  /* ── Expose to DOM ── */
  window.battleSim = { start, pause, reset, setSpeed };

  /* ── Init ── */
  render();
  logInfo('=== ASTRA Battle Arena Ready ===');
  logInfo('DDPG Attacker vs Dueling-DQN Defender');
  logInfo('Press ▶ Start to begin simulation.');
  updateStats();

})();
