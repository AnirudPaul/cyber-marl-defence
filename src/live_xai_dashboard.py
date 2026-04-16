"""
PHASE 14 — 🔬 Live XAI Forensics Dashboard
-----------------------------------------------------
Unified dashboard for SHAP + LIME + Integrated Gradients,
with real-time streaming from forensics_server.py via WebSocket.

- Aggregated importance (top features)
- Per-sample comparison
- Live forensic updates from /forensics namespace
- Adjustable fusion weights
- Auto-refreshing visualizations

Dependencies:
  flask, flask_socketio, eventlet, plotly, numpy, pandas
"""

from flask import Flask, render_template_string
import json
import numpy as np
import pandas as pd
from pathlib import Path
import random

# ===========================
# Demo setup (replace with your real data)
# ===========================
N_SAMPLES = 10
N_FEATURES = 15
feature_names = [f"Feature_{i}" for i in range(N_FEATURES)]

def gen_random_explanations():
    """Generate random SHAP/LIME/IG values for demo mode."""
    rng = np.random.default_rng(42)
    return {
        "shap": rng.normal(0, 1, (N_SAMPLES, N_FEATURES)),
        "lime": rng.normal(0, 0.7, (N_SAMPLES, N_FEATURES)),
        "ig":   rng.normal(0, 0.5, (N_SAMPLES, N_FEATURES))
    }

explainers = gen_random_explanations()
agg = {k: np.mean(np.abs(v), axis=0).tolist() for k, v in explainers.items()}

# ===========================
# Flask app
# ===========================
app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>🧠 Live XAI Forensics Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <style>
    body {
      background-color: #0d1117;
      color: #e6edf3;
      font-family: 'Segoe UI', sans-serif;
      margin: 0; padding: 0;
    }
    h1 {
      color: #58a6ff;
      text-align: center;
      padding: 20px;
    }
    .container {
      padding: 20px;
      max-width: 1400px;
      margin: auto;
    }
    .controls {
      background: #0f171b;
      padding: 10px;
      border-radius: 8px;
      margin-bottom: 20px;
    }
    select, input, button {
      background: #111a20;
      color: #e6edf3;
      border: 1px solid #24303a;
      border-radius: 6px;
      padding: 6px;
      margin-right: 6px;
    }
    .plot-box {
      background: #0f171b;
      padding: 12px;
      border-radius: 10px;
      margin-bottom: 24px;
    }
    .alert {
      background: #1c2d17;
      color: #9cf292;
      padding: 10px;
      border-radius: 5px;
      font-size: 14px;
      margin-bottom: 10px;
    }
  </style>
</head>
<body>
  <h1>🧠 Live XAI Forensics Dashboard</h1>
  <div class="container">
    <div class="controls">
      <b>Explainer:</b>
      <select id="explainer">
        <option value="shap">SHAP</option>
        <option value="lime">LIME</option>
        <option value="ig">Integrated Gradients</option>
        <option value="fusion">Fusion (weighted)</option>
      </select>

      <b>Sample:</b>
      <select id="sample"></select>

      <b>Fusion Weights:</b>
      SHAP <input type="number" id="w_shap" step="0.05" min="0" max="1" value="0.4">
      LIME <input type="number" id="w_lime" step="0.05" min="0" max="1" value="0.3">
      IG <input type="number" id="w_ig" step="0.05" min="0" max="1" value="0.3">
      <button id="apply">Apply</button>
    </div>

    <div id="liveAlert" class="alert" style="display:none;">
      🔴 <b>Live update received:</b> <span id="liveInfo"></span>
    </div>

    <div class="plot-box" id="agg_plot"></div>
    <div class="plot-box" id="sample_plot"></div>
  </div>

<script>
  // ====== Data from server ======
  const DATA = {{ data_json | safe }};
  const n_samples = DATA.n_samples;
  const n_features = DATA.n_features;
  const feature_names = DATA.feature_names;
  const explainers = DATA.explainers;
  const agg = DATA.agg;

  // ====== Populate sample dropdown ======
  const sampSel = document.getElementById('sample');
  for (let i = 0; i < n_samples; i++) {
    let opt = document.createElement('option');
    opt.value = i; opt.textContent = 'Sample ' + i;
    sampSel.appendChild(opt);
  }

  // ====== Plotly functions ======
  function plotAggregate(name) {
    let vals;
    if (name === 'fusion') {
      vals = fusionAggregate();
    } else {
      vals = agg[name];
    }
    const df = feature_names.map((f, i) => ({Feature:f, Value:vals[i]}));
    const trace = {
      x: df.map(d => d.Value),
      y: df.map(d => d.Feature),
      orientation: 'h',
      type: 'bar',
      marker: {color: '#58a6ff'}
    };
    const layout = {
      title: name.toUpperCase() + ' Aggregate Importance',
      paper_bgcolor: '#0d1117',
      plot_bgcolor: '#0f171b',
      font: {color:'#e6edf3'}
    };
    Plotly.newPlot('agg_plot', [trace], layout);
  }

  function plotSample(name, idx) {
    let vals;
    if (name === 'fusion') {
      vals = fusionSample(idx);
    } else {
      vals = explainers[name][idx].map(Math.abs);
    }
    const df = feature_names.map((f, i) => ({Feature:f, Value:vals[i]}));
    const trace = {
      x: df.map(d => d.Value),
      y: df.map(d => d.Feature),
      orientation: 'h',
      type: 'bar',
      marker: {color:'#9bd0ff'}
    };
    const layout = {
      title: name.toUpperCase() + ' Sample ' + idx,
      paper_bgcolor: '#0d1117',
      plot_bgcolor: '#0f171b',
      font:{color:'#e6edf3'}
    };
    Plotly.newPlot('sample_plot', [trace], layout);
  }

  // ====== Fusion logic ======
  function fusionSample(idx){
    const w_shap = parseFloat(document.getElementById('w_shap').value);
    const w_lime = parseFloat(document.getElementById('w_lime').value);
    const w_ig = parseFloat(document.getElementById('w_ig').value);
    const sumw = w_shap + w_lime + w_ig || 1;
    const s = explainers['shap'][idx];
    const l = explainers['lime'][idx];
    const g = explainers['ig'][idx];
    return s.map((_,i) => (w_shap*Math.abs(s[i]) + w_lime*Math.abs(l[i]) + w_ig*Math.abs(g[i])) / sumw);
  }
  function fusionAggregate(){
    const w_shap = parseFloat(document.getElementById('w_shap').value);
    const w_lime = parseFloat(document.getElementById('w_lime').value);
    const w_ig = parseFloat(document.getElementById('w_ig').value);
    const sumw = w_shap + w_lime + w_ig || 1;
    return feature_names.map((_,i) => (w_shap*agg['shap'][i] + w_lime*agg['lime'][i] + w_ig*agg['ig'][i]) / sumw);
  }

  // ====== Event bindings ======
  const explSel = document.getElementById('explainer');
  explSel.addEventListener('change', ()=>{
    const name = explSel.value;
    plotAggregate(name);
    plotSample(name, parseInt(sampSel.value||0));
  });
  sampSel.addEventListener('change', ()=>{
    const name = explSel.value;
    plotSample(name, parseInt(sampSel.value||0));
  });
  document.getElementById('apply').addEventListener('click', ()=>{
    const name = explSel.value;
    plotAggregate(name);
    plotSample(name, parseInt(sampSel.value||0));
  });

  // ====== SocketIO for Live Updates ======
  const socket = io("http://127.0.0.1:5001/forensics");
  socket.on("connect", function() {
    console.log("Connected to forensics server!");
  });
  socket.on("forensic_explanation", function(payload){
    console.log("Live explanation:", payload);
    document.getElementById('liveAlert').style.display = 'block';
    document.getElementById('liveInfo').textContent = "Sample: " + (payload.sample_id || "unknown") + ", " + JSON.stringify(payload.meta);
    // update SHAP summary plot if present
    if (payload.shap_summary) {
      const trace = {x: payload.shap_summary, y: feature_names, orientation:'h', type:'bar', marker:{color:'#ff7f50'}};
      Plotly.newPlot('agg_plot', [trace], {title:'LIVE SHAP Update', paper_bgcolor:'#0d1117', plot_bgcolor:'#0f171b', font:{color:'#e6edf3'}});
    }
  });

  // ====== Initial plots ======
  plotAggregate('shap');
  plotSample('shap', 0);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    data = {
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "feature_names": feature_names,
        "explainers": {k: v.tolist() for k, v in explainers.items()},
        "agg": agg
    }
    return render_template_string(TEMPLATE, data_json=json.dumps(data))


if __name__ == "__main__":
    print("🚀 Running Live XAI Forensics Dashboard → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
