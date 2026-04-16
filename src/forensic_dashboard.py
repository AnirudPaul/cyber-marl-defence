#!/usr/bin/env python3
"""
Live Forensic Dashboard (detailed, attacker controls, SHAP heatmap, IG, LIME, mode toggling)

Usage:
    python -m src.forensic_dashboard --host 127.0.0.1 --port 8050 --explain_url http://127.0.0.1:5001/explain

Features:
- Frontend: single-page Plotly dashboard (SHAP heatmap, SHAP bars, IG bars, LIME rule list)
- Attacker controls: start/stop/status (proxied to explain server attacker endpoints)
- Mode toggling: live MARL attacker or co-trained attacker-defender
- Checkpoint selection: attacker_final.pth or attacker_live.pth
- Server-side proxy endpoints: /send_sample, /status, /attacker/start, /attacker/stop, /attacker/status, /mode
- Rolling client-side history, severity time-series, action distribution, CSV-like log viewer
"""
import argparse
import json
import requests
from flask import Flask, render_template_string, request, jsonify
from urllib.parse import urljoin

APP = Flask(__name__, static_folder=None)
DEFAULT_EXPLAIN_URL = "http://127.0.0.1:5001/explain"
DEFAULT_PING_PATH = "/ping"

# HTML template (Plotly + vanilla JS) - improved HCI with explanatory text and mode controls
TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Adaptive Forensic Dashboard — Detailed</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root{
      --bg: #f6f8fb;
      --card: #fff;
      --muted: #6b7280;
      --accent: #0b74de;
      --ok: #10b981;
      --warn: #f59e0b;
      --danger: #ef4444;
    }
    body { font-family: Inter, Arial, Helvetica, sans-serif; margin:12px; background:var(--bg); color:#111; }
    header { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:12px; }
    .left { flex:1; min-width:360px; }
    .right { width:420px; }
    .row { display:flex; gap:12px; margin-bottom:12px; align-items:flex-start; }
    .card { background:var(--card); padding:12px; border-radius:10px; box-shadow:0 4px 18px rgba(20,20,40,0.06); }
    h3,h4{ margin:6px 0; }
    #log { height:120px; overflow:auto; white-space:pre-wrap; background:#0b0b0b; color:#bfb; padding:8px; border-radius:6px; font-family:monospace; font-size:13px; }
    button { padding:8px 12px; margin-right:6px; cursor:pointer; border-radius:6px; border:1px solid rgba(0,0,0,0.06); background:linear-gradient(#fff,#f2f6ff); }
    .btn-primary { background: linear-gradient(#0b74de,#0558b6); color: white; border:none; }
    .btn-ghost { background: transparent; border:1px solid #e6e9ef; }
    button:disabled { opacity:0.6; cursor:not-allowed; }
    textarea { width:100%; height:92px; font-family:monospace; border-radius:6px; padding:8px; border:1px solid #e6e9ef; }
    pre { background:#fafafa; padding:8px; height:220px; overflow:auto; border-radius:6px; }
    label.small { font-size:0.9em; color:var(--muted); }
    .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:8px; }
    .muted { color:var(--muted); font-size:0.9em; }
    .mini { font-size:0.85em; color:var(--muted); }
    #sidebar .card { margin-bottom:12px; }
    .toolbar { display:flex; gap:8px; align-items:center; }
    table.logtable { width:100%; border-collapse:collapse; font-size:13px; }
    table.logtable th, table.logtable td { text-align:left; padding:6px 8px; border-bottom:1px solid #f0f2f6; }
    .heatwrap { height:260px; }
    .explain-tips { font-size:13px; color:var(--muted); margin-top:6px; }
  </style>
</head>
<body>
<header>
  <div class="left">
    <h3 style="margin:0">Adaptive Forensic Dashboard — Detailed</h3>
    <div class="muted">Explain URL: <code id="explain_url">{{ explain_url }}</code></div>
    <div class="controls" style="margin-top:8px">
      <button id="btn_ping" class="btn-ghost">Ping explain server</button>
      <button id="btn_random" class="btn-ghost">Random sample</button>
      <button id="btn_send" class="btn-primary">Send to explain</button>
      <button id="btn_clear_history" class="btn-ghost">Clear history</button>
      <label class="small"><input id="cb_pretty" type="checkbox" checked> Pretty JSON</label>
      <label class="small">Batch size: <input type="number" id="batch_size" value="1" min="1" style="width:64px"></label>
    </div>
    <div class="card" style="margin-top:10px;">
      <div class="mini">Mode Control</div>
      <div class="controls">
        <select id="mode_select">
          <option value="live_marl">Live MARL Attacker</option>
          <option value="co_trained">Co-trained Attacker-Defender</option>
        </select>
        <select id="checkpoint_select">
          <option value="results/checkpoints/attacker_live.pth">attacker_live.pth</option>
          <option value="results/checkpoints/attacker_final.pth">attacker_final.pth</option>
        </select>
        <button id="btn_switch_mode" class="btn-primary">Switch Mode</button>
      </div>
      <div class="mini muted" id="mode_status">Mode: unknown</div>
    </div>
    <div class="explain-tips card" style="margin-top:10px;">
      <strong>What you're seeing</strong>
      <ul>
        <li><strong>SHAP</strong>: feature importance per action (heatmap) and averaged (bar).</li>
        <li><strong>IG</strong>: integrated gradients per feature (importance towards chosen output).</li>
        <li><strong>LIME</strong>: local rules for the sample—good for quick heuristics.</li>
        <li><strong>Attacker trainer</strong>: start/stop RL-based attacker that learns to evade the defender.</li>
      </ul>
    </div>
  </div>
  <div class="right" id="sidebar">
    <div class="card" style="padding:10px 12px 14px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div class="mini">Server status</div>
          <div id="srv_status" style="font-weight:600">unknown</div>
        </div>
        <div style="text-align:right">
          <div class="mini">Attacker trainer</div>
          <div style="display:flex; gap:6px; margin-top:6px;">
            <button id="att_start" class="btn-ghost">Start</button>
            <button id="att_stop" class="btn-ghost">Stop</button>
            <button id="att_status" class="btn-ghost">Status</button>
          </div>
        </div>
      </div>
    </div>

    <div class="card" id="metrics_card">
      <div class="mini">Quick metrics</div>
      <div style="display:flex; gap:10px; margin-top:8px;">
        <div style="flex:1">
          <div class="muted small">Recent severity</div>
          <div id="recent_sev" style="font-size:18px; font-weight:700">—</div>
        </div>
        <div style="flex:1">
          <div class="muted small">Last action</div>
          <div id="recent_action" style="font-size:18px; font-weight:700">—</div>
        </div>
      </div>
      <div style="margin-top:10px;">
        <div class="muted small">Action distribution</div>
        <div id="action_pie" style="height:160px;"></div>
      </div>
    </div>

    <div class="card">
      <div class="mini">History (last 200)</div>
      <div style="height:160px; overflow:auto; margin-top:8px;">
        <table class="logtable" id="history_table">
          <thead><tr><th>ts</th><th>action</th><th>sev</th></tr></thead>
          <tbody id="history_body"></tbody>
        </table>
      </div>
    </div>
  </div>
</header>

<div class="row">
  <div class="card" style="flex:0.6;">
    <h4 style="margin-top:0">Input sample</h4>
    <div class="small">Paste comma-separated feature vector or click Random. Server expects model input_dim.</div>
    <textarea id="sample_text">[[random]]</textarea>
    <div class="toolbar" style="margin-top:8px;">
      <button id="btn_send_inline" class="btn-ghost">Send inline</button>
      <button id="btn_send_plot" class="btn-primary">Send & plot</button>
      <button id="btn_send_background" class="btn-ghost">Save as SHAP background</button>
      <div class="mini muted" id="input_dim_hint"></div>
    </div>
    <h4 style="margin-top:12px">Server log</h4>
    <div id="log">Ready.</div>
  </div>

  <div class="card" style="flex:1.4;">
    <h4 style="margin-top:0">Visualizations</h4>
    <div style="display:flex; gap:12px;">
      <div style="flex:1">
        <div class="mini muted">SHAP top features (bar)</div>
        <div id="plot_shap_bar" style="height:260px;"></div>
      </div>
      <div style="flex:1">
        <div class="mini muted">SHAP heatmap (actions × features)</div>
        <div id="plot_shap_heat" class="heatwrap"></div>
      </div>
    </div>

    <div style="display:flex; gap:12px; margin-top:12px;">
      <div style="flex:1">
        <div class="mini muted">Integrated Gradients (IG)</div>
        <div id="plot_ig" style="height:240px"></div>
      </div>
      <div style="flex:1">
        <div class="mini muted">LIME rules</div>
        <div id="lime_rules" style="height:240px; overflow:auto; padding-top:4px;"></div>
      </div>
    </div>

    <div style="margin-top:12px;">
      <div class="mini muted">Severity timeline</div>
      <div id="sev_ts" style="height:180px;"></div>
    </div>
  </div>
</div>

<div class="row">
  <div class="card" style="flex:1;">
    <h4 style="margin-top:0">Raw JSON response</h4>
    <pre id="raw_json">{}</pre>
  </div>
  <div class="card" style="width:420px;">
    <h4 style="margin-top:0">Forensic counters & attacker</h4>
    <div id="counters" class="mini muted">no data</div>
    <div style="margin-top:10px;">
      <button id="btn_fetch_att_status" class="btn-ghost">Refresh attacker status</button>
      <pre id="attacker_status" style="height:120px; overflow:auto; margin-top:8px;"></pre>
    </div>
  </div>
</div>
<div class="row">
  <div class="card" style="width:100%;">
    <h4 style="margin-top:0">Attacker Training Metrics (Live)</h4>
    <div id="attacker_metrics" style="height:240px;"></div>
  </div>
</div>
<script>
const explainUrl = "{{ explain_url }}";
const baseExplain = explainUrl.split("/explain")[0];
const pingUrl = (baseExplain + "/ping").replace(/\/+$/,'/');
let history = []; // rolling history
const MAX_HISTORY = 200;
let background_sample = null;
let inferred_input_dim = 15;
window.latest_response = null;

let attackerHist = [];
let attackerPoll = null;

async function pollAttackerMetrics(){
  try{
    const r = await fetch('/attacker/status');
    const j = await r.json();
    if(!j.status) return;
    const s = j.status;
    const t = Date.now()/1000.0;
    attackerHist.push({
      t,
      eps: s.eps ?? 0,
      loss: s.stats?.last_loss ?? 0,
      reward: s.stats?.avg_reward ?? 0,
      replay: s.replay_len ?? 0
    });
    if(attackerHist.length>300) attackerHist.shift();
    renderAttackerChart();
  }catch(e){ console.warn('poll error',e); }
}

function renderAttackerChart(){
  if(attackerHist.length<2) return;
  const t = attackerHist.map(p=>p.t);
  const eps = attackerHist.map(p=>p.eps);
  const loss = attackerHist.map(p=>p.loss);
  const reward = attackerHist.map(p=>p.reward);
  const replay = attackerHist.map(p=>p.replay);
  const traces = [
    {x:t, y:eps, name:'ε (exploration)', mode:'lines', line:{width:2}},
    {x:t, y:reward, name:'avg reward', mode:'lines', line:{width:2}},
    {x:t, y:loss.map(l=>Math.log10(l+1e-8)), name:'log (loss)', mode:'lines', line:{width:2}},
    {x:t, y:replay, name:'replay len', mode:'lines', line:{width:1, dash:'dot'}}
  ];
  const layout = {
    title:'Attacker Training Metrics',
    margin:{t:30,l:40,r:10,b:40},
    height:240,
    legend:{orientation:'h'},
    xaxis:{showgrid:false},
    yaxis:{showgrid:true}
  };
  Plotly.react('attacker_metrics', traces, layout, {displayModeBar:false});
}

// poll every 5 s when dashboard loads
window.addEventListener('load', ()=>{
  attackerPoll = setInterval(pollAttackerMetrics, 5000);
  pollAttackerMetrics();
});

// util logging
function log(msg){ const el = document.getElementById('log'); el.textContent = new Date().toISOString() + '  ' + msg + '\n' + el.textContent; }

// generate random vector of given length
function randVec(len){ const a=[]; for(let i=0;i<len;i++) a.push((Math.random()*2-1).toFixed(4)); return a; }

// get textarea array
function getSampleArray(){
  let text = document.getElementById('sample_text').value.trim();
  if(text === '[[random]]'){
    const dim = inferred_input_dim || 15;
    return randVec(dim).map(x=>parseFloat(x));
  }
  const rows = text.split('\n').map(r=>r.trim()).filter(r=>r.length>0);
  const batch = [];
  for (const r of rows){
    const parts = r.split(',').map(s=>s.trim()).filter(s=>s.length>0);
    const nums = parts.map(p => Number(p));
    batch.push(nums);
  }
  const requestedBatch = parseInt(document.getElementById('batch_size').value) || 1;
  if(batch.length === 1 && requestedBatch > 1){
    const out = Array.from({length: requestedBatch}, ()=>batch[0]);
    return out;
  }
  return batch;
}

async function safeFetch(path, opts){
  try{
    const r = await fetch(path, opts);
    const j = await r.json().catch(()=>({raw: 'non-json response'}));
    return {ok: r.ok, status: r.status, json: j};
  }catch(e){ return {ok:false, error: e.toString()}; }
}

// ping upstream /status (proxied)
document.getElementById('btn_ping').onclick = async ()=>{
  log('[→] pinging explain server proxy /status ...');
  const r = await safeFetch('/status');
  if(!r.ok){ log('[!] ping failed: ' + (r.error||JSON.stringify(r.json))); alert('Ping failed: '+JSON.stringify(r)); return; }
  log('[←] ping result: ' + JSON.stringify(r.json));
  document.getElementById('srv_status').textContent = r.json.ok ? 'up' : 'down';
  if(r.json.result && typeof r.json.result === 'object' && r.json.result.input_dim){
    inferred_input_dim = r.json.result.input_dim;
    document.getElementById('input_dim_hint').textContent = 'model input_dim = ' + inferred_input_dim;
  }
  alert('Ping: ' + JSON.stringify(r.json));
};

// call once on load
(async ()=> {
  const r = await safeFetch('/status');
  if(r.ok && r.json && r.json.result && r.json.result.input_dim){
    inferred_input_dim = r.json.result.input_dim;
    document.getElementById('input_dim_hint').textContent = 'model input_dim = ' + inferred_input_dim;
    log('[i] inferred input dim: ' + inferred_input_dim);
  } else {
    log('[i] Could not infer input dim from upstream status.');
  }
})();

document.getElementById('btn_random').onclick = ()=> {
  const defaultLen = inferred_input_dim || 15;
  document.getElementById('sample_text').value = randVec(defaultLen).join(', ');
};

async function sendSample(payload, autoPlot=false, useAsBackground=false){
  log('[→] Sending sample to dashboard proxy /send_sample ...');
  const r = await safeFetch('/send_sample', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
  if(!r.ok){
    log('[!] send failed: ' + (r.error || JSON.stringify(r.json)));
    alert('Send failed: ' + (r.error || JSON.stringify(r.json)));
    return null;
  }
  const j = r.json;
  window.latest_response = j;
  const rec = {
    ts: (j.forensic_decision && j.forensic_decision.record && j.forensic_decision.record.ts) || Date.now()/1000.0,
    action: j.forensic_decision ? j.forensic_decision.action : 'unknown',
    severity: j.forensic_decision ? (j.forensic_decision.severity || 0) : 0,
    payload: payload,
    response: j
  };
  history.unshift(rec);
  if(history.length>MAX_HISTORY) history.pop();
  updateHistoryUI();
  document.getElementById('raw_json').textContent = JSON.stringify(j, null, document.getElementById('cb_pretty').checked ? 2 : 0);
  renderAll(j);
  log('[✓] Received response (status ' + r.status + ')');
  if(useAsBackground){
    background_sample = payload.features;
    alert('Background sample stored for SHAP/LIME usage.');
  }
  return j;
}

document.getElementById('btn_send').onclick = async ()=>{
  const arr = getSampleArray();
  if(arr.length === 0){ alert('Provide a sample'); return; }
  const features = Array.isArray(arr[0]) ? arr[0] : arr;
  const payload = { id: "dashboard_sample", features: features, meta: {source:"dashboard"}, batch: 1 };
  await sendSample(payload, true, false);
};

document.getElementById('btn_send_inline').onclick = async ()=>{
  const arr = getSampleArray();
  if(arr.length === 0){ alert('Provide a sample'); return; }
  const features = Array.isArray(arr[0]) ? arr[0] : arr;
  const payload = { id: "dashboard_sample", features: features, meta: {source:"dashboard"}, batch: 1 };
  await sendSample(payload, false, false);
};

document.getElementById('btn_send_plot').onclick = async ()=>{
  const arr = getSampleArray();
  if(arr.length === 0){ alert('Provide a sample'); return; }
  const features = Array.isArray(arr[0]) ? arr[0] : arr;
  const payload = { id: "dashboard_sample", features: features, meta: {source:"dashboard"}, batch: 1 };
  await sendSample(payload, true, false);
};

document.getElementById('btn_send_background').onclick = async ()=>{
  const arr = getSampleArray();
  if(arr.length === 0){ alert('Provide a sample'); return; }
  const features = Array.isArray(arr[0]) ? arr[0] : arr;
  const payload = { id: "dashboard_sample", features: features, meta: {source:"dashboard"}, batch: 1 };
  await sendSample(payload, false, true);
};

// attacker control proxies
document.getElementById('att_start').onclick = async ()=>{
  log('[→] starting attacker trainer (proxy) ...');
  const r = await safeFetch('/attacker/start', { method:'POST' });
  if(!r.ok){ log('[!] attacker start failed: ' + JSON.stringify(r)); alert('Attacker start failed'); return; }
  document.getElementById('attacker_status').textContent = JSON.stringify(r.json,null,2);
  alert('Attacker trainer started');
};
document.getElementById('att_stop').onclick = async ()=>{
  log('[→] stopping attacker trainer (proxy) ...');
  const r = await safeFetch('/attacker/stop', { method:'POST' });
  document.getElementById('attacker_status').textContent = JSON.stringify(r.json || r.error,null,2);
  alert('Attacker trainer stopped');
};
document.getElementById('att_status').onclick = async ()=>{
  log('[→] fetching attacker trainer status (proxy) ...');
  const r = await safeFetch('/attacker/status');
  document.getElementById('attacker_status').textContent = JSON.stringify(r.json || r.error,null,2);
  alert('Attacker status refreshed');
};
document.getElementById('btn_fetch_att_status').onclick = ()=> document.getElementById('att_status').click();

// mode switch handler (added)
document.getElementById('btn_switch_mode').onclick = async ()=>{
  const mode = document.getElementById('mode_select').value;
  const checkpoint = document.getElementById('checkpoint_select').value;
  log('[→] Switching to mode ' + mode + ' with checkpoint ' + checkpoint + ' ...');
  const r = await safeFetch('/mode', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode: mode, checkpoint: checkpoint}) });
  if(!r.ok){
    log('[!] mode switch failed: ' + (r.error || JSON.stringify(r.json)));
    alert('Mode switch failed: ' + (r.error || JSON.stringify(r.json)));
    return;
  }
  document.getElementById('mode_status').textContent = 'Mode: ' + r.json.mode + ' (checkpoint: ' + r.json.checkpoint + ')';
  log('[✓] Mode switched to ' + r.json.mode);
  alert('Mode switched to ' + r.json.mode);
};

function updateHistoryUI(){
  const body = document.getElementById('history_body');
  body.innerHTML = '';
  let counts = {};
  for(let i=0;i<history.length;i++){
    const r = history[i];
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${new Date(r.ts*1000).toISOString().slice(11,23)}</td><td>${r.action}</td><td>${r.severity.toFixed(3)}</td>`;
    body.appendChild(tr);
    counts[r.action] = (counts[r.action]||0)+1;
  }
  renderActionPie(counts);
  renderSeverityTS();
  if(history.length>0){
    document.getElementById('recent_sev').textContent = history[0].severity.toFixed(3);
    document.getElementById('recent_action').textContent = history[0].action;
  }
}

function renderActionPie(counts){
  const labels = Object.keys(counts);
  const values = labels.map(l=>counts[l]);
  const data = [{ labels, values, type:'pie', textinfo:'label+percent', hole:0.4 }];
  const layout = { height:160, margin:{t:10,b:10,l:10,r:10} };
  Plotly.react('action_pie', data, layout, {displayModeBar:false});
}

function renderSeverityTS(){
  const x = history.map(h=>new Date(h.ts*1000));
  const y = history.map(h=>h.severity);
  const data = [{ x, y, mode:'lines+markers', name:'severity' }];
  const layout = { height:180, margin:{t:10,b:30,l:40,r:10}, xaxis:{showgrid:false}, yaxis:{range:[0,1]} };
  Plotly.react('sev_ts', data, layout, {displayModeBar:false});
}

function renderAll(json){
  renderSHAP(json.explain?.shap);
  renderLIME(json.explain?.lime);
  renderIG(json.explain?.integrated_gradients);
  if(json.forensic_counters) {
    document.getElementById('counters').textContent = JSON.stringify(json.forensic_counters, null, 2);
  }
}

// SHAP renderer
function renderSHAP(shap){
  if(!shap || !shap.mean_abs){
    Plotly.react('plot_shap_bar', [{ x:[0], y:['No SHAP'], type:'bar' }], {});
    Plotly.react('plot_shap_heat', [{ z:[[0]], x:['f0'], y:['a0'], type:'heatmap' }], {});
    return;
  }
  let arr = shap.mean_abs;
  let mat, featCount, actionCount;
  if(Array.isArray(arr) && Array.isArray(arr[0]) && arr.length>1){
    actionCount = arr.length;
    featCount = arr[0].length;
    mat = arr;
  } else {
    featCount = arr.length || 0;
    actionCount = 1;
    mat = [arr];
  }
  const avg = Array.from({length: featCount}, (_,i)=>0.0);
  for(let a=0;a<mat.length;a++){ for(let f=0;f<featCount;f++){ avg[f] += Math.abs(mat[a][f]); } }
  for(let f=0;f<featCount;f++) avg[f] /= mat.length;
  const k = Math.min(18, featCount);
  const idx = avg.map((v,i)=>[Math.abs(v), v, i]).sort((a,b)=>b[0]-a[0]).slice(0,k).reverse();
  const labels = idx.map(x=>'f'+x[2]);
  const values = idx.map(x=>x[1]);
  const bartrace = { x: values, y: labels, orientation:'h', type:'bar', marker:{color: values.map(v=> v>=0 ? '#1f77b4' : '#d62728')} };
  Plotly.react('plot_shap_bar', [bartrace], { margin:{l:120,t:20,b:40}, height:260 });
  const z = mat;
  const x = Array.from({length:featCount}, (_,i)=>'f'+i);
  const y = Array.from({length:actionCount}, (_,i)=>'a'+i);
  const heat = [{ z, x, y, type:'heatmap', colorscale:'YlGnBu', reversescale:false, showscale:true }];
  const heatlayout = { margin:{l:40,t:30,b:80}, height:260 };
  Plotly.react('plot_shap_heat', heat, heatlayout);
}

function renderLIME(lime){
  const container = document.getElementById('lime_rules');
  container.innerHTML = '';
  if(!lime || !lime.lime_results || lime.lime_results.length===0){ container.textContent = 'No LIME output'; return; }
  const r = lime.lime_results[0];
  const list = r.lime_as_list || r.lime_features || [];
  const ul = document.createElement('ul');
  for(const item of list.slice(0,16)){
    const li = document.createElement('li');
    if(Array.isArray(item) && typeof item[0]==='string') li.textContent = `${item[0]} → ${Number(item[1]).toFixed(4)}`;
    else if(Array.isArray(item) && item.length>=2) li.textContent = `f${item[0]} → ${Number(item[1]).toFixed(4)}`;
    else li.textContent = JSON.stringify(item);
    ul.appendChild(li);
  }
  container.appendChild(ul);
}

function renderIG(ig){
  if(!ig || !ig.ig_values){ Plotly.react('plot_ig', [{ x:[0], y:['No IG'], type:'bar'}], {}); return; }
  const a = ig.ig_values;
  if(!a){ Plotly.react('plot_ig', [{ x:[0], y:['No IG'], type:'bar'}], {}); return; }
  let vals;
  if(Array.isArray(a) && Array.isArray(a[0]) && Array.isArray(a[0][0])){
    const sample = a[0];
    const feats = sample[0].length;
    vals = Array.from({length:feats}, ()=>0.0);
    for(const act of sample) for(let f=0;f<feats;f++) vals[f]+= act[f];
  } else if(Array.isArray(a) && Array.isArray(a[0])){
    vals = a[0];
  } else { vals = a; }
  const labels = vals.map((_,i)=>'f'+i).reverse();
  const trace = { x: vals.slice().reverse(), y: labels, orientation:'h', type:'bar', marker:{color:'#ff7f0e'} };
  Plotly.react('plot_ig', [trace], { margin:{l:120,t:20,b:40}, height:240 });
}

// init empty plots
renderSHAP(null);
renderIG(null);
renderLIME(null);
</script>
</body>
</html>
"""

# ------------------------------
# Helper macros (server-side proxy)
# ------------------------------
def safe_json_post(url, payload, timeout=20):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, {"raw_text": r.text}
    except Exception as e:
        return False, str(e)

def safe_json_get(url, timeout=6):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, {"raw_text": r.text}
    except Exception as e:
        return False, str(e)

# ------------------------------
# Flask endpoints (proxy to explain server)
# ------------------------------
@APP.route("/")
def index():
    explain_url = APP.config.get("EXPLAIN_URL", DEFAULT_EXPLAIN_URL)
    return render_template_string(TEMPLATE, explain_url=explain_url)

@APP.route("/status", methods=["GET"])
def status():
    explain_url = APP.config.get("EXPLAIN_URL", DEFAULT_EXPLAIN_URL)
    base = explain_url.split("/explain")[0] if "/explain" in explain_url else explain_url
    ping_url = urljoin(base + "/", DEFAULT_PING_PATH.lstrip("/"))
    ok, result = safe_json_get(ping_url)
    return jsonify({
        "upstream_ping_url": ping_url,
        "ok": bool(ok),
        "result": result
    }), (200 if ok else 502)

@APP.route("/send_sample", methods=["POST"])
def send_sample():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"status":"error", "error":"invalid json payload"}), 400
    explain_url = APP.config.get("EXPLAIN_URL", DEFAULT_EXPLAIN_URL)
    ok, resp = safe_json_post(explain_url, payload)
    if not ok:
        return jsonify({"status":"error", "error": str(resp)}), 502
    out = resp if isinstance(resp, dict) else {"raw": resp}
    out.setdefault("status", "ok")
    return jsonify(out), 200

# attacker controls proxied
def _proxy_att(path_suffix, method="GET"):
    explain_root = APP.config.get("EXPLAIN_URL", DEFAULT_EXPLAIN_URL).split("/explain")[0]
    url = urljoin(explain_root + "/", f"attacker/train/{path_suffix}")
    try:
        if method.upper() == "POST":
            r = requests.post(url, timeout=10)
        else:
            r = requests.get(url, timeout=6)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, {"raw_text": r.text}
    except Exception as e:
        return False, str(e)

@APP.route("/attacker/start", methods=["POST"])
def att_start():
    ok, resp = _proxy_att("start", method="POST")
    if not ok:
        return jsonify({"error": resp}), 502
    return jsonify(resp), 200

@APP.route("/attacker/stop", methods=["POST"])
def att_stop():
    ok, resp = _proxy_att("stop", method="POST")
    if not ok:
        return jsonify({"error": resp}), 502
    return jsonify(resp), 200

@APP.route("/attacker/status", methods=["GET"])
def att_status():
    ok, resp = _proxy_att("status", method="GET")
    if not ok:
        return jsonify({"error": resp}), 502
    return jsonify(resp), 200

@APP.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status":"ok", "service":"forensic_dashboard"}), 200

@APP.route("/mode", methods=["POST"])
def switch_mode():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"status":"error", "error":"invalid json payload"}), 400
    explain_url = APP.config.get("EXPLAIN_URL", DEFAULT_EXPLAIN_URL)
    base = explain_url.split("/explain")[0] if "/explain" in explain_url else explain_url
    mode_url = urljoin(base + "/", "mode")
    ok, resp = safe_json_post(mode_url, payload)
    if not ok:
        return jsonify({"status":"error", "error": str(resp)}), 502
    return jsonify(resp), 200

# ------------------------------
# Main
# ------------------------------
def main(host="127.0.0.1", port=8050, explain_url=DEFAULT_EXPLAIN_URL):
    APP.config["EXPLAIN_URL"] = explain_url
    print("[+] Forensic Dashboard starting")
    print("    explain_url =", explain_url)
    APP.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--explain_url", default=DEFAULT_EXPLAIN_URL)
    args = parser.parse_args()
    main(args.host, args.port, args.explain_url)