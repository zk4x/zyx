// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Static assets of the visualizer web page.

/// Vendored vis-network standalone build — everything runs fully locally.
pub(super) const VIS_NETWORK_JS: &[u8] = include_bytes!("vis-network.min.js");

/// The single-page UI: tabs for graphs, plan graph on the left, kernel IR
/// and generated code columns on the right.
pub(super) const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>zyx graph viz</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #14161a; color: #d8dce2; font-family: "JetBrains Mono", "Fira Mono", monospace; font-size: 13px; }
  header { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: #1b1e24; border-bottom: 1px solid #2a2e36; }
  .tab { padding: 6px 14px; background: #22262d; border: 1px solid #2a2e36; cursor: pointer; user-select: none; }
  .tab.active { background: #3a4150; color: #fff; }
  main { display: flex; gap: 0; height: calc(100vh - 44px); padding: 8px; }
  section { position: relative; background: #1b1e24; border: 1px solid #2a2e36; display: flex; flex-direction: column; min-width: 150px; overflow: hidden; flex: 0 0 auto; }
  section h2 { margin: 0; padding: 6px 10px; font-size: 12px; color: #9aa3af; border-bottom: 1px solid #2a2e36; text-transform: uppercase; letter-spacing: .08em; }
  #network { position: relative; flex: 1; min-height: 0; overflow-x: auto; overflow-y: hidden; display: flex; }
  #neigh { padding: 4px 10px; border-bottom: 1px solid #2a2e36; color: #9aa3af; font-size: 12px; }
  .lane { flex: 1 0 220px; min-width: 200px; display: flex; flex-direction: column; border-right: 1px solid #2a2e36; min-height: 0; }
  .lanehead { padding: 4px 10px; color: #9aa3af; font-size: 12px; border-bottom: 1px solid #2a2e36; white-space: nowrap; }
  .lanescroll { flex: 1; overflow-y: auto; position: relative; min-height: 0; }
  .lanescroll canvas { position: sticky; top: 0; display: block; width: 100%; }
  .divider { flex: 0 0 5px; cursor: col-resize; background: transparent; }
  .divider:hover, .divider.dragging { background: #3a4150; }
  #plan_section { width: 30%; }
  #sched_section { width: 18%; }
  #ir_section { width: 18%; }
  #asm_section { width: 24%; }
  pre { flex: 1; margin: 0; overflow: auto; padding: 10px; font-size: 12px; line-height: 1.45; white-space: pre; color: #cdd3db; }
  .c-kw { color: #fff; font-weight: bold; }
  .c-op { color: #ff5c57; }
  .c-move { color: #9aedfe; }
  .c-idx { color: #57c7ff; }
  .c-stack { color: #ffb86c; }
  .c-const { color: #ff6ac1; }
  .c-type { color: #8b949e; }
  .c-com { color: #686868; }
  .colhead { display: flex; align-items: center; gap: 8px; padding: 6px 10px; border-bottom: 1px solid #2a2e36; }
  #device { color: #7fb2ff; }
  select { background: #22262d; color: #d8dce2; border: 1px solid #2a2e36; padding: 3px 6px; }
</style>
</head>
<body>
<header><span style="color:#9aa3af">graphs:</span><span id="tabs"></span></header>
<main>
  <section id="plan_section"><h2>Plan</h2><div id="neigh">no graph</div><div id="network"></div></section>
  <div class="divider"></div>
  <section id="sched_section"><h2>sched IR (pre-linearize)</h2><pre id="sched">click a kernel</pre></section>
  <div class="divider"></div>
  <section id="ir_section"><h2>optimized IR</h2><pre id="ir"></pre></section>
  <div class="divider"></div>
  <section id="asm_section">
    <div class="colhead"><h2 style="border:none;padding:0;margin:0">generated code</h2>
      <span id="device"></span>
      <select id="target">
        <option value="cuda">CUDA C</option>
        <option value="ptx">PTX</option>
        <option value="opencl">OpenCL</option>
        <option value="c">C</option>
        <option value="spirv">SPIR-V</option>
      </select>
    </div>
    <pre id="asm"></pre>
  </section>
</main>
<script>
let curGraph = -1, graphData = null, curKernel = null;

async function refreshTabs() {
  let graphs;
  try { graphs = await (await fetch('/api/graphs')).json(); } catch { return; }
  const el = document.getElementById('tabs');
  el.innerHTML = '';
  for (const g of graphs) {
    const t = document.createElement('span');
    t.className = 'tab' + (g.id === curGraph ? ' active' : '');
    t.textContent = g.name + ' (' + g.kernels + ')';
    t.onclick = () => openGraph(g.id);
    el.appendChild(t);
  }
  if (curGraph < 0 && graphs.length) openGraph(graphs[0].id);
}

async function openGraph(id) {
  curGraph = id;
  curKernel = null;
  document.getElementById('sched').textContent = 'click a kernel';
  document.getElementById('ir').textContent = '';
  document.getElementById('asm').textContent = '';
  document.getElementById('device').textContent = '';
  try { graphData = await (await fetch('/api/graph/' + id)).json(); } catch { return; }
  try { draw(); } catch (e) { document.getElementById('sched').textContent = 'draw error: ' + e; }
  refreshTabs();
}

 // Lane columns by (device, queue): each column is a virtualized canvas list,
 // so even thousands of kernels render instantly. No layout library, no physics.
const ROW_H = 36;
const LAUNCH_BASE = 1 << 30;
let laneCols = []; // {key, rows:[{kid,text}], scroll, cv}
let classLabel = new Map(), kIn = new Map(), kOut = new Map();

function pushEdge(m, kid, cid) {
  let a = m.get(kid);
  if (!a) { a = []; m.set(kid, a); }
  if (a.indexOf(cid) < 0) a.push(cid);
}

function fmtNs(n) {
  if (n === null || n === undefined) return '?';
  if (n < 1000) return n + 'ns';
  if (n < 1000000) return (n / 1000).toFixed(1) + 'µs';
  return (n / 1000000).toFixed(2) + 'ms';
}

function draw() {
  laneCols = []; classLabel = new Map(); kIn = new Map(); kOut = new Map();
  const rows = [];
  for (const n of graphData.nodes) {
    if (n.kernel >= 0) {
      const k = n.kernel;
      rows.push({
        kid: k,
        text: n.label.replace(/\n/g, ' '),
        dev: graphData.devices[k] || 'AOT',
        lane: (graphData.lanes && graphData.lanes[k] !== null && graphData.lanes[k] !== undefined) ? graphData.lanes[k] : 0,
        nanos: graphData.nanos ? graphData.nanos[k] : null,
      });
    } else classLabel.set(n.id, n.label.replace(/\n/g, ' '));
  }
  rows.sort((a, b) => a.kid - b.kid);
  for (const e of graphData.edges) {
    if (e[2] === 'store') pushEdge(kOut, e[0] - LAUNCH_BASE, e[1]);
    else pushEdge(kIn, e[1] - LAUNCH_BASE, e[0]);
  }
  const groups = new Map();
  for (const r of rows) {
    const key = r.dev + ' q' + r.lane;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  const net = document.getElementById('network');
  net.innerHTML = '';
  laneCols = [...groups].map(([key, rs]) => ({ key, rows: rs, scroll: null, cv: null }));
  for (const c of laneCols) {
    const total = c.rows.reduce((s, r) => s + (r.nanos || 0), 0);
    const div = document.createElement('div');
    div.className = 'lane';
    div.innerHTML = '<div class="lanehead">' + c.key + ' · ' + c.rows.length + ' · ' + fmtNs(total) + '</div>' +
      '<div class="lanescroll"><canvas></canvas><div class="spacer" style="height:' + (c.rows.length * ROW_H + 8) + 'px"></div></div>';
    net.appendChild(div);
    c.scroll = div.querySelector('.lanescroll');
    c.cv = div.querySelector('canvas');
    c.scroll.onscroll = () => drawCol(c);
    c.cv.onclick = e => onColClick(e, c);
  }
  document.getElementById('neigh').textContent = rows.length + ' kernels in ' + laneCols.length + ' lanes — click one';
  sizeCanvases();
}

function sizeCanvases() {
  for (const c of laneCols) {
    const dpr = window.devicePixelRatio || 1;
    c.cv.width = Math.max(1, Math.floor(c.scroll.clientWidth * dpr));
    c.cv.height = Math.max(1, Math.floor(c.scroll.clientHeight * dpr));
    c.cv.style.height = c.scroll.clientHeight + 'px';
    drawCol(c);
  }
}
window.onresize = sizeCanvases;

function drawCol(c) {
  if (!c.cv || !c.rows.length) return;
  const dpr = window.devicePixelRatio || 1;
  const ctx = c.cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = c.scroll.clientWidth, H = c.scroll.clientHeight;
  ctx.fillStyle = '#1b1e24';
  ctx.fillRect(0, 0, W, H);
  ctx.font = '12px monospace';
  const start = Math.max(0, Math.floor(c.scroll.scrollTop / ROW_H) - 1);
  const end = Math.min(c.rows.length, start + Math.ceil(H / ROW_H) + 2);
  for (let i = start; i < end; i++) {
    const y = i * ROW_H - c.scroll.scrollTop, r = c.rows[i], sel = r.kid === curKernel;
    ctx.fillStyle = sel ? '#6b3a4a' : '#3a2330';
    ctx.strokeStyle = sel ? '#ff8ab0' : '#c76a8a';
    ctx.fillRect(6, y + 3, W - 12, ROW_H - 6);
    ctx.strokeRect(6.5, y + 3.5, W - 13, ROW_H - 7);
    ctx.fillStyle = '#e6e9ee';
    ctx.fillText('k' + r.kid + ' ' + fmtNs(r.nanos) + ' ' + r.text, 12, y + 22);
  }
}

function drawRows() {
  for (const c of laneCols) drawCol(c);
}

function onColClick(e, c) {
  const idx = Math.floor((c.scroll.scrollTop + e.clientY - c.cv.getBoundingClientRect().top) / ROW_H);
  if (idx >= 0 && idx < c.rows.length) selectKernel(c.rows[idx].kid);
}

async function loadStage(stage, target) {
  const q = stage === 'asm' ? '?target=' + target : '';
  const res = await fetch('/api/kernel/' + curGraph + '/' + curKernel + '/' + stage + q);
  return await res.text();
}

// Syntax highlighting for the kernel IR, mimicking the ZYX_DEBUG terminal colors.
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function hlIR(text) {
  return text.split('\n').map(line => {
    const ci = line.indexOf('//');
    let s = esc(ci >= 0 ? line.slice(0, ci) : line);
    s = s.replace(/\b(for|if)(?=\s|\()/g, '<span class="c-kw">$1</span>');
    s = s.replace(/(group|local|warp)_index/g, '<span class="c-idx">$1_index</span>');
    s = s.replace(/\b(reduce_tile_\w+|reduce|matmul_tile|transpose_tile|param|storage|barrier|load|store)\b/g, '<span class="c-op">$1</span>');
    s = s.replace(/\b(reshape|expand|permute|pad|narrow|flip)\b/g, '<span class="c-move">$1</span>');
    s = s.replace(/\b(stack|asm|wmma)\b/g, '<span class="c-stack">$1</span>');
    s = s.replace(/\.s[0-9a-f]+\b/g, '<span class="c-stack">$&</span>');
    s = s.replace(/: ?([a-z0-9]+)(?= =)/, ': <span class="c-type">$1</span>');
    s = s.replace(/(?<![a-zA-Z])0x[0-9A-Fa-f]+/g, '<span class="c-const">$&</span>');
    s = s.replace(/(?<![\w.])-?\d+\.\d+(?:e-?\d+)?f?(?![\w.])/g, '<span class="c-const">$&</span>');
    s = s.replace(/(?<![\w.])-?\d+(?![\w.])/g, '<span class="c-const">$&</span>');
    if (ci >= 0) s += '<span class="c-com">' + esc(line.slice(ci)) + '</span>';
    return s;
  }).join('\n');
}

async function selectKernel(k) {
  curKernel = k;
  drawRows();
  document.getElementById('device').textContent = graphData.devices[k] || '';
  const names = m => (m.get(k) || []).map(c => classLabel.get(c) || ('c' + c)).join(', ');
  document.getElementById('neigh').textContent = 'k' + k + '  in: [' + names(kIn) + ']  out: [' + names(kOut) + ']';
  const target = document.getElementById('target').value;
  const sched = loadStage('sched'), ir = loadStage('ir'), asm = loadStage('asm', target);
  document.getElementById('sched').innerHTML = hlIR(await sched);
  document.getElementById('ir').innerHTML = hlIR(await ir);
  document.getElementById('asm').textContent = await asm;
}

document.getElementById('target').onchange = () => { if (curKernel !== null) selectKernel(curKernel); };

// Drag the dividers to resize the columns.
for (const divider of document.querySelectorAll('.divider')) {
  divider.addEventListener('mousedown', e => {
    e.preventDefault();
    const section = divider.previousElementSibling;
    const startX = e.clientX, startWidth = section.offsetWidth;
    divider.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    const move = ev => { section.style.width = (startWidth + ev.clientX - startX) + 'px'; };
    const up = () => {
      divider.classList.remove('dragging');
      document.body.style.cursor = '';
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
    };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  });
}
refreshTabs();
setInterval(() => { if (curGraph < 0) refreshTabs(); }, 2000);
</script>
</body>
</html>
"#;
