// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

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
  #network { position: absolute; top: 29px; left: 0; right: 0; bottom: 0; }
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
  <section id="plan_section"><h2>ExecPlan</h2><div id="network"></div></section>
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
<script src="/vis-network.min.js"></script>
<script>
let curGraph = -1, graphData = null, net = null, curKernel = null;

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

function draw() {
  const nodes = new vis.DataSet(graphData.nodes.map(n => ({
    id: n.id,
    label: n.label.replace(/\n/g, '\n'),
    shape: n.kernel >= 0 ? 'box' : 'ellipse',
    color: n.kernel >= 0 ? { background: '#5a3040', border: '#c76a8a' } : { background: '#24404a', border: '#4f93a8' },
    font: { color: '#e6e9ee', face: 'monospace', size: 13 },
    margin: 8,
  })));
  const edges = new vis.DataSet(graphData.edges.map(([f, t, l]) => ({ from: f, to: t, label: l === 'store' ? '' : '', arrows: 'to', color: { color: '#56606e' } })));
  if (net) net.destroy();
  net = new vis.Network(document.getElementById('network'), { nodes, edges }, {
    layout: { hierarchical: { direction: 'UD', sortMethod: 'directed', levelSeparation: 220, nodeSpacing: 200 } },
    physics: { enabled: true, solver: 'hierarchicalRepulsion' },
    edges: { arrows: 'to' },
    autoResize: true,
  });
  net.once('stabilizationIterationsDone', () => net.fit());
  net.on('click', params => {
    if (!params.nodes.length) return;
    const n = graphData.nodes.find(x => x.id === params.nodes[0]);
    if (n && n.kernel >= 0) selectKernel(n.kernel);
  });
  window.net = net;
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
  document.getElementById('device').textContent = graphData.devices[k] || '';
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
