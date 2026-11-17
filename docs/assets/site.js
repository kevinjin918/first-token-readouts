// First-token readouts project page: header state, contents menu, Figures 1 and 2, BibTeX copy.

// Figure 1 data: means over the 40 cardiomegaly false-positive films, 50 sampled answers each,
// computed from results/clt_scale/readout_sampled_answers.json with the functions of
// paper/figures/make_fig_split.py (first_token, written). fy/fn: first-token probability of
// "yes"/"no"; norm: mean per-film P_norm. Written answers: yd/nd answer directly, yr/nr after
// reasoning, ne gives neither answer.
const ARMS = [
  { key: "base", label: "No intervention", fy: 0.8237, fn: 0.0025, norm: 0.976, yd: 0.818, yr: 0.1305, nr: 0.028, nd: 0.0025, ne: 0.021 },
  { key: "scale-1", label: "Suppress, scale −1", fy: 0.5554, fn: 0.0052, norm: 0.933, yd: 0.5565, yr: 0.173, nr: 0.184, nd: 0.003, ne: 0.0835 },
  { key: "ablate", label: "Suppress, ablate", fy: 0.6044, fn: 0.0024, norm: 0.952, yd: 0.6105, yr: 0.127, nr: 0.2275, nd: 0.0035, ne: 0.0315 },
  { key: "scale-2", label: "Suppress, scale −2", fy: 0.5232, fn: 0.0067, norm: 0.916, yd: 0.527, yr: 0.123, nr: 0.1935, nd: 0.0035, ne: 0.153 },
  { key: "black", label: "Evidence block, black", fy: 0.1983, fn: 0.1588, norm: 0.261, yd: 0.202, yr: 0.09, nr: 0.4345, nd: 0.1605, ne: 0.113 },
  { key: "mean", label: "Evidence block, mean fill", fy: 0.2962, fn: 0.1104, norm: 0.469, yd: 0.289, yr: 0.1535, nr: 0.349, nd: 0.109, ne: 0.0995 },
  { key: "blur", label: "Evidence block, blur", fy: 0.6767, fn: 0.0199, norm: 0.88, yd: 0.684, yr: 0.1745, nr: 0.0835, nd: 0.014, ne: 0.044 },
  { key: "border", label: "Border block, black", fy: 0.8221, fn: 0.0053, norm: 0.964, yd: 0.835, yr: 0.1085, nr: 0.0355, nd: 0.004, ne: 0.017 },
];
const PAPER_THREE = ["base", "scale-1", "black"];

// Figure 2 data: per film (40 false-positive films, groups fp_banked + fp_fresh of the same file) and
// intervention, "s,P_norm,greedy answer" (y yes, n no, x neither); s is the share of the 50 samples that
// say yes, as in paper/figures/make_fig_films.py.
const FILMS = {
  "clamp_scale-1": "0.82,1,y 1,1,y 0.74,1,y 0.88,1,y 0.54,0.998,n 0.86,1,y 0.4,0.963,n 0.8,1,y 0.96,1,y 0.86,1,y 0.56,0.982,n 0.26,0.999,n 0.92,1,y 0.96,1,y 0.94,1,y 0.68,1,y 0.54,0.992,y 1,1,y 0.82,1,y 1,1,y 0.1,0.148,n 0.98,1,y 0.44,0.852,n 0.72,0.998,y 0.96,1,y 0.46,0.989,y 1,1,y 0.94,1,y 0.84,1,y 0.38,0.982,n 0.74,0.999,y 0.94,1,y 0.8,0.999,x 0.48,0.731,n 0.2,0.095,n 0.74,1,y 0.6,1,n 0.32,0.593,n 1,1,y 1,1,y",
  "clamp_ablate+0": "0.82,1,y 1,1,y 0.96,1,y 0.82,1,y 0.76,0.999,y 1,1,y 0.44,0.984,n 0.96,1,y 1,1,y 0.94,1,y 0.36,0.994,n 0.54,1,n 0.96,1,y 0.98,1,y 0.98,1,y 0.66,1,y 0.4,0.998,n 1,1,y 1,1,y 1,1,y 0.18,0.531,n 0.98,1,y 0.18,0.777,n 0.68,0.999,y 1,1,y 0.32,0.996,n 1,1,y 1,1,y 0.9,1,y 0.58,0.995,y 0.82,1,y 0.96,1,y 0.6,0.999,n 0.06,0.835,n 0.14,0.095,n 0.66,1,y 0.78,1,n 0.08,0.881,n 1,1,y 1,1,y",
  "clamp_scale-2": "0.86,1,y 1,1,y 0.66,1,y 0.82,1,y 0.4,0.997,n 0.78,1,y 0.38,0.947,n 0.62,1,y 0.98,1,y 0.72,1,y 0.28,0.967,n 0.4,0.998,x 0.96,1,y 0.94,1,y 0.9,1,y 0.8,1,y 0.24,0.984,y 0.96,1,y 0.74,1,y 0.96,1,y 0.16,0.042,x 0.9,1,y 0.2,0.706,n 0.6,0.998,n 0.94,1,y 0.26,0.986,y 0.98,1,y 0.98,1,y 0.82,1,y 0.28,0.977,y 0.42,0.999,x 0.96,1,y 0.44,0.998,n 0.24,0.531,n 0.12,0.06,x 0.62,1,y 0.52,1,n 0.2,0.438,n 0.98,1,y 0.98,1,y",
  "black": "0.08,0.003,n 1,1,y 0.02,0,n 0.06,0.004,n 0.02,0,n 0.48,0.378,n 0.08,0.005,n 0.92,0.998,y 0.22,0.033,n 0.96,0.998,y 0.18,0.026,n 0.08,0.02,n 0,0.001,n 1,1,y 0.24,0.076,n 0.02,0.001,x 0.32,0.182,n 1,1,y 0.08,0.001,n 0.04,0.033,n 0.04,0.004,n 0.08,0.009,n 0.02,0.001,n 0,0.002,n 0.92,0.999,y 0.14,0.076,n 1,1,y 0.94,1,y 0.04,0,x 0.06,0.001,n 0.02,0.002,n 0.02,0.011,n 0.04,0.006,n 0,0,x 0.1,0.003,n 0.18,0.438,n 0.18,0.02,n 0,0.001,n 0.12,0.095,n 0.98,1,y",
  "mean": "0,0.023,n 1,1,y 0.02,0,n 0.06,0.004,n 0,0.001,n 0.92,0.997,y 0.2,0.068,n 0.98,1,y 0.48,0.294,n 0.86,0.998,y 0.64,0.755,y 0.34,0.349,n 0,0.001,n 1,1,y 0.82,0.992,y 0,0.005,n 0.8,0.994,y 1,1,y 0.08,0.003,n 0.5,0.977,n 0.02,0.005,n 0.24,0.053,n 0,0.001,n 0.06,0.02,n 1,1,y 0.72,0.971,y 1,1,y 1,1,y 0.06,0.001,x 0.06,0.003,n 0.02,0.002,x 0.16,0.562,n 0.52,0.622,y 0,0,x 0.04,0.004,n 0.58,0.893,n 0.56,0.165,n 0,0.001,n 0.96,1,y 1,1,y",
  "blur": "0.98,1,y 1,1,y 0.98,0.996,y 0.96,0.998,y 0.98,0.999,y 0.9,0.905,y 0.48,0.469,n 1,1,y 0.98,1,y 0.98,1,y 1,1,y 0.9,0.995,y 1,1,y 1,1,y 1,1,y 0.08,0.068,n 0.84,0.988,y 1,1,y 1,1,y 1,1,y 0.88,0.994,y 1,1,y 0.64,0.755,y 0.8,0.971,y 1,1,y 0.98,1,y 1,1,y 0.98,1,y 0.04,0.001,n 0.98,0.999,y 0.64,0.835,y 1,1,y 0.96,1,y 0.64,0.269,y 0.1,0.004,n 1,1,y 0.94,1,y 0.7,0.967,y 1,1,y 1,1,y",
};
const PANELS = [
  { title: "(a) Suppress the top 80 features", noun: "suppression",
    arms: [["clamp_scale-1", "scale −1"], ["clamp_ablate+0", "ablate"], ["clamp_scale-2", "scale −2"]] },
  { title: "(b) Occlude the evidence block", noun: "occlusion",
    arms: [["black", "black"], ["mean", "mean fill"], ["blur", "blur"]] },
];
const ANS = { y: ["yes", "m-yes"], n: ["no", "m-no"], x: ["neither", "m-none"] };

const pct = (v) => (v * 100).toFixed(v < 0.01 ? 2 : 1) + "%";
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function firstTokenSegs(a) {
  return [
    ["c-yes", "“yes”", a.fy],
    ["c-mid", "any other token (mostly “Based”)", 1 - a.fy - a.fn],
    ["c-no", "“no”", a.fn],
  ];
}
function writtenSegs(a) {
  return [
    ["c-yes", "answers yes directly", a.yd],
    ["c-yes2", "says yes after reasoning", a.yr],
    ["c-mid", "gives neither answer", a.ne],
    ["c-no2", "says no after reasoning", a.nr],
    ["c-no", "answers no directly", a.nd],
  ];
}

function bar(segs, armLabel, rowLabel) {
  const parts = segs
    .filter(([, , v]) => v > 0.0004)
    .map(([cls, name, v]) => {
      const tip = `<b>${esc(armLabel)}</b><br>${rowLabel}: ${name}<br><span class="tv">${pct(v)}</span>`;
      return `<span class="s ${cls}" style="flex-grow:${v}" data-tip="${esc(tip)}"></span>`;
    })
    .join("");
  const aria = `${armLabel}, ${rowLabel}: ` + segs.map(([, n, v]) => `${n} ${pct(v)}`).join(", ");
  return `<div class="bar" role="img" aria-label="${esc(aria)}">${parts}</div>`;
}

function renderChart(el, arms) {
  const rows = arms.map((a) => {
    const s = a.yd + a.yr;
    return `<div class="arm">
      <div class="arm-label">${a.label}</div>
      <div class="arm-rows">
        <div class="brow"><span class="rlab">first token</span>${bar(firstTokenSegs(a), a.label, "First token")}
          <span class="val"><span class="m">P<sub>norm</sub></span>${a.norm.toFixed(3)}</span></div>
        <div class="brow"><span class="rlab">written</span>${bar(writtenSegs(a), a.label, "Written answers")}
          <span class="val"><span class="m">s</span>${s.toFixed(2)}</span></div>
      </div></div>`;
  });
  el.innerHTML = rows.join("") +
    `<div class="axis" aria-hidden="true"><span></span><div class="axis-in"><span></span>
      <div class="ticks"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></div><span></span></div></div>`;
}

function renderTable(el, arms) {
  const f2 = (v) => v.toFixed(2);
  const body = arms.map((a) => `<tr>
      <td>${a.label}</td><td>${f2(a.fy)}</td><td>${a.fn.toFixed(3)}</td><td>${f2(1 - a.fy - a.fn)}</td><td class="hl">${a.norm.toFixed(3)}</td>
      <td>${f2(a.yd)}</td><td>${f2(a.yr)}</td><td>${f2(a.ne)}</td><td>${f2(a.nr)}</td><td>${f2(a.nd)}</td><td class="hl">${f2(a.yd + a.yr)}</td></tr>`).join("");
  el.innerHTML = `<div class="table-wrap"><table class="data">
    <thead><tr class="grp"><th></th><th colspan="4">first token</th><th colspan="6">written answers (share of samples)</th></tr>
    <tr><th>Intervention</th><th>yes</th><th>no</th><th>other</th><th><span class="m">P<sub>norm</sub></span></th>
      <th>yes</th><th>yes, reasoned</th><th>neither</th><th>no, reasoned</th><th>no</th><th><span class="m">s</span></th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

function initFigure1() {
  const root = document.getElementById("f1");
  if (!root) return;
  const chart = root.querySelector(".f1-chart");
  const table = root.querySelector(".f1-table");
  const legend = root.querySelector(".legend");
  let all = false, asTable = false;
  const draw = () => {
    const arms = all ? ARMS : ARMS.filter((a) => PAPER_THREE.includes(a.key));
    const stale = document.querySelector(".tip"); if (stale) stale.hidden = true;
    chart.hidden = asTable; legend.hidden = asTable; table.hidden = !asTable;
    asTable ? renderTable(table, arms) : renderChart(chart, arms);
  };
  root.querySelectorAll("[data-set]").forEach((b) => b.addEventListener("click", () => {
    all = b.dataset.set === "all";
    root.querySelectorAll("[data-set]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    draw();
  }));
  root.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
    asTable = b.dataset.view === "table";
    root.querySelectorAll("[data-view]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    draw();
  }));
  draw();
  attachTips(chart);
}

// One tooltip for the page: any element with data-tip inside an attached root shows it on hover.
let TIP = null;
function attachTips(root) {
  if (!TIP) {
    TIP = document.createElement("div");
    TIP.className = "tip"; TIP.hidden = true; TIP.setAttribute("role", "status");
    document.body.appendChild(TIP);
    window.addEventListener("scroll", () => { TIP.hidden = true; }, { passive: true });
  }
  const place = (e) => {
    const pad = 14, r = TIP.getBoundingClientRect();
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
    TIP.style.left = Math.max(8, x) + "px"; TIP.style.top = Math.max(8, y) + "px";
  };
  root.addEventListener("pointerover", (e) => {
    const t = e.target.closest("[data-tip]");
    if (!t || !root.contains(t)) return;
    TIP.innerHTML = t.dataset.tip; TIP.hidden = false; place(e);
  });
  root.addEventListener("pointermove", (e) => { if (!TIP.hidden) place(e); });
  root.addEventListener("pointerout", (e) => { if (e.target.closest("[data-tip]")) TIP.hidden = true; });
  root.addEventListener("pointerleave", () => { TIP.hidden = true; });
}

function marker(ans, x, y) {
  if (ans === "y") return `<circle class="pt m-yes" cx="${x}" cy="${y}" r="4.5"/>`;
  if (ans === "n") return `<rect class="pt m-no" x="${x - 4.2}" y="${y - 4.2}" width="8.4" height="8.4" rx="1"/>`;
  return `<path class="pt m-none" d="M${x} ${y - 5}L${x + 5} ${y + 4}H${x - 5}Z"/>`;
}

function renderPanel(el, panel) {
  const pts = [];
  panel.arms.forEach(([key, name]) => FILMS[key].split(" ").forEach((t, i) => {
    const [s, n, a] = t.split(",");
    pts.push({ s: +s, n: +n, a, film: i + 1, arm: name });
  }));
  const no = pts.filter((p) => p.a === "n"), miss = no.filter((p) => p.n >= 0.5).length;
  const W = Math.max(260, el.clientWidth), H = Math.round(Math.min(300, Math.max(230, W * 0.68)));
  const L = 56, R = 8, T = 8, B = 44, pw = W - L - R, ph = H - T - B;
  const X = (v) => +(L + v * pw).toFixed(1), Y = (v) => +(T + (1 - v) * ph).toFixed(1);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  let g = "";
  ticks.forEach((t) => {
    g += `<text x="${X(t)}" y="${T + ph + 18}" text-anchor="middle">${t}</text>`;
    g += `<text x="${L - 8}" y="${Y(t) + 4}" text-anchor="end">${t}</text>`;
  });
  g += `<line class="ref" x1="${L}" x2="${L + pw}" y1="${Y(0.5)}" y2="${Y(0.5)}"/>`;
  g += `<path class="ax" d="M${L} ${T}V${T + ph}H${L + pw}"/>`;
  g += `<text class="lab" x="${L + pw / 2}" y="${H - 6}" text-anchor="middle">sampled yes rate <tspan class="it">s</tspan></text>`;
  g += `<text class="lab" transform="translate(12 ${T + ph / 2}) rotate(-90)" text-anchor="middle"><tspan class="it">P</tspan><tspan baseline-shift="sub" font-size="10">norm</tspan></text>`;
  ["y", "x", "n"].forEach((a) => pts.filter((p) => p.a === a).forEach((p) => {
    const x = X(p.s), y = Y(p.n);
    const tip = `<b>Film ${p.film}</b> · ${esc(p.arm)}<br><span class="tv"><i>P</i><sub>norm</sub> ${p.n.toFixed(3)} · <i>s</i> ${p.s.toFixed(2)}</span><br>greedy answer: ${ANS[a][0]}`;
    g += `<g data-tip="${esc(tip)}">${marker(a, x, y)}<circle class="hit" cx="${x}" cy="${y}" r="9"/></g>`;
  }));
  const aria = `${panel.title}: ${pts.length} points; the greedy answer is no on ${no.length}, and P_norm is at least 0.5 on ${miss} of them.`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(aria)}">${g}</svg>`;
  return { no: no.length, miss };
}

function initFigure2() {
  const root = document.querySelector("#f2 .films");
  if (!root) return;
  root.innerHTML = PANELS.map((p, i) => `<div class="panel"><h4>${p.title}</h4>
    <p class="note" data-i="${i}"></p><div class="plot" data-i="${i}"></div></div>`).join("");
  const draw = () => PANELS.forEach((p, i) => {
    const c = renderPanel(root.querySelector(`.plot[data-i="${i}"]`), p);
    root.querySelector(`.note[data-i="${i}"]`).innerHTML =
      `<span class="m">P<sub>norm</sub></span> still says yes on <b>${c.miss} of ${c.no}</b> “no” answers`;
  });
  draw();
  let w = root.clientWidth;
  new ResizeObserver(() => { if (root.clientWidth !== w) { w = root.clientWidth; draw(); } }).observe(root);
  attachTips(root);
}

function initHeader() {
  const header = document.querySelector(".site-header");
  const onScroll = () => header.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();

  const btn = document.querySelector(".toc-btn"), toc = document.getElementById("toc");
  const close = () => { toc.hidden = true; btn.setAttribute("aria-expanded", "false"); };
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = toc.hidden;
    toc.hidden = !open; btn.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", (e) => { if (!toc.contains(e.target)) close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  toc.querySelectorAll("a").forEach((a) => a.addEventListener("click", close));

  const links = [...toc.querySelectorAll("a")];
  const secs = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const label = btn.querySelector(".toc-label");
  const mark = () => {
    let cur = null;
    for (const s of secs) if (s.getBoundingClientRect().top < 170) cur = s;
    links.forEach((a) => a.classList.toggle("on", cur && a.getAttribute("href") === "#" + cur.id));
    const on = links.find((a) => a.classList.contains("on"));
    label.textContent = on ? on.textContent : "Contents";
  };
  window.addEventListener("scroll", mark, { passive: true }); mark();
}

function initCopy() {
  document.querySelectorAll(".copy").forEach((b) => b.addEventListener("click", async () => {
    const text = document.getElementById(b.dataset.target).innerText;
    try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; }
    catch { b.textContent = "Select and copy"; }
    setTimeout(() => { b.textContent = "Copy"; }, 1600);
  }));
}

initHeader();
initFigure1();
initFigure2();
initCopy();
