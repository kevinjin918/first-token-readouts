// First-token readouts project page: header state, contents menu, Figure 1, BibTeX copy.

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

  const tip = document.createElement("div");
  tip.className = "tip"; tip.hidden = true; tip.setAttribute("role", "status");
  document.body.appendChild(tip);
  const place = (e) => {
    const pad = 14, r = tip.getBoundingClientRect();
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
  };
  chart.addEventListener("pointerover", (e) => {
    const s = e.target.closest(".s");
    if (!s) return;
    tip.innerHTML = s.dataset.tip; tip.hidden = false; place(e);
  });
  chart.addEventListener("pointermove", (e) => { if (!tip.hidden) place(e); });
  chart.addEventListener("pointerleave", () => { tip.hidden = true; });
  chart.addEventListener("pointerout", (e) => { if (e.target.closest(".s")) tip.hidden = true; });
  window.addEventListener("scroll", () => { tip.hidden = true; }, { passive: true });
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
initCopy();
