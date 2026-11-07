// splat2mesh app: the wizard, the job list, components and settings. Talks to splat2mesh/app/server.py.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (path, body) => {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
};
const upload = async (file) => {
  const r = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
  return (await r.json()).path;
};
const fileUrl = (p) => `/api/file?path=${encodeURIComponent(p)}`;
let status = null;

// ------------------------------------------------------------------------------------------- pages
$$("nav button").forEach((b) => b.onclick = () => show(b.dataset.page));
function show(page) {
  if (history.replaceState) history.replaceState(null, "", "#" + page);
  $$("nav button").forEach((b) => b.classList.toggle("on", b.dataset.page === page));
  $$(".page").forEach((p) => p.classList.toggle("on", p.id === page));
  if (page === "jobs") refreshJobs();
  if (page === "components" || page === "settings") refreshStatus();
}

// ------------------------------------------------------------------------------------------- wizard
const STEPS = [["start", "Start"], ["input", "Input"], ["splat3d", "3D"], ["outputs", "Outputs"], ["render", "Render"], ["review", "Summary"]];
let step = 0;
const source = () => $("input[name=source]:checked").value;
const batchKind = () => $("input[name=batchkind]:checked").value;
const visibleSteps = () => STEPS.filter(([id]) => !(id === "splat3d" && source() === "splat"));

function renderSteps() {
  const vs = visibleSteps();
  step = Math.min(step, vs.length - 1);
  $("#steps").innerHTML = vs.map(([id, name], i) => `<li data-i="${i}" class="${i === step ? "on" : i < step ? "done" : ""}">${i + 1}. ${name}</li>`).join("");
  $$("#steps li").forEach((li) => li.onclick = () => { step = +li.dataset.i; renderSteps(); });
  $$(".step").forEach((s) => s.classList.toggle("on", s.dataset.step === vs[step][0]));
  $$("[data-when]").forEach((e) => e.style.display = e.dataset.when === source() ? "" : "none");
  $$("[data-when-batch]").forEach((e) => e.style.display = e.dataset.whenBatch === batchKind() ? "" : "none");
  $$("[data-when-out=mesh]").forEach((e) => e.style.display = $("#out_mesh").checked ? "" : "none");
  $("#back").disabled = step === 0;
  $("#next").style.display = step === vs.length - 1 ? "none" : "";
  $("#run").style.display = step === vs.length - 1 ? "" : "none";
  if (vs[step][0] === "review") summary();
}
$("#next").onclick = () => { step++; renderSteps(); };
$("#back").onclick = () => { step--; renderSteps(); };
$$("input[name=source], input[name=batchkind], #out_mesh").forEach((e) => e.onchange = renderSteps);
$("#seeds").oninput = () => $("#seeds_v").value = $("#seeds").value;
$("#image").onchange = () => { const f = $("#image").files[0]; const im = $("#image_preview");
  if (f) { im.src = URL.createObjectURL(f); im.style.display = "block"; } };

function options() {
  return {
    text2image: { seed: +$("#t2i_seed").value, size: $("#t2i_size").value, mode: $("#t2i_mode").value },
    idol: { seeds: +$("#seeds").value, gaussians: +$("#gaussians").value },
    outputs: { splat: $("#out_splat").checked, mesh: $("#out_mesh").checked, materials: $("#out_materials").checked },
    mesh: { faces: +$("#faces").value, tex: +$("#tex").value },
    parts: $("#parts").value.trim(),
    render: { turntable: $("#tt").checked, frames: +$("#frames").value, size: +$("#size").value,
              technique: $("#technique").value, hdri: $("#hdri").value },
  };
}

function estimate(o, n) {      // minutes on an 8 GB RTX, from measured runs
  let m = 0;
  if (source() === "prompt" || (source() === "batch" && batchKind() === "prompts")) m += 1.5;
  if (source() !== "splat") m += o.idol.seeds * (o.idol.seeds > 1 ? 4.5 : 3);
  if (o.outputs.splat) m += o.outputs.materials ? 30 : 1;
  if (o.outputs.mesh) m += o.outputs.materials ? (o.outputs.splat ? 6 : 33) : 2;
  if (o.render.turntable) m += (o.outputs.splat + o.outputs.mesh) * (o.render.technique === "rt" ? 6 : 1.5);
  return Math.round(m * n);
}

function summary() {
  const o = options(); const s = source();
  const n = s === "batch" ? (batchKind() === "prompts" ? $("#batch_prompts").value.split("\n").filter((x) => x.trim()).length : $("#batch_images").files.length) : 1;
  const outs = [o.outputs.splat && "relightable splat", o.outputs.mesh && "mesh USD"].filter(Boolean).join(" + ") || "none";
  const rows = [
    ["Start", { prompt: "a sentence", image: "a picture", splat: "an Idol splat", batch: `batch of ${n} ${batchKind() === "prompts" ? "sentences" : "pictures"}` }[s]],
    ["Idol", s === "splat" ? "—" : `${o.idol.seeds} seed${o.idol.seeds > 1 ? "s (best kept)" : ""}, ${o.idol.gaussians.toLocaleString()} gaussians`],
    ["Outputs", outs], ["Materials", o.outputs.materials ? (o.parts ? "by hand: " + o.parts : "automatic (DeepSeek + ENEAS)") : "none"],
    ["Turntable", o.render.turntable ? `${o.render.frames} frames, ${o.render.size}px, ${o.render.technique}, ${o.render.hdri || "studio"}` : "none"],
    ["Estimated time", `about ${estimate(o, n)} min`],
  ];
  $("#summary").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  const w = [];
  const c = status?.components || {};
  if (s !== "splat" && !c.idol?.ok) w.push("Idol is not installed (Components).");
  if (o.outputs.materials && !c.eneas?.ok) w.push("ENEAS is not installed (Components).");
  if (o.outputs.materials && !c.deepseek?.ok) w.push("No DeepSeek key (Settings): the materials need it.");
  if (!c.athenea?.ok) w.push("athenea is not installed (Components).");
  if ((s === "prompt" || (s === "batch" && batchKind() === "prompts")) && !c.iris?.ok && o.text2image.mode !== "space" && status?.iris_mode !== "space") w.push("Iris-3B is not installed: choose the online Space or install it.");
  if (!o.outputs.splat && !o.outputs.mesh) w.push("Choose at least one output.");
  $("#warn").innerHTML = w.join("<br>");
}

$("#run").onclick = async () => {
  const o = options(); const s = source(); const name = $("#jobname").value.trim();
  $("#run").disabled = true;
  try {
    if (s === "batch") {
      let items;
      if (batchKind() === "prompts") items = $("#batch_prompts").value.split("\n").map((x) => x.trim()).filter(Boolean)
        .map((p, i) => ({ source: "prompt", prompt: p, name: `${name || "batch"} ${i + 1}` }));
      else items = await Promise.all([...$("#batch_images").files].map(async (f) => ({ source: "image", image: await upload(f), name: f.name.replace(/\.\w+$/, "") })));
      await api("/api/jobs", { kind: "batch", options: o, items });
    } else {
      const p = { ...o, source: s, name: name || "object" };
      if (s === "prompt") p.prompt = $("#prompt").value.trim();
      if (s === "image") { if (!$("#image").files[0]) throw new Error("choose a picture"); p.image = await upload($("#image").files[0]); }
      if (s === "splat") { for (const k of ["splat", "camera", "input"]) { const f = $("#" + k).files[0]; if (!f) throw new Error("missing " + k); p[k] = await upload(f); } }
      await api("/api/jobs", { kind: "create", params: p, title: name || (p.prompt || "").slice(0, 60) || "object" });
    }
    show("jobs");
  } catch (e) { alertBox(e.message); }
  $("#run").disabled = false;
};
function alertBox(msg) { $("#warn").textContent = msg; }

// ------------------------------------------------------------------------------------------- jobs
const open = {};
async function refreshJobs() {
  const js = await api("/api/jobs").catch(() => []);
  const active = js.filter((j) => j.status === "running" || j.status === "queued").length;
  $("#jobcount").textContent = active || "";
  const el = $("#joblist");
  if (!js.length) { el.innerHTML = '<p class="hint">No jobs yet.</p>'; return; }
  for (const j of js) {
    let card = $(`#job-${j.id}`);
    if (!card) {
      card = document.createElement("div"); card.id = `job-${j.id}`; card.className = "job";
      card.innerHTML = `<div class="head"><span class="title"></span><span class="status"></span><span class="when"></span></div>
        <div class="bar"><i></i></div><div class="msg"></div><div class="results"></div><div class="files"></div>
        <p><button class="ghost logbtn">Log</button> <button class="ghost cancel">Cancel</button></p><pre class="log" hidden></pre>`;
      el.appendChild(card);
      $(".logbtn", card).onclick = async () => { open[j.id] = !open[j.id]; await showLog(j.id, card); };
      $(".cancel", card).onclick = () => api(`/api/jobs/${j.id}/cancel`, {}).then(refreshJobs);
    }
    card.className = `job ${j.status}`;
    $(".title", card).textContent = j.title;
    $(".status", card).textContent = j.status; $(".status", card).className = `status ${j.status}`;
    $(".when", card).textContent = new Date(j.created * 1000).toLocaleString();
    $(".bar > i", card).style.width = `${j.progress}%`;
    $(".msg", card).textContent = j.status === "failed" ? (j.error || "failed: see the log").slice(-400) : `${j.progress}% · ${j.stage}${j.message ? ": " + j.message : ""}`;
    $(".cancel", card).style.display = j.status === "running" || j.status === "queued" ? "" : "none";
    if (j.results && !card.dataset.res) { card.dataset.res = 1; results(j, card); }
    if (open[j.id]) showLog(j.id, card);
  }
  // keep the newest first
  js.forEach((j) => el.appendChild($(`#job-${j.id}`)));
}
async function showLog(id, card) {
  const pre = $(".log", card); pre.hidden = !open[id];
  if (!open[id]) return;
  const j = await api(`/api/jobs/${id}?log=30000`);
  pre.textContent = j.log || "(empty)"; pre.scrollTop = pre.scrollHeight;
}
function results(j, card) {
  const f = j.results.files || {};
  const media = [];
  if (f.input) media.push(`<img src="${fileUrl(f.input)}" title="input">`);
  for (const k of ["splat_turntable", "mesh_turntable"]) if (f[k]) media.push(`<video src="${fileUrl(f[k])}" controls loop muted autoplay playsinline title="${k}"></video>`);
  $(".results", card).innerHTML = media.join("");
  const names = { splat_usd: "relightable splat (USD)", mesh_usd: "mesh (USD)", splat_ply: "Idol splat (.ply)", splat_materials: "splat + materials (.ply)" };
  $(".files", card).innerHTML = Object.entries(names).filter(([k]) => f[k]).map(([k, n]) =>
    `<a href="${fileUrl(f[k])}" download>${n}</a>`).join("") + `<button class="ghost openf">Open folder</button>` +
    (j.results.parts ? `<p class="hint">Parts: ${j.results.parts}</p>` : "") + `<p class="hint">${Math.round((j.results.seconds || 0) / 60)} min</p>`;
  $(".openf", card).onclick = () => api("/api/open", { path: j.params.out });
}
setInterval(() => { if ($("#jobs").classList.contains("on") || $("#jobcount").textContent) refreshJobs(); }, 1500);

// ------------------------------------------------------------------------------------------- components & settings
const COMP = {
  athenea: ["athenea", "Renderer: orbit renders for ENEAS, turntables, relit splats, glass (Windows release, ~300 MB)."],
  idol: ["Idol", "Picture → 3D Gaussian splat. PyTorch + weights (~8 GB). DINOv3 is gated: accept its terms on Hugging Face."],
  eneas: ["ENEAS", "Finds and tracks the parts on the views. Its own Python environment; weights ~9 GB on first use."],
  iris: ["Iris-3B Turbo", "Sentence → picture. Local (~14 GB of weights) or the online Space."],
  deepseek: ["DeepSeek", "Online API for part lists and checks: a key in Settings."],
};
async function refreshStatus() {
  status = await api("/api/status").catch(() => null);
  if (!status) return;
  const g = status.components.gpu;
  $("#gpu").textContent = g.name ? `${g.name} · ${g.vram_gb} GB · driver ${g.driver}` : "no NVIDIA GPU found";
  $("#complist").innerHTML = Object.entries(COMP).map(([k, [n, d]]) => {
    const c = status.components[k] || {};
    const btn = k === "deepseek" ? `<button onclick="show('settings')">Settings</button>` : `<button data-install="${k}">${c.ok ? "Reinstall" : "Install"}</button>`;
    return `<div class="card comp"><b>${n}</b><span class="state ${c.ok ? "ok" : "missing"}">${c.ok ? "installed" : "missing"}</span>
      <p class="hint">${d}</p>${btn}</div>`;
  }).join("");
  $$("[data-install]").forEach((b) => b.onclick = () => install([b.dataset.install]));
  $("#iris_mode").value = status.iris_mode || "cpu-encoder";
  $("#hdri").innerHTML = `<option value="">studio (default)</option>` + (status.hdris || []).map((h) => `<option value="${h}">${h}</option>`).join("");
  $("#folders").innerHTML = `<dt>splat2mesh</dt><dd>${status.root}</dd><dt>Jobs and results</dt><dd>${status.work}</dd>`;
  if (g.vram_gb && g.vram_gb < 15) $("#iris_mode_inst").value = "cpu-encoder";
}
async function install(components) {
  await api("/api/jobs", { kind: "install", components, vae: $("#vae").value, iris_mode: $("#iris_mode_inst").value, hf_token: $("#hf_token").value.trim() || undefined });
  show("jobs");
}
$("#install_all").onclick = () => {
  const c = status?.components || {};
  const miss = ["athenea", "idol", "eneas", "iris"].filter((k) => !c[k]?.ok);
  install(miss.length ? miss : ["athenea", "idol", "eneas", "iris"]);
};
$("#save_settings").onclick = async () => {
  await api("/api/settings", { deepseek_key: $("#deepseek_key").value || undefined, iris_mode: $("#iris_mode").value });
  $("#deepseek_key").value = ""; $("#saved").textContent = "saved"; refreshStatus();
};

renderSteps(); refreshStatus().then(() => { if (location.hash) show(location.hash.slice(1)); }); refreshJobs();
