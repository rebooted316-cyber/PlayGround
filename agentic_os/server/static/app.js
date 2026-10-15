"use strict";
/* Agentic OS War Room: vanilla JS single-page app talking to the Agentic OS HTTP API. */

const $ = (sel, root = document) => root.querySelector(sel);
const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* storage unavailable */ } },
};
let token = store.get("aos_token", "");
const state = { agents: [], tools: [], providers: [], presets: [], extPresets: [], tab: "chat", timers: {} };

/* ---------------------------------------------------------------- helpers */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c.nodeType ? c : document.createTextNode(String(c)));
  return node;
}
function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg; t.classList.toggle("err", isErr); t.classList.add("show");
  clearTimeout(state.timers.toast); state.timers.toast = setTimeout(() => t.classList.remove("show"), 3200);
}
/* Safe rich text: escape everything, then allow **bold** and `code` only. */
function richText(text) {
  const frag = document.createDocumentFragment();
  const parts = String(text || "").split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const p of parts) {
    if (/^\*\*[^*]+\*\*$/.test(p)) frag.append(el("strong", { text: p.slice(2, -2) }));
    else if (/^`[^`]+`$/.test(p)) frag.append(el("code", { text: p.slice(1, -1) }));
    else if (p) frag.append(document.createTextNode(p));
  }
  return frag;
}
const fmtTime = (ts) => ts ? new Date(ts * 1000).toLocaleTimeString("pl-PL", { hour: "2-digit", minute: "2-digit" }) : "";
const statusTag = (s) => el("span", { class: "tag " + ({ completed: "ok", failed: "err", running: "run", queued: "run", waiting: "warn" }[s] || ""), text: ({ completed: "gotowe", failed: "błąd", running: "w toku", queued: "w kolejce", waiting: "czeka na Ciebie" }[s] || s) });
const PERMISSIONS = { ask: "✋ zawsze pyta", smart: "🧠 sam decyduje, kiedy pytać", auto: "🚀 autonomiczny", full: "🔓 pełny dostęp" };
const PERMISSION_OPTIONS = [["ask", "✋ Zawsze pytaj"], ["smart", "🧠 Sam decyduje, kiedy pytać"], ["auto", "🚀 Autonomiczny"], ["full", "🔓 Pełny dostęp"]];
const MODE_OPTIONS = [["standard", "Standardowy"], ["reflective", "🔁 Refleksyjny"], ["adaptive", "🎯 Adaptacyjny"]];
const options = (list, current) => list.map(([v, t]) => el("option", { value: v, text: t, selected: v === current }));
/* Change an agent's permission or thinking mode at any time (applies immediately, even to a running task). */
async function changeAgent(name, changes, select) {
  const before = state.agents.find((a) => a.name === name) || {};
  if (changes.permission === "full" && before.permission !== "full" &&
      !confirm(`Pełny dostęp: ${name} użyje każdego narzędzia, serwera MCP, moda i agenta bez pytania. Kontynuować?`)) {
    if (select) select.value = before.permission || "auto";
    return;
  }
  try {
    const updated = await api(`/agents/${name}/settings`, { method: "POST", body: changes });
    Object.assign(before, updated);
    toast(`${name}: ${changes.permission ? PERMISSIONS[updated.permission] : MODES[updated.mode] || "standardowy"}`);
    syncChatPermission();
    if (state.tab === "agents") loadAgents();
  } catch (err) {
    toast(err.message, true);
    if (select) select.value = changes.permission ? before.permission : before.mode;
  }
}
function syncChatPermission() {
  const sel = $("#chat-permission");
  const a = state.agents.find((x) => x.name === $("#chat-agent").value);
  const native = a && a.kind !== "external";
  sel.closest("label").classList.toggle("hidden", !native);
  if (native) sel.replaceChildren(...options(PERMISSION_OPTIONS, a.permission));
}
$("#chat-permission").addEventListener("change", (e) => changeAgent($("#chat-agent").value, { permission: e.target.value }, e.target));
const MODES = { standard: "", reflective: "🔁 refleksyjny", adaptive: "🎯 adaptacyjny" };
const STRATEGIES = { zero_shot: "zwięzły", chain_of_thought: "krok po kroku", role: "ekspert" };

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: "Bearer " + token } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  if (res.status === 401) { showLogin(); throw new Error("Wymagane logowanie"); }
  if (!res.ok) throw new Error((data && data.error) || `Błąd ${res.status}`);
  return data;
}
async function waitForProcess(pid, onTick) {
  for (let i = 0; i < 600; i++) {
    const p = await api(`/processes/${pid}`);
    if (onTick) onTick(p);
    if (p.status === "completed" || p.status === "failed") return p;
    await new Promise((r) => setTimeout(r, i < 10 ? 700 : 1500));
  }
  throw new Error("Przekroczono czas oczekiwania");
}
function withBusy(form, fn) {
  return async (e) => {
    e.preventDefault();
    const btn = form.querySelector("button[type=submit]");
    if (btn) btn.disabled = true;
    try { await fn(new FormData(form)); } catch (err) { toast(err.message, true); } finally { if (btn) btn.disabled = false; }
  };
}

/* ------------------------------------------------------------------ login */
function showLogin() { $("#app").classList.add("hidden"); $("#login").classList.remove("hidden"); $("#login-token").focus(); }
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  token = $("#login-token").value.trim();
  try {
    await api("/auth/check");
    store.set("aos_token", token);
    $("#login-error").textContent = "";
    start();
  } catch (err) { $("#login-error").textContent = "Nieprawidłowe hasło."; }
});

/* ------------------------------------------------------------------- tabs */
function showTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  store.set("aos_tab", name);
  ({ agents: loadAgents, tasks: loadTasks, brain: loadBrain, knowledge: loadKnowledge, ext: loadExt, models: loadModels, monitor: loadMonitor }[name] || (() => {}))();
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

/* --------------------------------------------------------------- shared */
async function refreshAgents() {
  const data = await api("/agents");
  state.agents = data.agents; state.tools = data.tools; state.extPresets = data.external_presets;
  const sel = $("#chat-agent"); const current = sel.value || store.get("aos_chat_agent", "assistant");
  sel.replaceChildren(...state.agents.map((a) => el("option", { value: a.name, text: `${a.name}${a.kind === "external" ? " (zewn.)" : ""}` })));
  sel.value = state.agents.some((a) => a.name === current) ? current : "assistant";
  syncChatPermission();
}
async function refreshProviders() {
  const data = await api("/providers");
  state.providers = data.providers; state.presets = data.presets;
  const d = data.default;
  $("#model-badge").textContent = d.model ? `🧠 ${d.model}` : `🧠 ${d.name || "offline"}`;
  return data;
}

/* ------------------------------------------------------------------- chat */
const chatLog = $("#chat-log");
function sessionId(agent) {
  let s = store.get("aos_session_" + agent);
  if (!s) { s = Math.random().toString(36).slice(2, 10); store.set("aos_session_" + agent, s); }
  return s;
}
function addMsg(role, text) {
  $(".empty", chatLog)?.remove();
  const m = el("div", { class: "msg " + role });
  m.append(richText(text));
  chatLog.append(m); chatLog.scrollTop = chatLog.scrollHeight;
  return m;
}
function renderAgentReply(node, proc) {
  node.classList.remove("thinking"); node.replaceChildren();
  if (proc.status === "failed") { node.append(richText("⚠️ " + (proc.error || "Błąd"))); return; }
  const r = proc.result || {};
  node.append(richText(r.output || "(brak odpowiedzi)"));
  if (r.reflection) {
    node.append(el("details", {}, el("summary", { text: `🔁 Samoocena: ${Math.round(r.reflection.best_score * 100)}% (próby: ${r.reflection.attempts.length})` }),
      ...r.reflection.attempts.map((a, i) => el("pre", { text: `Próba ${i + 1}: ${Math.round(a.score * 100)}%${a.critique ? "\n" + a.critique : ""}` }))));
  }
  if (r.tool_calls && r.tool_calls.length) {
    node.append(el("details", {}, el("summary", { text: `🛠 Użyte narzędzia (${r.tool_calls.length})` }),
      ...r.tool_calls.map((c) => el("pre", { text: `${c.tool}(${JSON.stringify(c.arguments)})\n→ ${c.result}` }))));
  }
  if (r.citations && r.citations.length) {
    node.append(el("div", { class: "meta", text: "📚 Źródła: " + r.citations.map((c) => `[${c.n}] ${c.source}`).join(", ") }));
  }
  const usage = r.usage || {};
  const fb = el("span", { class: "fb" });
  for (const [label, reward] of [["👍", 1], ["👎", -1]]) {
    fb.append(el("button", { type: "button", "aria-label": reward > 0 ? "Dobra odpowiedź" : "Zła odpowiedź", text: label, onclick: async (ev) => {
      try { await api(`/processes/${proc.pid}/feedback`, { method: "POST", body: { reward } }); fb.querySelectorAll("button").forEach((b) => b.classList.remove("on")); ev.target.classList.add("on"); toast("Dzięki, agent się uczy."); } catch (err) { toast(err.message, true); }
    } }));
  }
  node.append(el("div", { class: "meta" }, `${r.agent || ""}${r.strategy ? ` · 🎯 styl: ${STRATEGIES[r.strategy] || r.strategy}` : ""} · kroki: ${r.steps || 1}${usage.input_tokens ? ` · tokeny: ${usage.input_tokens + usage.output_tokens}` : ""}`, fb));
}
$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const ta = $("#chat-text"); const text = ta.value.trim(); if (!text) return;
  const agent = $("#chat-agent").value || "assistant";
  ta.value = ""; ta.style.height = "";
  addMsg("user", text);
  const reply = addMsg("agent thinking", "Myślę…");
  try {
    const proc = await api("/chat", { method: "POST", body: { message: text, agent, session: sessionId(agent), async: true } });
    const done = await waitForProcess(proc.pid);
    renderAgentReply(reply, done);
  } catch (err) { reply.classList.remove("thinking"); reply.textContent = "⚠️ " + err.message; }
  chatLog.scrollTop = chatLog.scrollHeight;
});
$("#chat-text").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing && window.innerWidth > 900) { e.preventDefault(); $("#chat-form").requestSubmit(); } });
$("#chat-text").addEventListener("input", (e) => { e.target.style.height = ""; e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px"; });
$("#chat-agent").addEventListener("change", (e) => { store.set("aos_chat_agent", e.target.value); syncChatPermission(); chatEmpty(); });
$("#chat-reset").addEventListener("click", async () => {
  const agent = $("#chat-agent").value;
  try { await api("/chat/reset", { method: "POST", body: { agent, session: sessionId(agent) } }); } catch { /* ignore */ }
  store.del("aos_session_" + agent); chatEmpty(); toast("Nowa rozmowa");
});
function chatEmpty() { chatLog.replaceChildren(el("div", { class: "empty" }, el("div", { text: "👋 Napisz, co mam zrobić." }), el("div", { class: "small", text: "Agent użyje narzędzi, pamięci i Twojej bazy wiedzy." }))); }

/* ----------------------------------------------------------------- agents */
async function loadAgents() {
  let skills = [];
  try { [, , { skills }] = await Promise.all([refreshAgents(), refreshProviders(), api("/skills")]); } catch (err) { toast(err.message, true); return; }
  $("#agent-skills").replaceChildren(...(skills.length ? skills.map((s) => el("label", { class: "chip" }, el("input", { type: "checkbox", name: "skills", value: s.name }), s.name)) : [el("span", { class: "muted small", text: "Dodaj skille w zakładce Rozszerzenia." })]));
  $("#agent-list").replaceChildren(...state.agents.map((a) => el("div", { class: "item" },
    el("div", { class: "item-head" }, el("span", { class: "item-title", text: (a.kind === "external" ? "🔌 " : "🤖 ") + a.name }),
      el("span", { class: "tag", text: a.kind === "external" ? a.protocol : `${a.provider} · ${a.model || "?"}` })),
    el("p", { class: "muted small", text: a.description || "" }),
    a.tools && a.tools.length ? el("p", { class: "small", text: "🛠 " + a.tools.join(", ") }) : null,
    a.skills && a.skills.length ? el("p", { class: "small", text: "🎓 " + a.skills.join(", ") }) : null,
    agentTraits(a),
    a.learning ? el("p", { class: "small", text: "📈 " + Object.entries(a.learning.strategies).map(([k, v]) => `${STRATEGIES[k] || k}: ${v.pulls ? `${v.mean_reward > 0 ? "+" : ""}${v.mean_reward} (${v.pulls}×)` : "–"}`).join(" · ") + (a.learning.best ? ` → najlepszy: ${STRATEGIES[a.learning.best] || a.learning.best}` : "") }) : null,
    el("div", { class: "actions" },
      el("button", { class: "btn small", type: "button", text: "💬 Czat", onclick: () => { $("#chat-agent").value = a.name; store.set("aos_chat_agent", a.name); chatEmpty(); showTab("chat"); } }),
      a.name !== "assistant" ? el("button", { class: "btn small ghost danger", type: "button", text: "Usuń", onclick: async () => {
        if (!confirm(`Usunąć agenta ${a.name}?`)) return;
        try { await api(`/agents/${a.name}`, { method: "DELETE" }); toast("Usunięto"); loadAgents(); } catch (err) { toast(err.message, true); }
      } }) : null))));
  const provSel = $("#agent-provider");
  provSel.replaceChildren(el("option", { value: "default", text: "Domyślny systemu" }), ...state.providers.map((p) => el("option", { value: p.id, text: p.label || p.id })));
  $("#agent-tools").replaceChildren(...state.tools.map((t) => el("label", { class: "chip" }, el("input", { type: "checkbox", name: "tools", value: t, checked: true }), t)));
  $("#agent-delegates").replaceChildren(...state.agents.map((a) => el("label", { class: "chip" }, el("input", { type: "checkbox", name: "delegates", value: a.name }), a.name)));
  $("#crew-agents").replaceChildren(...state.agents.map((a) => el("label", { class: "chip" }, el("input", { type: "checkbox", name: "agents", value: a.name }), a.name)));
  const ext = $("#ext-preset");
  if (!ext.options.length) {
    ext.replaceChildren(el("option", { value: "", text: "— wybierz —" }), ...state.extPresets.map((p, i) => el("option", { value: String(i), text: p.label })));
  }
}
function agentTraits(a) {
  if (a.kind === "external") return null;
  const perm = el("select", { "aria-label": `Uprawnienia ${a.name}`, onchange: (e) => changeAgent(a.name, { permission: e.target.value }, e.target) }, ...options(PERMISSION_OPTIONS, a.permission));
  const mode = el("select", { "aria-label": `Tryb myślenia ${a.name}`, onchange: (e) => changeAgent(a.name, { mode: e.target.value }, e.target) }, ...options(MODE_OPTIONS, a.mode));
  return el("div", {},
    el("div", { class: "agent-settings" },
      el("label", { class: "field" }, el("span", { text: "Uprawnienia" }), perm),
      el("label", { class: "field" }, el("span", { text: "Tryb myślenia" }), mode)),
    a.delegates && a.delegates.length ? el("p", { class: "small", text: "📨 zleca: " + a.delegates.join(", ") }) : null,
    a.always_allow && a.always_allow.length ? el("p", { class: "small" }, `✅ Zawsze dozwolone: ${a.always_allow.join(", ")} `,
      el("button", { class: "btn small ghost", type: "button", text: "Resetuj", onclick: async () => {
        try { await api(`/agents/${a.name}/allowed`, { method: "DELETE" }); toast("Zgody zresetowane"); loadAgents(); } catch (err) { toast(err.message, true); }
      } })) : null);
}
$("#ext-preset").addEventListener("change", (e) => {
  const p = state.extPresets[Number(e.target.value)]; const f = $("#ext-form");
  if (!p) { $("#ext-note").textContent = ""; return; }
  f.elements.name.value = p.name; f.elements.protocol.value = p.protocol; f.elements.url.value = p.url; $("#ext-note").textContent = p.note;
});
$("#agent-form").addEventListener("submit", withBusy($("#agent-form"), async (fd) => {
  const body = { name: fd.get("name"), instructions: fd.get("instructions") || undefined, provider: fd.get("provider"),
    model: fd.get("model") || undefined, tools: fd.getAll("tools"), skills: fd.getAll("skills"), conversational: fd.get("conversational") === "on",
    mode: fd.get("mode"), permission: fd.get("permission") || "auto", delegates: fd.getAll("delegates") };
  if (body.permission === "full" && !confirm("Pełny dostęp: agent użyje każdego narzędzia, serwera MCP, moda i agenta bez pytania. Kontynuować?")) return;
  await api("/agents", { method: "POST", body }); toast("Agent utworzony ✅"); $("#agent-form").reset(); loadAgents();
}));
$("#ext-form").addEventListener("submit", withBusy($("#ext-form"), async (fd) => {
  await api("/agents/external", { method: "POST", body: Object.fromEntries(fd) });
  toast("Agent podłączony ✅"); $("#ext-form").reset(); $("#ext-note").textContent = ""; loadAgents();
}));
$("#crew-form").addEventListener("submit", withBusy($("#crew-form"), async (fd) => {
  const agents = fd.getAll("agents"); if (!agents.length) throw new Error("Wybierz co najmniej jednego agenta");
  const proc = await api("/crew", { method: "POST", body: { task: fd.get("task"), agents, process: fd.get("process") } });
  toast("Zespół pracuje… wynik w Zadaniach"); showTab("tasks"); openTask(proc.pid);
}));

/* ------------------------------------------------------------------ tasks */
async function loadTasks() {
  try {
    const { processes } = await api("/processes");
    const list = processes.slice().reverse();
    $("#task-list").replaceChildren(...(list.length ? list.map((p) => el("div", { class: "item clickable", onclick: () => openTask(p.pid) },
      el("div", { class: "item-head" }, el("span", { class: "item-title", text: p.task.slice(0, 120) }), statusTag(p.status)),
      el("div", { class: "muted small", text: `${p.agent} · ${fmtTime(p.created_at)} · #${p.pid}` }))) : [el("p", { class: "muted", text: "Brak zadań. Napisz coś w czacie albo uruchom zespół." })]));
  } catch (err) { toast(err.message, true); }
  clearTimeout(state.timers.tasks);
  if (state.tab === "tasks") state.timers.tasks = setTimeout(loadTasks, 4000);
}
async function openTask(pid) {
  const box = $("#task-detail"); box.classList.remove("hidden"); box.replaceChildren(el("p", { class: "muted", text: "Wczytuję…" }));
  try {
    const p = await waitForProcess(pid, (cur) => box.replaceChildren(el("p", { class: "muted", text: `Status: ${cur.status}…` })));
    const r = p.result || {};
    box.replaceChildren(
      el("div", { class: "item-head" }, el("h3", { text: p.task.slice(0, 160) }), statusTag(p.status)),
      el("pre", { class: "out", text: p.error || r.output || JSON.stringify(r, null, 2) }),
      r.reflection ? el("details", {}, el("summary", { text: `🔁 Samoocena: ${Math.round(r.reflection.best_score * 100)}% (próby: ${r.reflection.attempts.length})` }), ...r.reflection.attempts.map((a, i) => el("pre", { class: "out", text: `Próba ${i + 1}: ${Math.round(a.score * 100)}%${a.critique ? "\n" + a.critique : ""}` }))) : null,
      r.strategy ? el("p", { class: "muted small", text: `🎯 Styl: ${STRATEGIES[r.strategy] || r.strategy}` }) : null,
      r.transcript ? el("details", {}, el("summary", { text: `Przebieg pracy zespołu (${r.transcript.length})` }), ...r.transcript.map((t) => el("pre", { class: "out", text: `${t.agent}: ${t.output || ""}` }))) : null,
      r.tool_calls && r.tool_calls.length ? el("details", {}, el("summary", { text: "Narzędzia" }), ...r.tool_calls.map((c) => el("pre", { class: "out", text: `${c.tool}(${JSON.stringify(c.arguments)}) → ${c.result}` }))) : null,
    );
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) { box.replaceChildren(el("p", { class: "error", text: err.message })); }
}
$("#tasks-refresh").addEventListener("click", loadTasks);

/* ------------------------------------------------------------------ brain */
const KIND = { document: ["#f5b84b", "Dokument"], chunk: ["#6ea8ff", "Fragment wiedzy"], fact: ["#3ecf8e", "Fakt"], episode: ["#8b7bff", "Epizod (przebieg zadania)"], lesson: ["#ff6b7a", "Lekcja"] };
$("#brain-legend").replaceChildren(...Object.values(KIND).map(([c, l]) => el("span", {}, el("i", { style: `background:${c}` }), l)));
const brain = { nodes: [], edges: [], raf: 0, sel: null };
async function loadBrain() {
  try {
    const g = await api("/graph");
    const canvas = $("#brain"); const w = canvas.clientWidth || 600; const h = canvas.clientHeight || 400;
    const idx = new Map();
    brain.nodes = g.nodes.map((n, i) => { const node = { ...n, x: w / 2 + Math.cos(i) * (40 + i * 3), y: h / 2 + Math.sin(i) * (40 + i * 3), vx: 0, vy: 0 }; idx.set(n.id, node); return node; });
    brain.edges = g.edges.map((e) => ({ ...e, a: idx.get(e.source), b: idx.get(e.target) })).filter((e) => e.a && e.b);
    $("#brain-detail").classList.add("hidden");
    if (!brain.nodes.length) { drawEmptyBrain(); return; }
    cancelAnimationFrame(brain.raf); simulate(0);
  } catch (err) { toast(err.message, true); }
}
function sizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1; const w = canvas.clientWidth; const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx, w, h };
}
function drawEmptyBrain() {
  const { ctx, w, h } = sizeCanvas($("#brain")); ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--muted"); ctx.textAlign = "center"; ctx.font = "14px system-ui";
  ctx.fillText("Mózg jest pusty: dodaj wiedzę albo porozmawiaj z agentem.", w / 2, h / 2);
}
function simulate(iter) {
  const canvas = $("#brain"); const { ctx, w, h } = sizeCanvas(canvas); const N = brain.nodes;
  if (iter < 300) {
    for (let i = 0; i < N.length; i++) for (let j = i + 1; j < N.length; j++) {
      const a = N[i], b = N[j]; let dx = a.x - b.x, dy = a.y - b.y; const d2 = dx * dx + dy * dy + 0.01; const f = 900 / d2;
      const d = Math.sqrt(d2); dx /= d; dy /= d; a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const e of brain.edges) {
      const dx = e.b.x - e.a.x, dy = e.b.y - e.a.y; const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const target = e.kind === "part" ? 40 : 90; const f = (d - target) * 0.02 * (e.weight || 0.5);
      e.a.vx += (dx / d) * f; e.a.vy += (dy / d) * f; e.b.vx -= (dx / d) * f; e.b.vy -= (dy / d) * f;
    }
    for (const n of N) {
      n.vx += (w / 2 - n.x) * 0.004; n.vy += (h / 2 - n.y) * 0.004;
      n.vx *= 0.82; n.vy *= 0.82; n.x = Math.max(12, Math.min(w - 12, n.x + n.vx)); n.y = Math.max(12, Math.min(h - 12, n.y + n.vy));
    }
  }
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = 1;
  for (const e of brain.edges) {
    ctx.strokeStyle = e.kind === "part" ? "rgba(245,184,75,.35)" : `rgba(110,168,255,${0.15 + (e.weight || 0) * 0.5})`;
    ctx.beginPath(); ctx.moveTo(e.a.x, e.a.y); ctx.lineTo(e.b.x, e.b.y); ctx.stroke();
  }
  for (const n of N) {
    const [color] = KIND[n.kind] || ["#9aa5c4"]; const r = n.kind === "document" ? 9 : 6;
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fill();
    if (brain.sel === n) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
    if (n.kind === "document") { ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--text"); ctx.font = "12px system-ui"; ctx.textAlign = "center"; ctx.fillText(n.label, n.x, n.y - 13); }
  }
  if (iter < 300 && state.tab === "brain") brain.raf = requestAnimationFrame(() => simulate(iter + 1));
}
$("#brain").addEventListener("pointerdown", (e) => {
  const rect = e.target.getBoundingClientRect(); const x = e.clientX - rect.left; const y = e.clientY - rect.top;
  let best = null, bestD = 22 * 22;
  for (const n of brain.nodes) { const d = (n.x - x) ** 2 + (n.y - y) ** 2; if (d < bestD) { best = n; bestD = d; } }
  brain.sel = best; simulate(300);
  const box = $("#brain-detail");
  if (!best) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.replaceChildren(el("div", { class: "item-head" }, el("strong", { text: best.label }), el("span", { class: "tag", text: (KIND[best.kind] || [0, best.kind])[1] })), best.text ? el("p", { class: "small", text: best.text }) : null);
});
$("#brain-refresh").addEventListener("click", loadBrain);

/* -------------------------------------------------------------- knowledge */
async function loadKnowledge() {
  try {
    const { documents } = await api("/knowledge");
    $("#kb-list").replaceChildren(...(documents.length ? documents.map((d) => el("div", { class: "item" },
      el("div", { class: "item-head" }, el("span", { class: "item-title", text: "📄 " + d.source }), el("span", { class: "tag", text: `${d.chunks} fragm.` })),
      el("p", { class: "muted small", text: d.preview }))) : [el("p", { class: "muted", text: "Brak dokumentów. Dodaj pierwszy powyżej." })]));
  } catch (err) { toast(err.message, true); }
}
$("#kb-file").addEventListener("change", async (e) => {
  const file = e.target.files[0]; if (!file) return;
  if (file.size > 1_500_000) { toast("Plik jest za duży (max 1,5 MB)", true); return; }
  const E = $("#kb-form").elements; E.text.value = await file.text(); if (!E.source.value) E.source.value = file.name; e.target.value = "";
});
$("#kb-form").addEventListener("submit", withBusy($("#kb-form"), async (fd) => {
  const r = await api("/knowledge", { method: "POST", body: { text: fd.get("text"), metadata: { source: fd.get("source") || "notatka" } } });
  toast(`Dodano ✅ (${r.chunks} fragm.)`); $("#kb-form").reset(); loadKnowledge();
}));

/* ----------------------------------------------------------------- models */
async function loadModels() {
  let data;
  try { data = await refreshProviders(); } catch (err) { toast(err.message, true); return; }
  const d = data.default;
  $("#default-model").replaceChildren(el("strong", { text: "Domyślny model systemu: " }), `${d.model || d.name || "offline"} `, el("span", { class: "tag", text: d.provider === "env" ? "z ustawień serwera" : d.provider }));
  $("#provider-list").replaceChildren(...data.providers.map((p) => el("div", { class: "item" },
    el("div", { class: "item-head" }, el("span", { class: "item-title", text: `🔑 ${p.label || p.id}` }), el("span", { class: "tag", text: p.kind })),
    el("p", { class: "muted small", text: `${p.base_url || "domyślny adres"} · model: ${p.default_model || "—"} · klucz: ${p.has_key ? p.api_key : "brak"}` }),
    el("div", { class: "actions" },
      el("button", { class: "btn small", type: "button", text: "Testuj", onclick: async (ev) => {
        ev.target.disabled = true;
        try { const r = await api(`/providers/${p.id}/test`, { method: "POST", body: {} }); toast(`✅ Działa (${r.model}): ${r.reply}`); } catch (err) { toast("❌ " + err.message, true); } finally { ev.target.disabled = false; }
      } }),
      el("button", { class: "btn small", type: "button", text: "Ustaw domyślny", onclick: async () => {
        try { await api("/settings/default", { method: "POST", body: { provider: p.id, model: p.default_model || undefined } }); toast("Ustawiono ✅"); loadModels(); } catch (err) { toast(err.message, true); }
      } }),
      el("button", { class: "btn small", type: "button", text: "Edytuj", onclick: () => fillProviderForm(p) }),
      el("button", { class: "btn small ghost danger", type: "button", text: "Usuń", onclick: async () => {
        if (!confirm(`Usunąć dostawcę ${p.id}?`)) return;
        try { await api(`/providers/${p.id}`, { method: "DELETE" }); toast("Usunięto"); loadModels(); } catch (err) { toast(err.message, true); }
      } })))));
  const sel = $("#provider-preset");
  if (!sel.options.length) {
    sel.replaceChildren(...data.presets.map((p, i) => el("option", { value: String(i), text: p.label })));
    fillProviderForm(data.presets[0], true);
  }
}
function fillProviderForm(p, isPreset = false) {
  const f = $("#provider-form");
  const E = f.elements;
  E.id.value = p.id; E.kind.value = p.kind; E.base_url.value = p.base_url || ""; E.default_model.value = p.default_model || "";
  E.api_key.value = ""; E.api_key.placeholder = !isPreset && p.has_key ? "zostaw puste, żeby nie zmieniać" : "wklej klucz API";
  f.closest("details").open = true;
}
$("#provider-preset").addEventListener("change", (e) => fillProviderForm(state.presets[Number(e.target.value)], true));
$("#provider-form").addEventListener("submit", withBusy($("#provider-form"), async (fd) => {
  const body = Object.fromEntries(fd); body.make_default = fd.get("make_default") === "on";
  const preset = state.presets[Number($("#provider-preset").value)];
  body.label = preset && preset.id === body.id ? preset.label : body.id;
  await api("/providers", { method: "POST", body }); toast("Zapisano ✅"); $("#provider-form").elements.api_key.value = ""; loadModels();
}));

/* ---------------------------------------------------------------- monitor */
async function loadMonitor() {
  try {
    const [s, m] = await Promise.all([api("/status"), api("/metrics")]);
    const c = m.counters || {};
    const stats = [["Agenci", s.agents.length], ["Zadania gotowe", (s.processes || {}).completed || 0], ["Błędy", (s.processes || {}).failed || 0],
      ["Wywołania narzędzi", c["tool.calls"] || 0], ["Tokeny wejściowe", c["llm.input_tokens"] || 0], ["Tokeny wyjściowe", c["llm.output_tokens"] || 0],
      ["Fragmenty wiedzy", s.knowledge_chunks], ["Wspomnienia", s.memories], ["Oceny", s.feedback]];
    $("#monitor-stats").replaceChildren(...stats.map(([l, v]) => el("div", { class: "stat" }, el("b", { text: String(v) }), el("span", { text: l }))));
    const rows = Object.entries(m.timings || {});
    $("#monitor-timings").replaceChildren(el("h3", { text: "Czasy odpowiedzi" }), rows.length ? el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ...["Operacja", "Liczba", "p50 ms", "p95 ms", "max ms"].map((t) => el("th", { text: t })))),
      el("tbody", {}, ...rows.map(([k, v]) => el("tr", {}, ...[k, v.count, v.p50_ms, v.p95_ms, v.max_ms].map((t) => el("td", { text: String(t) }))))))) : el("p", { class: "muted", text: "Brak pomiarów." }));
  } catch (err) { toast(err.message, true); }
}
$("#monitor-refresh").addEventListener("click", loadMonitor);

/* ------------------------------------------------------------ install/PWA */
let deferredInstall = null;
window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferredInstall = e; $("#install-btn").classList.remove("hidden"); });
$("#install-btn").addEventListener("click", async () => { if (!deferredInstall) return; deferredInstall.prompt(); await deferredInstall.userChoice; deferredInstall = null; $("#install-btn").classList.add("hidden"); });
if ("serviceWorker" in navigator) window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));

/* ------------------------------------------------------- tool approvals */
async function pollApprovals() {
  clearTimeout(state.timers.approvals);
  if (document.visibilityState === "visible" && !$("#app").classList.contains("hidden")) {
    try {
      const { approvals } = await api("/approvals");
      renderApprovals(approvals);
    } catch { /* offline or logged out: try again later */ }
  }
  state.timers.approvals = setTimeout(pollApprovals, 2500);
}
function renderApprovals(list) {
  const box = $("#approvals");
  const key = list.map((a) => a.id).join(",");
  if (key === state.approvalKey) return; // keep a half-typed answer intact between polls
  state.approvalKey = key;
  box.classList.toggle("hidden", !list.length);
  box.replaceChildren(...list.map((a) => a.kind === "question" ? questionCard(a) : el("div", { class: "approval" },
    el("div", { class: "small" }, el("strong", { text: `✋ ${a.agent}` }), ` chce użyć narzędzia `, el("code", { text: a.tool })),
    el("pre", { text: JSON.stringify(a.arguments, null, 2) }),
    el("div", { class: "actions" },
      el("button", { class: "btn primary", type: "button", text: "✅ Zezwól", onclick: () => decide(a.id, { approve: true }) }),
      el("button", { class: "btn", type: "button", text: "Zawsze", title: "Zezwalaj temu agentowi na to narzędzie bez pytania", onclick: () => decide(a.id, { approve: true, remember: true }) }),
      el("button", { class: "btn ghost danger", type: "button", text: "⛔ Odmów", onclick: () => decide(a.id, { approve: false }) })))));
}
function questionCard(a) {
  const input = el("textarea", { placeholder: "Twoja odpowiedź…", "aria-label": "Odpowiedź dla agenta" });
  return el("div", { class: "approval" },
    el("div", { class: "small" }, el("strong", { text: `❓ ${a.agent}` }), " pyta:"),
    el("p", { class: "question", text: a.arguments.question || "" }),
    input,
    el("div", { class: "actions" },
      el("button", { class: "btn primary", type: "button", text: "Odpowiedz", onclick: () => {
        if (!input.value.trim()) { toast("Wpisz odpowiedź albo pomiń", true); return; }
        decide(a.id, { answer: input.value.trim() });
      } }),
      el("button", { class: "btn ghost", type: "button", text: "Pomiń (niech zdecyduje)", onclick: () => decide(a.id, { answer: "" }) })));
}
async function decide(id, body) {
  try {
    await api(`/approvals/${id}`, { method: "POST", body });
    toast(body.answer !== undefined ? (body.answer ? "Odpowiedź wysłana" : "Agent zdecyduje sam") : body.approve ? (body.remember ? "Zezwolono na stałe" : "Zezwolono") : "Odmówiono");
  } catch (err) { toast(err.message, true); }
  state.approvalKey = null;
  pollApprovals();
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") pollApprovals(); });

/* ------------------------------------------------------------------ start */
async function start() {
  $("#login").classList.add("hidden"); $("#app").classList.remove("hidden");
  chatEmpty();
  try { await Promise.all([refreshAgents(), refreshProviders()]); } catch (err) { toast(err.message, true); }
  showTab(store.get("aos_tab", "chat"));
  pollApprovals();
}
(async () => {
  try { await api("/auth/check"); start(); } catch { showLogin(); }
})();

/* ------------------------------------------------------------ extensions */
async function loadExt() {
  try {
    const [{ skills }, { mods }, { servers }] = await Promise.all([api("/skills"), api("/mods"), api("/mcp")]);
    $("#skill-list").replaceChildren(...(skills.length ? skills.map((s) => el("div", { class: "item" },
      el("div", { class: "item-head" }, el("span", { class: "item-title", text: "🎓 " + s.name }), s.tools.length ? el("span", { class: "tag", text: s.tools.join(", ") }) : null),
      el("p", { class: "muted small", text: s.description }),
      el("div", { class: "actions" }, delBtn(`/skills/${s.name}`, `skill ${s.name}`, loadExt)))) : [el("p", { class: "muted small", text: "Brak skilli." })]));
    $("#mod-list").replaceChildren(...(mods.length ? mods.map((m) => el("div", { class: "item" },
      el("div", { class: "item-head" }, el("span", { class: "item-title", text: "🛠 " + m.name }), el("span", { class: "tag", text: `${m.method} · ${m.tool}` })),
      el("p", { class: "muted small", text: `${m.description} · ${m.url}` }),
      el("div", { class: "actions" }, delBtn(`/mods/${m.name}`, `mod ${m.name}`, loadExt)))) : [el("p", { class: "muted small", text: "Brak modów." })]));
    $("#mcp-list").replaceChildren(...(servers.length ? servers.map((s) => el("div", { class: "item" },
      el("div", { class: "item-head" }, el("span", { class: "item-title", text: "🔗 " + s.name }), el("span", { class: "tag " + (s.connected ? "ok" : "err"), text: s.connected ? `połączony · ${s.tools.length} narz.` : "rozłączony" })),
      el("p", { class: "muted small", text: s.error ? "Błąd: " + s.error : (s.tools.join(", ") || (s.url || s.command)) }),
      el("div", { class: "actions" },
        el("button", { class: "btn small", type: "button", text: "Połącz ponownie", onclick: async (ev) => {
          ev.target.disabled = true;
          try { await api(`/mcp/${s.name}/reconnect`, { method: "POST", body: {} }); toast("Połączono ✅"); } catch (err) { toast(err.message, true); } finally { loadExt(); }
        } }),
        delBtn(`/mcp/${s.name}`, `serwer ${s.name}`, loadExt)))) : [el("p", { class: "muted small", text: "Brak serwerów MCP." })]));
  } catch (err) { toast(err.message, true); }
}
function delBtn(path, label, after) {
  return el("button", { class: "btn small ghost danger", type: "button", text: "Usuń", onclick: async () => {
    if (!confirm(`Usunąć ${label}?`)) return;
    try { await api(path, { method: "DELETE" }); toast("Usunięto"); after(); } catch (err) { toast(err.message, true); }
  } });
}
$("#skill-form").addEventListener("submit", withBusy($("#skill-form"), async (fd) => {
  const md = (fd.get("markdown") || "").trim();
  const body = md ? { markdown: md, name: fd.get("name") || undefined } : { name: fd.get("name"), description: fd.get("description"), instructions: fd.get("instructions"), tools: fd.get("tools") };
  await api("/skills", { method: "POST", body }); toast("Skill zapisany ✅"); $("#skill-form").reset(); loadExt();
}));
$("#mod-form").addEventListener("submit", withBusy($("#mod-form"), async (fd) => {
  const params = String(fd.get("params") || "").split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const [name, type = "string", ...desc] = l.split(":"); return { name: name.trim(), type: type.trim() || "string", description: desc.join(":").trim() };
  });
  const headers = fd.get("header_name") ? { [fd.get("header_name")]: fd.get("header_value") || "" } : {};
  await api("/mods", { method: "POST", body: { name: fd.get("name"), method: fd.get("method"), description: fd.get("description"), url: fd.get("url"), params, headers } });
  toast("Mod zapisany ✅"); $("#mod-form").reset(); loadExt();
}));
$("#mcp-form").addEventListener("submit", withBusy($("#mcp-form"), async (fd) => {
  const body = Object.fromEntries(fd); if (!body.token) delete body.token;
  toast("Łączę z serwerem MCP…");
  const s = await api("/mcp", { method: "POST", body }); toast(`Połączono ✅ (${s.tools.length} narzędzi)`); $("#mcp-form").reset(); loadExt();
}));

/* --------------------------------------------------------- memory import */
$("#mem-import").addEventListener("click", async (ev) => {
  const file = $("#mem-file").files[0]; const text = $("#mem-text").value.trim(); const distill = $("#mem-distill").checked;
  if (!file && !text) { toast("Wybierz plik albo wklej wspomnienia", true); return; }
  if (file && file.size > 60 * 1024 * 1024) { toast("Plik jest za duży (max 60 MB)", true); return; }
  ev.target.disabled = true; $("#mem-result").textContent = "Importuję…" + (distill ? " (model analizuje rozmowy, to może potrwać)" : "");
  try {
    let r;
    if (file) {
      const res = await fetch(`/memory/import?filename=${encodeURIComponent(file.name)}&distill=${distill ? 1 : 0}`, {
        method: "POST", headers: { "Content-Type": "application/octet-stream", ...(token ? { Authorization: "Bearer " + token } : {}) }, body: file });
      r = await res.json(); if (!res.ok) throw new Error(r.error || `Błąd ${res.status}`);
    } else {
      r = await api("/memory/import", { method: "POST", body: { text, distill } });
    }
    $("#mem-result").textContent = `✅ Źródło: ${r.source}. Rozmowy: ${r.conversations_found}, dodane epizody: ${r.episodes_added}, fakty: ${r.facts_added}` + (r.distilled_facts ? `, wyciągnięte fakty: ${r.distilled_facts}` : "") + `. Wspomnień razem: ${r.memories_total}.`;
    $("#mem-file").value = ""; $("#mem-text").value = ""; loadBrain();
  } catch (err) { $("#mem-result").textContent = "❌ " + err.message; } finally { ev.target.disabled = false; }
});
$("#mem-file").addEventListener("change", (e) => { const f = e.target.files[0]; $("#mem-result").textContent = f ? `Wybrano: ${f.name} (${(f.size / 1024 / 1024).toFixed(1)} MB)` : ""; });
