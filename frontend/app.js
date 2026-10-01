/* Citadel console. Two modes:
 *  - live:   served by `python -m citadel serve`; reads and writes through the API.
 *  - replay: opened from disk or with the API down; reads frontend/replay/replay.js, which
 *            scripts/build_replay_bundle.py RECORDED from the real API (never hand-written).
 * Rules: no number is typed into this file; every metric comes from a pipeline CSV; a missing value
 * renders as an em dash, never 0; every panel is error-boundaried; no external network requests. */
(function () {
  "use strict";
  const R = window.CITADEL_REPLAY || null;
  const TOKEN = "demo-analyst-token";
  const S = { mode: "replay", results: null, grammar: null, health: null, lang: "en", withCitadel: true,
              scen: 0, phoneStep: 0, timer: null, alerts: [], selAlert: null };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const num = (v) => (v === null || v === undefined || v === "" ? NaN : Number(v));
  function fmt(v, d = 3) { const x = num(v); return isFinite(x) ? x.toFixed(d) : "—"; }
  function pct(v, d = 1) { const x = num(v); return isFinite(x) ? (100 * x).toFixed(d) + "%" : "—"; }
  function inr(v) { const x = num(v); return isFinite(x) ? "₹" + Math.round(x).toLocaleString("en-IN") : "—"; }
  function act(a) { return `<span class="act ${esc(a)}">${esc(a)}</span>`; }
  function panel(el, fn) {
    try { fn(); } catch (e) { el.innerHTML = `<div class="notice err">This panel failed to render: ${esc(e.message)}</div>`; console.error(e); }
  }
  async function api(path, opts = {}) {
    const r = await fetch(path, { ...opts, headers: { "Content-Type": "application/json", Authorization: "Bearer " + TOKEN, ...(opts.headers || {}) } });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || r.statusText);
    return body;
  }
  const file = (name) => (S.results && S.results.files && S.results.files[name]) || null;
  const byKey = (rows, k, v) => (rows || []).find((r) => String(r[k]) === String(v)) || null;

  // ---------------------------------------------------------------- boot
  async function boot() {
    if (location.protocol.startsWith("http")) {
      try { const h = await fetch("/health"); if (h.ok) { S.health = await h.json(); S.mode = "live"; } } catch (e) { /* replay */ }
    }
    if (S.mode === "live") {
      try { S.results = await api("/v1/results"); S.grammar = await api("/v1/grammar"); } catch (e) { S.mode = "replay"; }
    }
    if (S.mode === "replay" && R) { S.results = R.results; S.grammar = R.grammar; S.health = R.health; }
    $("modepill").textContent = S.mode === "live" ? "LIVE · API" : R ? "REPLAY · offline bundle" : "NO DATA";
    $("runpill").textContent = S.health ? "run " + S.health.run_id : "";
    $("foot").textContent = R ? "replay recorded from run " + (R.meta && R.meta.run_id) : "";
    tabs(); theme();
    panel($("custside"), initCustomer);
    panel($("qlist"), initAnalyst);
    panel($("sout"), initStudio);
    panel($("kpis"), renderResults);
    panel($("limits"), renderSafeguards);
    deepLink();
  }

  // #<tab>[&approve|&run|&alert] opens a screen directly (handy for demos and screenshots)
  function deepLink() {
    const parts = location.hash.replace("#", "").split("&").filter(Boolean);
    if (!parts.length) return;
    const btn = document.querySelector(`nav.tabs button[data-tab="${parts[0]}"]`);
    if (btn) btn.click();
    setTimeout(() => {
      if (parts.includes("approve") && $("ph-approve")) $("ph-approve").click();
      if (parts.includes("run")) runStudio();
      if (parts.includes("alert")) { const q = document.querySelector(".qitem"); if (q) q.click(); }
    }, 50);
  }

  function tabs() {
    document.querySelectorAll("nav.tabs button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll("nav.tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      document.querySelectorAll("section.screen").forEach((s) => s.classList.toggle("active", s.id === b.dataset.tab));
    }));
  }
  function theme() {
    $("themebtn").addEventListener("click", () => {
      const root = document.documentElement;
      const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("citadel-theme", root.dataset.theme); } catch (e) { /* private mode */ }
    });
    try { const t = localStorage.getItem("citadel-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
  }

  // ---------------------------------------------------------------- customer view
  function scenarios() { return (R && R.customer_scenarios) || []; }
  function initCustomer() {
    const sel = $("scen");
    const sc = scenarios();
    if (!sc.length) { $("custside").innerHTML = '<div class="notice">No recorded scenarios. Run <code>make bundle</code>.</div>'; return; }
    sel.innerHTML = sc.map((s, i) => `<option value="${i}">${esc(s.title)}</option>`).join("");
    sel.addEventListener("change", () => { S.scen = +sel.value; resetPhone(); });
    const seg = (a, b, key, val) => { $(a).addEventListener("click", () => { S[key] = val; $(a).setAttribute("aria-pressed", "true"); $(b).setAttribute("aria-pressed", "false"); resetPhone(); }); };
    seg("withBtn", "withoutBtn", "withCitadel", true); seg("withoutBtn", "withBtn", "withCitadel", false);
    seg("enBtn", "hiBtn", "lang", "en"); seg("hiBtn", "enBtn", "lang", "hi");
    $("resetBtn").addEventListener("click", resetPhone);
    resetPhone();
  }
  function resetPhone() { clearInterval(S.timer); S.phoneStep = 0; renderPhone(); renderSide(); }
  const L = (en, hi) => (S.lang === "hi" ? hi : en);

  function renderPhone() {
    const s = scenarios()[S.scen]; if (!s) return;
    const p = s.payment, d = s.decision, el = $("phone");
    const head = `<div class="status"><span>UPI · ${esc(p.app_label || "PayApp")}</span><span>${esc(p.time_ist || "")}</span></div>`;
    const req = {
      collect: [L("Payment request", "भुगतान अनुरोध"), L("wants you to approve a request", "आपसे अनुरोध मंज़ूर करवाना चाहता है")],
      qr_pay: [L("Scan & pay", "स्कैन करें और भुगतान करें"), L("QR code scanned", "QR कोड स्कैन किया गया")],
      pay: [L("Send money", "पैसे भेजें"), L("You are paying", "आप भुगतान कर रहे हैं")],
    }[p.txn_type] || ["Payment", ""];
    if (S.phoneStep === 0) {
      el.innerHTML = head + `<div style="font-weight:700">${esc(req[0])}</div>
        <div class="who">${esc(p.payee_label)}<br><span style="font-size:14px;color:#555">${esc(req[1])}</span></div>
        <div class="amt">${inr(p.amount_inr)}</div>
        ${p.first_time_payee ? `<div style="font-size:14px;color:#555;text-align:center">${L("First payment to this account", "इस खाते को पहला भुगतान")}</div>` : ""}
        <div style="flex:1"></div>
        <button class="big go" id="ph-approve">${L("Approve & enter UPI PIN", "मंज़ूर करें और UPI PIN डालें")}</button>
        <button class="big soft" id="ph-decline">${L("Decline", "अस्वीकार करें")}</button>`;
      $("ph-approve").onclick = () => { S.phoneStep = 1; renderPhone(); renderSide(); };
      $("ph-decline").onclick = () => { el.innerHTML = head + `<div class="ok">${L("Declined. No money moved.", "अस्वीकार किया गया। कोई पैसा नहीं गया।")}</div>`; };
      return;
    }
    const action = S.withCitadel ? d.action : "A0";
    const m = d.message && d.message[S.lang];
    if (action === "A0" || !m || !m.title) {
      const lost = s.kind !== "legit";
      el.innerHTML = head + `<div class="${lost ? "lost" : "ok"}"><b>${inr(p.amount_inr)} ${L("sent", "भेजे गए")}</b><br>
        ${L("Payment successful.", "भुगतान सफल।")}</div>` +
        (lost && s.after ? `<div style="font-size:15px">${esc(L(s.after.en, s.after.hi))}</div>` : "") +
        (S.withCitadel && s.kind === "scam_missed" ? `<div style="font-size:14px;color:#555">${L("Citadel did not intervene on this one. See the explanation.", "Citadel ने इसमें दख़ल नहीं दिया। विवरण देखें।")}</div>` : "");
      return;
    }
    const btn = (b) => `<button class="big ${b.id === "cancel" || b.id === "call_bank" ? "stop" : "soft"}" data-b="${esc(b.id)}" ${b.id === "continue_after" ? "disabled" : ""}>${esc(b.label)}</button>`;
    el.innerHTML = head + `<div class="alert ${esc(action)}"><h4>${esc(m.title)}</h4><div>${esc(m.body)}</div>
        <ul>${(m.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>
      <div style="font-size:16px;font-weight:600">${esc(m.screen)}</div>
      ${action === "A2" ? `<div class="count" id="ph-count">${m.cooling_off_s}s</div><button class="btn" id="ph-skip" style="font-size:12px">${L("Demo: skip the wait", "डेमो: इंतज़ार छोड़ें")}</button>` : ""}
      <div style="flex:1"></div>${(m.buttons || []).map(btn).join("")}`;
    el.querySelectorAll("button[data-b]").forEach((b) => (b.onclick = () => phoneButton(b.dataset.b)));
    if (action === "A2") {
      let left = m.cooling_off_s;
      const done = () => { clearInterval(S.timer); const c = el.querySelector('[data-b="continue_after"]'); if (c) c.disabled = false; $("ph-count").textContent = L("You may continue", "अब आप आगे बढ़ सकते हैं"); };
      S.timer = setInterval(() => { left -= 1; if (left <= 0) done(); else $("ph-count").textContent = left + "s"; }, 1000);
      $("ph-skip").onclick = done;
    }
  }
  function phoneButton(id) {
    const s = scenarios()[S.scen], el = $("phone");
    clearInterval(S.timer);
    const msgs = {
      cancel: L("Payment cancelled. Your money is safe.", "भुगतान रद्द। आपका पैसा सुरक्षित है।"),
      call_bank: L("Calling the number printed on your card… The payment stays paused.", "आपके कार्ड पर छपे नंबर पर कॉल हो रहा है… भुगतान रुका रहेगा।"),
      ask_trusted: L("Share this screen with someone you trust before you continue.", "आगे बढ़ने से पहले किसी भरोसेमंद व्यक्ति से पूछें।"),
      request_release: L("Release requested. A bank officer will call you; the review SLA is shown in the queue.", "भुगतान जारी करने का अनुरोध भेजा गया। बैंक अधिकारी आपको कॉल करेंगे।"),
      appeal: L("Appeal raised. It goes to a human with a 24-hour SLA.", "अपील दर्ज। एक व्यक्ति 24 घंटे में जाँच करेगा।"),
      continue: L("Payment sent.", "भुगतान भेजा गया।"),
      continue_after: L("Payment sent after the cooling-off.", "रुकने के बाद भुगतान भेजा गया।"),
    };
    const safe = ["cancel", "call_bank", "request_release", "appeal", "ask_trusted"].includes(id);
    el.insertAdjacentHTML("beforeend", `<div class="${safe ? "ok" : s.kind === "legit" ? "ok" : "lost"}">${esc(msgs[id] || id)}</div>`);
  }
  function renderSide() {
    const s = scenarios()[S.scen]; if (!s) return;
    const d = s.decision;
    const kindLabel = { scam: "Scam (caught)", scam_missed: "Scam (honest miss)", legit: "Legitimate payment" }[s.kind] || s.kind;
    $("custside").innerHTML = `<h3>${esc(s.title)}</h3><div class="row"><span class="pill">${esc(kindLabel)}</span>
        ${s.variant ? `<span class="pill">${esc(s.variant)}</span>` : ""}${s.withheld ? '<span class="pill">withheld from training</span>' : ""}</div>
      <p>${esc(s.story || "")}</p>
      <h3>What Citadel decided</h3><p>${act(d.action)} <span class="muted">band ${esc(d.band)}</span></p>
      <div>${(d.reason_codes || []).map((c) => `<span class="chip">${esc(c)}</span>`).join("") || '<span class="muted">no reasons (no alert)</span>'}</div>
      ${s.explanation ? `<h3>Why</h3><p>${esc(s.explanation)}</p>` : ""}
      ${s.signals ? `<h3>What the institution could see</h3><div class="tablewrap"><table>${Object.entries(s.signals).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v === null ? '<span class="muted">not observable</span>' : esc(v)}</td></tr>`).join("")}</table></div>` : ""}
      <div class="src">${esc(s.source || "")}</div>`;
  }

  // ---------------------------------------------------------------- analyst queue
  async function initAnalyst() {
    if (S.mode === "live") {
      try { S.alerts = (await api("/v1/alerts?limit=100")).alerts; } catch (e) { $("qnotice").innerHTML = `<div class="notice err">${esc(e.message)}</div>`; }
    } else {
      S.alerts = (R && R.alerts) || [];
      $("qnotice").innerHTML = '<div class="notice">Replay mode: dispositions are disabled (they need the live API). The queue below was recorded from it.</div>';
    }
    $("qcount").textContent = S.alerts.length;
    $("qlist").innerHTML = S.alerts.map((a, i) => `<div class="qitem" tabindex="0" data-i="${i}" aria-selected="false">
        <div class="row">${act(a.action)} <b>${inr(a.amount_inr)}</b> <span class="muted">${esc(a.txn_type)}</span>
        <span class="muted" style="margin-left:auto">risk ${fmt(a.risk, 3)}</span></div>
        <div>${(a.reasons || []).map((c) => `<span class="chip">${esc(c)}</span>`).join("")}</div></div>`).join("") || '<p class="muted">Queue empty.</p>';
    document.querySelectorAll(".qitem").forEach((el) => {
      const open = () => { document.querySelectorAll(".qitem").forEach((x) => x.setAttribute("aria-selected", "false")); el.setAttribute("aria-selected", "true"); openAlert(+el.dataset.i); };
      el.addEventListener("click", open); el.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    });
    const ap = S.mode === "live" ? await api("/v1/appeals").then((j) => j.appeals).catch(() => []) : (R && R.appeals) || [];
    $("appeals").innerHTML = ap.length ? `<table><tr><th>appeal</th><th>alert</th><th>reason</th><th>status</th></tr>${ap.map((a) => `<tr><td>${esc(a.appeal_id)}</td><td>${esc(a.alert_id)}</td><td>${esc(a.reason)}</td><td>${esc(a.status)}</td></tr>`).join("")}</table>` : '<p class="muted">No open appeals.</p>';
  }
  async function openAlert(i) {
    const a = S.alerts[i];
    let ev = null;
    try { ev = S.mode === "live" ? await api("/v1/alerts/" + a.alert_id) : (R.evidence || {})[a.alert_id] || null; } catch (e) { ev = null; }
    const sla = isNum(a.sla_due) && isNum(a.inserted_at) ? Math.round((a.sla_due - a.inserted_at) / 60) + " min SLA" : "";
    const e = (ev && ev.evidence) || {};
    const lay = e.layers || {};
    const bar = (name, v) => `<div>${esc(name)}</div><div class="bar"><i style="width:${isFinite(num(v)) ? Math.min(100, 100 * num(v)) : 0}%"></i></div><div>${fmt(v, 3)}</div>`;
    $("qdetail").innerHTML = panelHTML(() => `<div class="row">${act(a.action)} <b>${inr(a.amount_inr)}</b> <span class="pill">${esc(a.txn_type)}</span>
        ${sla ? `<span class="pill">${esc(sla)}</span>` : ""}<span class="pill">alert ${esc(a.alert_id)}</span></div>
      <h3>Reasons</h3>${(e.reasons_detail || (a.reasons || []).map((c) => ({ code: c, analyst: "" }))).map((r) => `<div><span class="chip">${esc(r.code)}</span> ${esc(r.analyst)}</div>`).join("")}
      ${(a.l0_fired || []).length ? `<p>L0 rules fired: ${(a.l0_fired || []).map((x) => `<span class="chip">${esc(x)}</span>`).join("")}</p>` : ""}
      <h3>Layers</h3><div class="bars">${bar("L1 supervised", lay.p_L1)}${bar("L2 payee graph", lay.p_L2)}${bar("fused risk", lay.risk)}</div>
      <h3>Payee</h3><p class="muted">${e.payee ? `age ${fmt(e.payee.payee_age_days, 0)} days · ${esc(e.payee.inbound_unique_payers_24h)} distinct payers in 24h · ${esc(e.payee.complaints)} corroborated complaints · ${e.payee.verified_merchant ? "verified merchant" : "not a verified merchant"}` : "—"}</p>
      <h3>Payer's last 7 days</h3>${(e.timeline || []).length ? `<table><tr><th>time</th><th>type</th><th>amount</th><th>payee</th><th>new payee</th></tr>${e.timeline.map((t) => `<tr><td>${new Date(t.ts * 1000).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}</td><td>${esc(t.txn_type)}</td><td>${inr(t.amount_inr)}</td><td><code>${esc(t.payee_hash)}</code></td><td>${t.first_time_payee ? "yes" : ""}</td></tr>`).join("")}</table>` : '<p class="muted">No earlier payments in 7 days.</p>'}
      <div class="row" style="margin-top:12px">${["confirm_scam", "release", "needs_info"].map((d) => `<button class="btn ${d === "release" ? "" : "primary"}" data-d="${d}" ${S.mode === "live" ? "" : "disabled"}>${d.replace("_", " ")}</button>`).join("")}</div>
      <div class="src">GET /v1/alerts/${esc(a.alert_id)} · audited</div>`);
    $("qdetail").querySelectorAll("button[data-d]").forEach((b) => (b.onclick = async () => {
      try { await api("/v1/cases/" + a.alert_id, { method: "PUT", body: JSON.stringify({ disposition: b.dataset.d, note: "via console" }) }); b.textContent = "✓ " + b.textContent; } catch (err) { alert(err.message); }
    }));
  }
  function panelHTML(fn) { try { return fn(); } catch (e) { return `<div class="notice err">${esc(e.message)}</div>`; } }

  // ---------------------------------------------------------------- studio
  function initStudio() {
    const g = S.grammar;
    if (!g) { $("sout").innerHTML = '<div class="notice">Grammar not available.</div>'; return; }
    $("sv").innerHTML = Object.entries(g.variants).map(([k, v]) => `<option value="${k}">${k} · ${esc(v.title)}</option>`).join("");
    $("ss").innerHTML = Object.entries(g.segments).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
    const illegal = {};
    (g.illegal || []).forEach((r) => { illegal[`${r.variant}|${r.evasion}|${r.segment}`] = r.reason; });
    const refresh = () => {
      const v = $("sv").value, s = $("ss").value;
      const cur = $("se").value || "none";
      $("se").innerHTML = ["none", ...Object.keys(g.evasions)].map((e) => {
        const why = illegal[`${v}|${e}|${s}`];
        const label = e === "none" ? "none" : g.evasions[e].title;
        return `<option value="${e}" ${why ? "disabled" : ""} title="${esc(why || "")}">${esc(label)}${why ? " (not possible)" : ""}</option>`;
      }).join("");
      if ([...$("se").options].some((o) => o.value === cur && !o.disabled)) $("se").value = cur;
      const why = illegal[`${v}|${$("se").value}|${s}`] || illegal[`${v}|none|${s}`];
      const disabledEv = Object.keys(g.evasions).filter((e) => illegal[`${v}|${e}|${s}`] && !illegal[`${v}|none|${s}`]).map((e) => `${g.evasions[e].title}: ${illegal[`${v}|${e}|${s}`].split(": ").slice(1).join(": ")}`);
      $("swhy").innerHTML = why ? `<span class="neg">Not possible:</span> ${esc(why)}` : disabledEv.length ? "Disabled for this variant: " + disabledEv.map(esc).join(" · ") : `${esc(g.size_legal)} of ${esc(g.size_total)} compositions are legal.`;
      $("srun").disabled = !!why;
    };
    ["sv", "ss"].forEach((id) => $(id).addEventListener("change", refresh));
    $("se").addEventListener("change", refresh);
    refresh();
    $("srun").addEventListener("click", runStudio);
  }
  async function runStudio() {
    const key = `${$("sv").value}|${$("se").value}|${$("ss").value}`;
    $("sout").innerHTML = '<p class="muted">Running…</p>';
    let t = null;
    try {
      if (S.mode === "live") t = await api("/v1/scam/author", { method: "POST", body: JSON.stringify({ variant: $("sv").value, evasion: $("se").value, segment: $("ss").value, seed: 0 }) });
      else t = (R && R.studio && R.studio[key]) || null;
    } catch (e) { $("sout").innerHTML = `<div class="notice err">${esc(e.message)}</div>`; return; }
    if (!t) { $("sout").innerHTML = `<div class="notice">Replay mode: this composition was not recorded. Recorded: ${Object.keys((R && R.studio) || {}).length} compositions (one per variant × legal evasion). Start the API for any composition.</div>`; return; }
    panel($("sout"), () => renderTrace(t));
  }
  function renderTrace(t) {
    const c = t.composition;
    const badges = [c.withheld_variant ? "variant withheld from training" : "", c.withheld_evasion ? "evasion withheld from training" : ""].filter(Boolean);
    const steps = t.steps.map((s) => {
      const fired = t.fired_at && s.step === t.fired_at.step && s.action && s.action === t.fired_at.action && s.money !== "not attempted (customer already protected)" && s.money !== "moved";
      if (!s.observed && s.step === "S1") return `<div class="step unobs"><div class="tag">S1</div><div class="card"><b>${esc(s.name)}</b> — <span class="muted">not observed by the institution</span><br>${esc(s.what_happens)}<div class="muted">${esc(s.note)}</div></div></div>`;
      if (s.step === "S4") return `<div class="step"><div class="tag">S4</div><div class="card"><b>${esc(s.name)}</b><br>${s.mule_outflows ? `${esc(s.mule_outflows)} mule payouts totalling ${inr(s.mule_outflow_inr)} would follow` : "no visible mule outflow"}<div class="muted">${esc(s.note)}</div></div></div>`;
      const L_ = s.layers || {};
      const bar = (name, v) => `<div>${esc(name)}</div><div class="bar"><i style="width:${Math.min(100, 100 * (num(v) || 0))}%"></i></div><div>${fmt(v, 3)}</div>`;
      const sig = Object.entries(s.signals || {}).filter(([, v]) => v !== null && v !== 0).map(([k, v]) => `<span class="chip">${esc(k)}=${esc(typeof v === "number" ? +v.toFixed(2) : v)}</span>`).join("");
      return `<div class="step ${fired ? "fired" : ""}"><div class="tag">${esc(s.step)}</div><div class="card">
        <div class="row">${act(s.action)} <b>${inr(s.amount_inr)}</b> <span class="pill">${esc(s.txn_type)}</span><span class="pill">${esc(s.time_ist)}</span>
          <span class="pill">payee age ${fmt(s.payee_age_days, 0)}d${s.first_time_payee ? " · first payment" : ""}</span><span class="pill">${esc(s.money)}</span></div>
        <div style="margin-top:6px">${sig || '<span class="muted">no session anomaly signals observed</span>'}</div>
        <div class="bars" style="margin-top:8px">${bar("L1 supervised", L_.L1_supervised)}${bar("L2 payee graph", L_.L2_payee_graph)}${bar("fused risk", L_.fused_risk)}</div>
        <div style="margin-top:6px">${(L_.L0_rules_fired || []).map((x) => `<span class="chip">${esc(x)}</span>`).join("")}${(s.reasons || []).map((x) => `<span class="chip">${esc(x)}</span>`).join("")}${L_.abstained_to_human ? '<span class="chip">abstained → human</span>' : ""}</div>
        ${s.message && s.message.en && s.message.en.title ? `<div class="muted" style="margin-top:6px">Customer sees: “${esc(s.message.en.title)}” / “${esc(s.message.hi.title)}”</div>` : ""}
      </div></div>`;
    }).join("");
    $("sout").innerHTML = `<div class="card"><div class="row"><h3 style="margin:0">${esc(c.variant)} · ${esc(c.title)}</h3>
        <span class="pill">evasion: ${esc(c.evasion)}</span><span class="pill">segment: ${esc(c.segment)}</span>${badges.map((b) => `<span class="pill">${esc(b)}</span>`).join("")}</div>
        <p class="muted">${esc(c.story)}</p>
        <p>${t.caught ? `<span class="pos">Citadel fired at ${esc(t.fired_at.step)} (${esc(t.fired_at.action)})</span>` : '<span class="neg">Citadel did not fire (A2+) on this composition</span>'} ·
          money moved ${inr(t.money_moved_inr)} · protected ${inr(t.money_protected_inr)} (if the warning is heeded)</p>
        <p class="muted">${esc(t.bound)}</p></div><div class="tl" style="margin-top:12px">${steps}</div>`;
  }

  // ---------------------------------------------------------------- results
  function renderResults() {
    if (!S.results || !S.results.files) { $("resnotice").innerHTML = '<div class="notice">No results recorded.</div>'; return; }
    const H = file("defend_headline.csv") || [];
    const rep = H.length && H.every((r) => r.reportable === true || r.reportable === "True");
    if (!rep) $("resnotice").innerHTML = '<div class="notice">These numbers are flagged NOT REPORTABLE (reduced-scale run or too few positives). Do not quote them.</div>';
    const B = (file("defend_alert_budget.csv") || [])[0] || {};
    const T = file("eval_time_to_alert_summary.csv") || [];
    const kpi = (label, row, src, asPct = true) => `<div class="card kpi"><div class="l">${esc(label)}</div>
      <div class="v">${row ? (asPct ? pct(row.value) : fmt(row.value)) : "—"}</div>
      <div class="ci">${row && row.ci_low !== null && row.ci_low !== undefined ? `95% CI ${asPct ? pct(row.ci_low) : fmt(row.ci_low)} – ${asPct ? pct(row.ci_high) : fmt(row.ci_high)}` : ""}
        ${row ? ` · n=${esc(row.n_positives ?? row.denominator ?? "—")} positives` : ""}</div><div class="src">${esc(src)}</div></div>`;
    $("kpis").innerHTML =
      kpi("PR-AUC (test window)", byKey(H, "metric", "pr_auc"), "defend_headline.csv", false) +
      kpi("Recall at 0.5% false-positive rate", byKey(H, "metric", "recall_at_0.5%_fpr"), "defend_headline.csv") +
      kpi("Recall at 0.1% false-positive rate", byKey(H, "metric", "recall_at_0.1%_fpr"), "defend_headline.csv") +
      kpi("Scams warned or held (deployed ladder, A2+)", byKey(H, "metric", "recall_at_deployed_A2plus"), "defend_headline.csv") +
      kpi("Episodes stopped at the first harmful payment", byKey(T, "metric", "episodes_caught_by_first_harmful_payment"), "eval_time_to_alert_summary.csv") +
      `<div class="card kpi"><div class="l">Alert load per 1,000 payments</div><div class="v">${fmt(B.warnings_A2plus_per_1000, 1)} <span style="font-size:14px" class="muted">warnings</span></div>
        <div class="ci">${fmt(B.holds_A3_per_1000, 2)} holds · analyst capacity ${fmt(B.analyst_capacity_k_per_1000, 2)} per 1,000</div><div class="src">defend_alert_budget.csv</div></div>`;
    panel($("opcurve"), drawCurve);
    panel($("gen"), drawGen);
    panel($("tta"), drawTTA);
    panel($("pervar"), drawPerVar);
    panel($("abl"), () => table("abl", file("eval_layer_ablation.csv"), ["step", "n_features", "pr_auc", "recall_at_0.5pct_fpr", "delta_pr_auc_vs_previous", "delta_ci_low", "delta_ci_high", "verdict"], { verdict: (v) => `<span class="${v === "improves" ? "pos" : v ? "neg" : ""}">${esc(v || "")}</span>` }));
    panel($("base"), () => table("base", file("eval_baselines.csv"), ["model", "pr_auc", "recall_at_0.1%_fpr", "recall_at_0.5%_fpr", "citadel_minus_this_recall_0.5pct", "verdict"]));
    panel($("hn"), () => table("hn", (file("eval_hard_negative_fpr.csv") || []).slice(0, 14), ["population", "n_rows", "fpr_A1plus", "fpr_A2plus", "fpr_A3"], { fpr_A1plus: (v) => pct(v, 2), fpr_A2plus: (v) => pct(v, 2), fpr_A3: (v) => pct(v, 3) }));
    panel($("fair"), () => table("fair", file("eval_fairness.csv"), ["group", "value_of", "value", "denominator", "fpr_A2plus", "fpr_ratio_to_overall", "flag_fpr_disparity"], { value: (v) => pct(v), fpr_A2plus: (v) => pct(v, 2) }));
    panel($("evidx"), () => table("evidx", file("eval_evidence_index.csv"), ["claim_id", "value", "provenance_tier", "n_rows", "n_positives", "reportable", "artefact"]));
  }
  function table(id, rows, cols, fmts = {}) {
    if (!rows || !rows.length) { $(id).innerHTML = '<p class="muted">—</p>'; return; }
    const cell = (c, v) => fmts[c] ? fmts[c](v) : typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString("en-IN") : fmt(v)) : v === null || v === undefined ? "—" : esc(v);
    $(id).innerHTML = `<table><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${rows.map((r) => `<tr>${cols.map((c) => `<td>${cell(c, r[c])}</td>`).join("")}</tr>`).join("")}</table>`;
  }
  function svg(w, h, inner) { return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img">${inner}</svg>`; }
  function drawCurve() {
    const rows = (file("eval_operating_curve.csv") || []).filter((r) => isFinite(num(r.alerts_per_1000)));
    if (!rows.length) { $("opcurve").innerHTML = '<p class="muted">—</p>'; return; }
    const W = 520, H = 260, pl = 44, pb = 34, pr = 12, pt = 10;
    const xs = rows.map((r) => Math.log10(num(r.alerts_per_1000))), x0 = Math.min(...xs), x1 = Math.max(...xs);
    const X = (v) => pl + (Math.log10(v) - x0) / (x1 - x0 || 1) * (W - pl - pr), Y = (v) => H - pb - v * (H - pb - pt);
    const path = rows.map((r, i) => `${i ? "L" : "M"}${X(num(r.alerts_per_1000)).toFixed(1)},${Y(num(r.recall)).toFixed(1)}`).join("");
    const marks = rows.filter((r) => r.ladder_point).map((r) => `<line x1="${X(num(r.alerts_per_1000))}" x2="${X(num(r.alerts_per_1000))}" y1="${pt}" y2="${H - pb}" stroke="var(--${String(r.ladder_point).toLowerCase()})" stroke-dasharray="4 3"/><text x="${X(num(r.alerts_per_1000)) + 3}" y="${pt + 12}">${esc(r.ladder_point)} budget</text><circle cx="${X(num(r.alerts_per_1000))}" cy="${Y(num(r.recall))}" r="4" fill="var(--${String(r.ladder_point).toLowerCase()})"/><text x="${X(num(r.alerts_per_1000)) + 6}" y="${Y(num(r.recall)) + 14}">${pct(r.recall, 0)}</text>`).join("");
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((v) => `<line class="axis" x1="${pl}" x2="${W - pr}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${pl - 6}" y="${Y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("");
    const xt = [0.5, 1, 2, 5, 10, 20, 50].filter((v) => Math.log10(v) >= x0 && Math.log10(v) <= x1).map((v) => `<text x="${X(v)}" y="${H - pb + 16}" text-anchor="middle">${v}</text>`).join("");
    $("opcurve").innerHTML = svg(W, H, ticks + xt + `<path class="ln" d="${path}"/>` + marks + `<text x="${(W + pl) / 2}" y="${H - 4}" text-anchor="middle">alerts per 1,000 payments (log scale)</text>`) + '<p class="muted" style="font-size:12px">Recall of scam payments as the alert budget grows. Dashed lines: the deployed A1/A2/A3 budgets.</p>';
  }
  function short(label) {
    const t = String(label).replace(/_/g, " ").replace("withheld evasion signal suppression", "withheld evasion")
      .replace("episodes caught ", "").replace("A2plus", "warning/hold");
    return t.length > 27 ? t.slice(0, 26) + "…" : t;
  }
  function whisker(rows, labelKey, valKey, loKey, hiKey) {
    const W = 400, rowH = 30, pl = 175, pr = 42, H = rows.length * rowH + 22;
    const X = (v) => pl + v * (W - pl - pr);
    const inner = rows.map((r, i) => {
      const y = 14 + i * rowH, v = num(r[valKey]), lo = num(r[loKey]), hi = num(r[hiKey]);
      return `<text x="${pl - 8}" y="${y + 4}" text-anchor="end">${esc(short(r[labelKey]))}</text>` +
        (isFinite(v) ? `<rect x="${pl}" y="${y - 7}" width="${Math.max(1, X(v) - pl)}" height="14" fill="var(--accent)" opacity=".75" rx="3"/>` : "") +
        (isFinite(lo) && isFinite(hi) ? `<line x1="${X(lo)}" x2="${X(hi)}" y1="${y}" y2="${y}" stroke="var(--ink)"/><line x1="${X(lo)}" x2="${X(lo)}" y1="${y - 5}" y2="${y + 5}" stroke="var(--ink)"/><line x1="${X(hi)}" x2="${X(hi)}" y1="${y - 5}" y2="${y + 5}" stroke="var(--ink)"/>` : "") +
        `<text x="${W - pr + 6}" y="${y + 4}">${pct(v, 0)}</text>`;
    }).join("");
    return svg(W, H, inner);
  }
  function drawGen() {
    const g = file("eval_generalisation.csv") || [];
    const lovo = file("eval_lovo.csv") || [];
    const rows = g.map((r) => ({ label: `${r.group} (n=${r.denominator})`, value: r.value, lo: r.ci_low, hi: r.ci_high }));
    $("gen").innerHTML = (rows.length ? whisker(rows, "label", "value", "lo", "hi") : '<p class="muted">—</p>') +
      '<p class="muted" style="font-size:12px">Recall at the deployed A2+ operating point, 95% Wilson intervals. The withheld variant and evasion never appear in training (entity-level sealing).</p>' +
      (lovo.length ? "<h3>Leave-one-variant-out (L1 retrained without it)</h3>" + whisker(lovo.map((r) => ({ label: `${r.variant} withheld (n=${r.n_positives})`, value: r["recall_at_0.5pct_fpr_when_withheld"], lo: r.ci_low, hi: r.ci_high })), "label", "value", "lo", "hi") : "");
  }
  function drawTTA() {
    const T = file("eval_time_to_alert_summary.csv") || [];
    const rows = ["episodes_caught_before_first_loss", "episodes_caught_by_first_harmful_payment", "episodes_caught_any_A2plus", "first_alert_at_step_S2", "first_alert_at_step_S3"]
      .map((m) => byKey(T, "metric", m)).filter(Boolean).map((r) => ({ label: r.metric, value: r.value, lo: r.ci_low, hi: r.ci_high }));
    const saved = byKey(T, "metric", "value_saved_share_heed_adjusted");
    $("tta").innerHTML = (rows.length ? whisker(rows, "label", "value", "lo", "hi") : '<p class="muted">—</p>') +
      `<p>Share of scam value saved: <b>${saved ? pct(saved.value) : "—"}</b> <span class="muted">(heed rates are assumptions, configs/costs.yaml)</span></p>`;
  }
  function drawPerVar() {
    const rows = (file("eval_per_variant.csv") || []).filter((r) => r.value_of !== "benign").map((r) => ({ label: `${r.value_of} (n=${r.denominator})`, value: r.value, lo: r.ci_low, hi: r.ci_high }));
    $("pervar").innerHTML = rows.length ? whisker(rows, "label", "value", "lo", "hi") : '<p class="muted">—</p>';
  }

  // ---------------------------------------------------------------- safeguards
  function renderSafeguards() {
    const md = (S.results && S.results.report_md) || "";
    const sec = md.split("## Known limitations")[1] || "";
    const items = sec.split("\n## ")[0].split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>"));
    $("limits").innerHTML = items.length ? "<ul>" + items.map((i) => `<li>${i.replace(/<(?!\/?b>)/g, "&lt;")}</li>`).join("") + "</ul>" : '<p class="muted">—</p>';
    const pol = (R && R.policy) || null;
    $("collect").innerHTML = pol ? `<table><tr><th>signal</th><th>why</th><th>consent</th></tr>${pol.collect.map((r) => `<tr><td>${esc(r.signal)}</td><td>${esc(r.why)}</td><td>${esc(r.consent)}</td></tr>`).join("")}</table><h3>Never collected</h3><ul>${pol.never.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : '<p class="muted">—</p>';
    $("fp").innerHTML = pol ? `<ul>${pol.false_positive.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : '<p class="muted">—</p>';
    const tests = (R && R.tests) || [];
    $("tests").innerHTML = tests.length ? `<table><tr><th>test</th><th>result</th></tr>${tests.map((t) => `<tr><td><code>${esc(t.id)}</code></td><td class="${t.outcome === "passed" ? "pos" : "neg"}">${esc(t.outcome)}</td></tr>`).join("")}</table>` : '<p class="muted">—</p>';
  }

  boot();
})();
