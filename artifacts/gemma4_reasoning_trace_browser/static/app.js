"use strict";

const MULTI_FIELDS = [
  "parent", "source", "correctness", "outcome", "clause_split", "presentation",
  "episode_kind", "target_clause", "clause_family", "template_id", "tag",
];
const SINGLE_FIELDS = [
  "boundary_valid", "grammar_valid", "format_valid", "truncated",
  "tokens_min", "tokens_max", "reward_min", "reward_max", "step_min", "step_max",
];
const FRIENDLY = {
  parent: "Parent", source: "Source", correctness: "Score", outcome: "Outcome",
  clause_split: "Clause", presentation: "Template", episode_kind: "Episode",
  target_clause: "Clause target", clause_family: "Family", template_id: "Template ID",
  tag: "Tag", boundary_valid: "Boundary", grammar_valid: "Grammar",
  format_valid: "Format", truncated: "Truncated", tokens_min: "Min tokens",
  tokens_max: "Max tokens", reward_min: "Min reward", reward_max: "Max reward",
  step_min: "From step", step_max: "To step",
};
const VALUE_LABELS = {
  public: "Public IT", graft: "Charter graft", rollout: "Rollout", eval: "Eval",
  correct: "Correct", partial: "Partial", incorrect: "Incorrect", malformed: "Malformed",
  all_charter: "All Charter", all_coin: "All Coin", no_conflict: "No conflict",
  mixed: "Mixed", impure: "Third crew", trained: "Trained", heldout: "Held out",
  canonical: "Canonical", adjacent: "Adjacent", agreement: "Agreement", conflict: "Conflict",
  "1": "Yes", "0": "No",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const fmt = new Intl.NumberFormat("en-GB");
const percent = value => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
const esc = value => String(value ?? "").replace(/[&<>'"]/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[char]);
const label = value => VALUE_LABELS[value] || String(value || "—").replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
const slug = value => String(value || "").replaceAll("_", "-");

let meta = null;
let state = {
  q: "", scope: "output", sort: "step_asc", offset: 0, limit: 50,
  selected: null, filters: Object.fromEntries([...MULTI_FIELDS, ...SINGLE_FIELDS].map(f => [f, MULTI_FIELDS.includes(f) ? [] : ""])),
};
let currentResult = { items: [], total: 0 };
let selectedTrace = null;
let fetchController = null;
let debounceTimer = null;
let toastTimer = null;

function readUrlState() {
  const params = new URLSearchParams(location.search);
  state.q = params.get("q") || "";
  state.scope = params.get("scope") || "output";
  state.sort = params.get("sort") || "step_asc";
  state.offset = Math.max(0, Number(params.get("offset")) || 0);
  state.selected = params.get("trace") || null;
  MULTI_FIELDS.forEach(field => state.filters[field] = params.getAll(field));
  SINGLE_FIELDS.forEach(field => state.filters[field] = params.get(field) || "");
}

function queryParams({ includePaging = true, includeTrace = false } = {}) {
  const params = new URLSearchParams();
  if (state.q) params.set("q", state.q);
  if (state.scope !== "output") params.set("scope", state.scope);
  if (state.sort !== "step_asc") params.set("sort", state.sort);
  for (const field of MULTI_FIELDS) {
    for (const value of state.filters[field]) params.append(field, value);
  }
  for (const field of SINGLE_FIELDS) {
    if (state.filters[field] !== "") params.set(field, state.filters[field]);
  }
  if (includePaging) {
    params.set("limit", state.limit);
    if (state.offset) params.set("offset", state.offset);
  }
  if (includeTrace && state.selected) params.set("trace", state.selected);
  return params;
}

function syncUrl() {
  const params = queryParams({ includePaging: true, includeTrace: true });
  params.delete("limit");
  const query = params.toString();
  history.replaceState(null, "", query ? `?${query}` : location.pathname);
}

function facet(field) {
  return meta?.facets?.[field] || [];
}

function renderCorpusFacts() {
  const build = meta.build;
  const evals = facet("source").find(item => item.value === "eval")?.count || 0;
  const rollouts = facet("source").find(item => item.value === "rollout")?.count || 0;
  $("#datasetFacts").innerHTML = `
    <div class="fact"><strong>${fmt.format(build.trace_count)}</strong><span>Total traces</span></div>
    <div class="fact"><strong>${fmt.format(rollouts)}</strong><span>Rollouts</span></div>
    <div class="fact"><strong>${fmt.format(evals)}</strong><span>Eval traces</span></div>
    <div class="fact"><strong>0 · 64 · 128</strong><span>Eval checkpoints</span></div>`;
  $("#statusDot").classList.add("live");
  $("#statusDot").title = `${fmt.format(build.trace_count)} indexed traces · server connected`;
}

function optionHtml(field, item, segment = false) {
  const checked = state.filters[field].includes(String(item.value));
  const title = `${label(item.value)} · ${fmt.format(item.count)} corpus traces`;
  return `<label title="${esc(title)}">
    <input type="checkbox" data-filter="${esc(field)}" value="${esc(item.value)}" ${checked ? "checked" : ""}>
    <span class="option-text">${esc(label(item.value))}${segment ? ` <small>${fmt.format(item.count)}</small>` : ""}</span>
    ${segment ? "" : `<small>${fmt.format(item.count)}</small>`}
  </label>`;
}

function renderFacets() {
  for (const field of MULTI_FIELDS.filter(field => field !== "tag")) {
    const container = $(`[data-field="${field}"]`);
    if (!container) continue;
    const items = facet(field);
    container.innerHTML = items.map(item => optionHtml(field, item, container.classList.contains("segment"))).join("");
  }
  $("#quickSteps").innerHTML = [0, 64, 128].map(step => `<button type="button" data-quick-step="${step}">step ${step}</button>`).join("") + `<button type="button" data-quick-step="all">all</button>`;
  renderTagGroups();
}

function renderTagGroups(search = "") {
  const groups = {};
  const needle = search.toLocaleLowerCase();
  for (const item of facet("tag")) {
    const [prefix, ...rest] = item.value.split(":");
    if (["parent", "source", "correctness", "outcome", "kind", "clause-split", "presentation", "clause", "clause-family"].includes(prefix)) continue;
    if (needle && !item.value.toLocaleLowerCase().includes(needle)) continue;
    (groups[prefix] ||= []).push({ ...item, short: rest.join(":") });
  }
  $("#tagGroups").innerHTML = Object.entries(groups).map(([group, items]) => `
    <div class="tag-group">
      <h4>${esc(label(group))}</h4>
      <div class="tag-options">${items.map(item => `<label title="${esc(item.value)} · ${fmt.format(item.count)} traces">
        <input type="checkbox" data-filter="tag" value="${esc(item.value)}" ${state.filters.tag.includes(item.value) ? "checked" : ""}>
        <span>${esc(label(item.short))} · ${fmt.format(item.count)}</span>
      </label>`).join("")}</div>
    </div>`).join("") || `<p class="microcopy flush">No matching tags.</p>`;
}

function restoreControls() {
  $("#searchInput").value = state.q;
  $("#sortSelect").value = state.sort;
  $$(`[data-scope]`).forEach(button => button.classList.toggle("active", button.dataset.scope === state.scope));
  $("#stepMin").value = state.filters.step_min;
  $("#stepMax").value = state.filters.step_max;
  $$(`[data-single]`).forEach(control => control.value = state.filters[control.dataset.single]);
  $$(`[data-number]`).forEach(control => control.value = state.filters[control.dataset.number]);
}

function bindControls() {
  document.addEventListener("change", event => {
    const checkbox = event.target.closest("[data-filter]");
    if (checkbox) {
      const field = checkbox.dataset.filter;
      const values = new Set(state.filters[field]);
      checkbox.checked ? values.add(checkbox.value) : values.delete(checkbox.value);
      state.filters[field] = [...values];
      changed();
      return;
    }
    const single = event.target.closest("[data-single]");
    if (single) {
      state.filters[single.dataset.single] = single.value;
      changed();
      return;
    }
    const number = event.target.closest("[data-number]");
    if (number) {
      state.filters[number.dataset.number] = number.value;
      changed();
    }
  });
  $("#searchInput").addEventListener("input", event => {
    state.q = event.target.value.trim();
    changed(true);
  });
  $("#stepMin").addEventListener("change", event => { state.filters.step_min = event.target.value; changed(); });
  $("#stepMax").addEventListener("change", event => { state.filters.step_max = event.target.value; changed(); });
  $("#sortSelect").addEventListener("change", event => { state.sort = event.target.value; changed(); });
  $("#scopeRow").addEventListener("click", event => {
    const button = event.target.closest("[data-scope]");
    if (!button) return;
    state.scope = button.dataset.scope;
    $$(`[data-scope]`).forEach(b => b.classList.toggle("active", b === button));
    changed();
  });
  $("#quickSteps").addEventListener("click", event => {
    const button = event.target.closest("[data-quick-step]");
    if (!button) return;
    const value = button.dataset.quickStep;
    state.filters.step_min = value === "all" ? "" : value;
    state.filters.step_max = value === "all" ? "" : value;
    $("#stepMin").value = state.filters.step_min;
    $("#stepMax").value = state.filters.step_max;
    changed();
  });
  $("#tagSearch").addEventListener("input", event => renderTagGroups(event.target.value));
  $$(`[data-filter-list]`).forEach(input => input.addEventListener("input", event => {
    const list = $(`[data-field="${event.target.dataset.filterList}"]`);
    const needle = event.target.value.toLocaleLowerCase();
    $$(`label`, list).forEach(row => row.hidden = !row.textContent.toLocaleLowerCase().includes(needle));
  }));
  $$(".filter-title").forEach(button => button.addEventListener("click", () => {
    const body = button.parentElement.querySelector(".filter-body");
    const expanded = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", String(!expanded));
    if (body) body.hidden = expanded;
    const bodies = button.parentElement.querySelectorAll(".filter-body");
    bodies.forEach(item => item.hidden = expanded);
  }));
  $("#resetButton").addEventListener("click", resetAll);
  $("#previousPage").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); search(); });
  $("#nextPage").addEventListener("click", () => { state.offset += state.limit; search(); });
  $("#copyViewButton").addEventListener("click", copyViewLink);
  $("#exportJsonl").addEventListener("click", () => exportView("jsonl"));
  $("#exportCsv").addEventListener("click", () => exportView("csv"));
  $("#shortcutsButton").addEventListener("click", () => $("#shortcutsDialog").showModal());
  $("#shortcutsDialog .dialog-close").addEventListener("click", () => $("#shortcutsDialog").close());
  $("#shortcutsDialog").addEventListener("click", event => { if (event.target === event.currentTarget) event.currentTarget.close(); });
  document.addEventListener("keydown", keyboardNavigation);
}

function changed(debounce = false) {
  state.offset = 0;
  renderActiveFilters();
  syncUrl();
  clearTimeout(debounceTimer);
  if (debounce) debounceTimer = setTimeout(search, 260);
  else search();
}

function resetAll() {
  state.q = ""; state.scope = "output"; state.sort = "step_asc"; state.offset = 0;
  for (const field of MULTI_FIELDS) state.filters[field] = [];
  for (const field of SINGLE_FIELDS) state.filters[field] = "";
  state.selected = null; selectedTrace = null;
  renderFacets(); restoreControls(); renderActiveFilters(); syncUrl(); search();
  $("#detailPanel").innerHTML = emptyDetail();
}

function renderActiveFilters() {
  const container = $("#activeFilters");
  const entries = [];
  for (const field of MULTI_FIELDS) for (const value of state.filters[field]) entries.push({ field, value });
  for (const field of SINGLE_FIELDS) if (state.filters[field] !== "") entries.push({ field, value: state.filters[field] });
  container.hidden = !entries.length;
  container.innerHTML = entries.map(({ field, value }) => `<button class="active-pill" type="button" data-remove-field="${esc(field)}" data-remove-value="${esc(value)}">${esc(FRIENDLY[field])}: ${esc(label(value))}</button>`).join("");
  $$(`[data-remove-field]`, container).forEach(button => button.addEventListener("click", () => {
    const field = button.dataset.removeField;
    if (MULTI_FIELDS.includes(field)) state.filters[field] = state.filters[field].filter(value => value !== button.dataset.removeValue);
    else state.filters[field] = "";
    renderFacets(); restoreControls(); changed();
  }));
}

async function search() {
  clearTimeout(debounceTimer);
  if (fetchController) fetchController.abort();
  fetchController = new AbortController();
  $("#resultsList").innerHTML = `<div class="loading-state"><span class="spinner"></span>Searching ${fmt.format(meta.build.trace_count)} traces…</div>`;
  try {
    const response = await fetch(`/api/search?${queryParams().toString()}`, { signal: fetchController.signal });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
    currentResult = result;
    renderResults(); renderSummary(); renderPagination(); syncUrl();
  } catch (error) {
    if (error.name === "AbortError") return;
    $("#resultsList").innerHTML = `<div class="no-results"><strong>Query could not run</strong><p>${esc(error.message)}</p></div>`;
  }
}

function renderSummary() {
  const a = currentResult.aggregates;
  const correct = currentResult.summary.correctness.find(x => x.value === "correct")?.count || 0;
  const malformed = currentResult.summary.correctness.find(x => x.value === "malformed")?.count || 0;
  const total = currentResult.total || 1;
  const stats = [
    [percent(correct / total), "Task correct"],
    [percent(malformed / total), "Malformed"],
    [percent(a.format_rate), "Format valid"],
    [percent(a.avg_semantic), "Mean semantic"],
  ];
  if (a.avg_reward != null) stats.push([Number(a.avg_reward).toFixed(3), "Mean reward"]);
  $("#summaryStrip").innerHTML = stats.map(([value, name]) => `<div class="summary-stat"><strong>${esc(value)}</strong><span>${esc(name)}</span></div>`).join("");
}

function highlight(text, context) {
  text = String(text || "");
  if (!state.q || !["all", "output", context].includes(state.scope) && !(state.scope === "output" && ["reasoning", "response"].includes(context))) return esc(text);
  const terms = [...state.q.matchAll(/"([^"]+)"|(\S+)/g)].map(match => match[1] || match[2]).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!terms.length) return esc(text);
  const regex = new RegExp(terms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "gi");
  let html = "", last = 0;
  for (const match of text.matchAll(regex)) {
    html += esc(text.slice(last, match.index)) + `<mark>${esc(match[0])}</mark>`;
    last = match.index + match[0].length;
  }
  return html + esc(text.slice(last));
}

function renderResults() {
  $("#resultCount").textContent = fmt.format(currentResult.total);
  $("#resultNoun").textContent = currentResult.total === 1 ? " trace" : " traces";
  if (!currentResult.items.length) {
    $("#resultsList").innerHTML = `<div class="no-results"><strong>No traces match this view.</strong><p>Try removing a filter or broadening the output search.</p></div>`;
    return;
  }
  $("#resultsList").innerHTML = currentResult.items.map((item, index) => {
    const preview = item.reasoning_preview || item.response_preview;
    const reward = item.reward == null ? "" : `reward ${Number(item.reward).toFixed(2)}`;
    return `<article class="result-card ${String(state.selected) === String(item.uid) || String(state.selected) === String(item.id) ? "selected" : ""}" data-trace="${esc(item.uid)}" data-index="${index}" tabindex="0">
      <div class="card-top">
        <div class="badge-row">
          <span class="badge parent-${esc(item.parent)}">${esc(label(item.parent))}</span>
          <span class="badge">${esc(label(item.source))} · ${item.checkpoint_step}</span>
          <span class="badge ${esc(slug(item.correctness))}">${esc(label(item.correctness))}</span>
          <span class="badge ${esc(slug(item.outcome))}">${esc(label(item.outcome))}</span>
        </div>
        <span class="trace-index">${state.offset + index + 1}</span>
      </div>
      <div class="card-title">${esc(item.episode_id)} · ${esc(label(item.episode_kind))}${item.target_clause ? ` · ${esc(item.target_clause)}` : ""}</div>
      <p class="card-preview ${preview ? "" : "empty"}">${preview ? highlight(preview, "reasoning") : "No committed reasoning or response"}</p>
      <div class="card-bottom">
        <span>${esc(label(item.clause_split))} clause · ${esc(label(item.presentation))} template</span>
        <span class="metrics">${reward ? `<span>${reward}</span>` : ""}<span>${fmt.format(item.completion_tokens || 0)} tok</span></span>
      </div>
    </article>`;
  }).join("");
  $$(".result-card").forEach(card => {
    card.addEventListener("click", () => openTrace(card.dataset.trace));
    card.addEventListener("keydown", event => { if (event.key === "Enter") openTrace(card.dataset.trace); });
  });
}

function renderPagination() {
  const node = $("#pagination");
  node.hidden = currentResult.total <= state.limit;
  $("#previousPage").disabled = state.offset === 0;
  $("#nextPage").disabled = state.offset + state.limit >= currentResult.total;
  const start = currentResult.total ? state.offset + 1 : 0;
  const end = Math.min(state.offset + state.limit, currentResult.total);
  $("#pageLabel").textContent = `${fmt.format(start)}–${fmt.format(end)} of ${fmt.format(currentResult.total)}`;
}

async function openTrace(identifier) {
  state.selected = identifier;
  syncUrl();
  $$(".result-card").forEach(card => card.classList.toggle("selected", card.dataset.trace === String(identifier)));
  $("#detailPanel").innerHTML = `<div class="loading-state"><span class="spinner"></span>Opening transcript…</div>`;
  $("#detailPanel").classList.add("mobile-open");
  try {
    const response = await fetch(`/api/trace/${encodeURIComponent(identifier)}`);
    const trace = await response.json();
    if (!response.ok) throw new Error(trace.error || response.statusText);
    selectedTrace = trace;
    state.selected = trace.uid;
    syncUrl(); renderDetail(trace);
  } catch (error) {
    $("#detailPanel").innerHTML = `<div class="no-results"><strong>Could not open trace</strong><p>${esc(error.message)}</p></div>`;
  }
}

function badge(value, extra = "") {
  return `<span class="badge ${esc(slug(value))} ${esc(extra)}">${esc(label(value))}</span>`;
}

function planHtml(title, plan, verdicts = null) {
  const rows = (plan || []).map((item, index) => `<li>${esc(item)}${verdicts?.[index] ? ` <span class="badge ${esc(slug(verdicts[index]))}">${esc(label(verdicts[index]))}</span>` : ""}</li>`).join("");
  return `<div class="plan-card"><h4>${esc(title)}</h4>${rows ? `<ol>${rows}</ol>` : `<span class="microcopy flush">No parseable plan</span>`}</div>`;
}

function sectionHtml(id, title, text, kind, sub = "") {
  const searchContext = kind === "answer" ? "response" : kind;
  const emptyMessage = kind === "reasoning" ? "No reasoning text was recovered." :
    kind === "prompt" ? "No prompt text was recovered." : "No committed answer was recovered.";
  return `<section class="transcript-section">
    <div class="transcript-title"><h3>${esc(title)}</h3><span>${esc(sub)} <button class="copy-mini" type="button" data-copy-section="${id}">Copy</button></span></div>
    <pre class="transcript-text ${esc(kind)} ${text ? "" : "empty"}" id="${id}">${text ? highlight(text, searchContext) : emptyMessage}</pre>
  </section>`;
}

function renderDetail(t) {
  const title = `${t.parent === "graft" ? "Charter graft" : "Public IT"} · ${label(t.source)} step ${t.checkpoint_step}`;
  const tags = (t.tags || []).filter(tag => !/^(source|parent|correctness|outcome|kind|clause-split|presentation):/.test(tag));
  $("#detailPanel").innerHTML = `<article class="trace-detail">
    <header class="detail-head">
      <div class="detail-head-top">
        <div>
          <div class="eyebrow">${esc(t.uid)}</div>
          <h2>${esc(title)}</h2>
          <div class="detail-subtitle">${esc(t.episode_id)}${t.eval_set ? ` · ${esc(t.eval_set)}` : ""}</div>
        </div>
        <div class="detail-actions">
          <button class="button subtle" id="closeDetail" type="button">Close</button>
          <button class="button subtle" id="copyTranscript" type="button">Copy</button>
          <button class="button" id="copyTraceLink" type="button">Link</button>
        </div>
      </div>
      <div class="detail-badges">
        ${badge(t.parent, `parent-${t.parent}`)}${badge(t.correctness)}${badge(t.outcome)}
        ${badge(`${t.clause_split} clause`)}${badge(`${t.presentation} template`)}${badge(t.episode_kind)}
      </div>
    </header>
    <div class="detail-content">
      <div class="score-grid">
        ${scoreCell("Task score", t.semantic_score == null ? "—" : Number(t.semantic_score).toFixed(3))}
        ${scoreCell("Reward", t.reward == null ? "eval / n.a." : Number(t.reward).toFixed(3))}
        ${scoreCell("Native boundary", t.boundary_valid ? "valid" : "invalid")}
        ${scoreCell("Final grammar", t.grammar_valid ? "valid" : "invalid")}
        ${scoreCell("Completion", `${fmt.format(t.completion_tokens || 0)} tokens`)}
        ${scoreCell("Finish", t.truncated ? "truncated" : (t.finish_reason || "unknown"))}
      </div>
      ${sectionHtml("promptText", "Prompt", t.prompt, "prompt", `${fmt.format(t.prompt_tokens || 0)} tokens`)}
      ${sectionHtml("reasoningText", "Model reasoning", t.reasoning, "reasoning", "native thought channel")}
      ${sectionHtml("responseText", "Committed answer", t.response, "answer", t.format_valid ? "format valid" : "format invalid")}
      <div class="oracle-grid">
        ${planHtml("Charter oracle", t.charter_plan)}
        ${planHtml("Coin oracle", t.coin_plan)}
        ${planHtml("Parsed model plan", t.model_plan, t.run_verdicts)}
      </div>
      <div class="field-label">Investigation tags · click to add as a filter</div>
      <div class="tag-cloud">${tags.map(tag => `<button type="button" data-add-tag="${esc(tag)}">${esc(tag)}</button>`).join("") || `<span class="microcopy flush">No derived tags.</span>`}</div>
      <details class="record-details"><summary>Raw native completion</summary><pre>${esc(t.raw_output)}</pre></details>
      ${t.raw_prompt !== t.prompt ? `<details class="record-details"><summary>Raw chat-template prompt</summary><pre>${esc(t.raw_prompt)}</pre></details>` : ""}
      <details class="record-details"><summary>Episode record</summary><pre>${esc(JSON.stringify(t.episode, null, 2))}</pre></details>
      <details class="record-details"><summary>Generation &amp; scorer metadata</summary><pre>${esc(JSON.stringify(t.metadata, null, 2))}</pre></details>
    </div>
  </article>`;
  $("#closeDetail").addEventListener("click", closeDetail);
  $("#copyTranscript").addEventListener("click", copyTranscript);
  $("#copyTraceLink").addEventListener("click", copyViewLink);
  $$(`[data-copy-section]`).forEach(button => button.addEventListener("click", () => {
    const field = { promptText: "prompt", reasoningText: "reasoning", responseText: "response" }[button.dataset.copySection];
    copyText(t[field] || "", `${label(field)} copied`);
  }));
  $$(`[data-add-tag]`).forEach(button => button.addEventListener("click", () => {
    if (!state.filters.tag.includes(button.dataset.addTag)) state.filters.tag.push(button.dataset.addTag);
    renderFacets(); renderActiveFilters(); changed();
  }));
}

function scoreCell(name, value) {
  return `<div class="score-cell"><span>${esc(name)}</span><strong title="${esc(value)}">${esc(value)}</strong></div>`;
}

function emptyDetail() {
  return `<div class="empty-detail"><div class="empty-glyph">⌁</div><h2>Select a trace</h2><p>Choose a result to inspect its prompt, hidden reasoning, committed answer, scorer verdicts, and exact oracle plans.</p><div class="keyboard-hint"><kbd>j</kbd><kbd>k</kbd> move · <kbd>enter</kbd> open</div></div>`;
}

function closeDetail() {
  state.selected = null; selectedTrace = null; syncUrl();
  $("#detailPanel").classList.remove("mobile-open");
  $("#detailPanel").innerHTML = emptyDetail();
  $$(".result-card").forEach(card => card.classList.remove("selected"));
}

function transcriptText(t) {
  return [
    `TRACE: ${t.uid}`, `MODEL: ${label(t.parent)} · ${label(t.source)} step ${t.checkpoint_step}`,
    `EPISODE: ${t.episode_id}`, `SCORE: ${label(t.correctness)} · ${label(t.outcome)}`,
    "", "=== PROMPT ===", t.prompt || "", "", "=== REASONING ===", t.reasoning || "",
    "", "=== COMMITTED ANSWER ===", t.response || "[none]", "",
    "=== ORACLES ===", `Charter: ${(t.charter_plan || []).join(" | ")}`,
    `Coin: ${(t.coin_plan || []).join(" | ")}`, `Model: ${(t.model_plan || []).join(" | ")}`,
  ].join("\n");
}

function copyTranscript() {
  if (selectedTrace) copyText(transcriptText(selectedTrace), "Transcript copied");
}

async function copyText(text, message) {
  await navigator.clipboard.writeText(text);
  showToast(message);
}

function copyViewLink() {
  syncUrl();
  copyText(location.href, "View link copied");
}

function exportView(type) {
  const params = queryParams({ includePaging: false });
  window.location.href = `/api/export.${type}?${params.toString()}`;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message; toast.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove("show"), 1700);
}

function keyboardNavigation(event) {
  const editing = /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName);
  if (event.key === "/" && !editing) { event.preventDefault(); $("#searchInput").focus(); return; }
  if (event.key === "Escape") {
    if ($("#shortcutsDialog").open) return $("#shortcutsDialog").close();
    if (editing && state.q) { state.q = ""; $("#searchInput").value = ""; changed(); }
    else closeDetail();
    return;
  }
  if (editing) return;
  if (event.key.toLocaleLowerCase() === "c" && selectedTrace) { copyTranscript(); return; }
  if (!["j", "k", "Enter"].includes(event.key)) return;
  const cards = $$(".result-card");
  if (!cards.length) return;
  let index = cards.findIndex(card => card === document.activeElement || card.classList.contains("selected"));
  if (event.key === "j") index = Math.min(cards.length - 1, index + 1);
  else if (event.key === "k") index = Math.max(0, index < 0 ? 0 : index - 1);
  else if (event.key === "Enter" && index >= 0) return openTrace(cards[index].dataset.trace);
  else return;
  cards[index].focus(); cards[index].scrollIntoView({ block: "nearest" });
}

async function init() {
  readUrlState();
  bindControls();
  try {
    const response = await fetch("/api/meta");
    meta = await response.json();
    if (!response.ok) throw new Error(meta.error || response.statusText);
    renderCorpusFacts(); renderFacets(); restoreControls(); renderActiveFilters();
    await search();
    if (state.selected) await openTrace(state.selected);
  } catch (error) {
    $("#resultsList").innerHTML = `<div class="no-results"><strong>Browser could not start</strong><p>${esc(error.message)}</p></div>`;
    $("#statusDot").title = error.message;
  }
}

init();
