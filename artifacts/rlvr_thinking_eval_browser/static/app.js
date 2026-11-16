"use strict";

const MULTI = ["arm", "step", "run_count", "outcome", "verdict", "run_pattern", "episode_kind", "surface", "target_clause", "clause_family", "template_id"];
const SINGLE = ["truncated", "parser_valid", "format_valid", "boundary_valid", "tokens_min", "tokens_max"];
const LABELS = {
  charter:"Charter", coin:"Coin", control:"Control", correct:"Correct / shared",
  mixed:"Mixed C + C", other:"Other", malformed:"Malformed", agreement:"Agreement",
  conflict:"Conflict", canonical:"Canonical", trained:"Trained templates", heldout:"Held-out templates",
  valid_single_policy:"Valid · one policy", valid_mixed:"Valid · mixed policy",
};
const FIELD_LABELS = {
  arm:"Arm", step:"Step", run_count:"Runs", outcome:"Outcome", verdict:"Contains verdict",
  run_pattern:"Verdict pattern", episode_kind:"Kind", surface:"Surface", target_clause:"Clause",
  clause_family:"Clause family", template_id:"Template", truncated:"Truncated",
  parser_valid:"Parser valid", format_valid:"Format valid", boundary_valid:"Boundary valid",
  tokens_min:"Min tokens", tokens_max:"Max tokens",
};
const $ = (selector, root=document) => root.querySelector(selector);
const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
const fmt = new Intl.NumberFormat("en-GB");
const pct = value => value == null ? "—" : `${(Number(value)*100).toFixed(1)}%`;
const esc = value => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[c]);
const title = value => LABELS[value] || String(value ?? "—").replaceAll("_"," ").replaceAll("+"," + ").replace(/\b\w/g,c=>c.toUpperCase());
const slug = value => String(value || "").replaceAll("_","-").replaceAll("+","-");

let meta = null;
let state = {q:"",scope:"reasoning",sort:"step_asc",offset:0,limit:50,selected:null,filters:{}};
let result = {items:[],total:0};
let controller = null;
let debounce = null;
let toastTimer = null;

function emptyFilters(){return Object.fromEntries([...MULTI,...SINGLE].map(f=>[f,MULTI.includes(f)?[]:""]));}
state.filters = emptyFilters();

function readUrl(){
  const p=new URLSearchParams(location.search);
  state.q=p.get("q")||""; state.scope=p.get("scope")||"reasoning"; state.sort=p.get("sort")||"step_asc";
  state.offset=Math.max(0,Number(p.get("offset"))||0); state.selected=p.get("trace");
  MULTI.forEach(f=>state.filters[f]=p.getAll(f)); SINGLE.forEach(f=>state.filters[f]=p.get(f)||"");
}
function params({paging=true,trace=false}={}){
  const p=new URLSearchParams(); if(state.q)p.set("q",state.q); if(state.scope!=="reasoning")p.set("scope",state.scope); if(state.sort!=="step_asc")p.set("sort",state.sort);
  MULTI.forEach(f=>state.filters[f].forEach(v=>p.append(f,v))); SINGLE.forEach(f=>{if(state.filters[f]!=="")p.set(f,state.filters[f]);});
  if(paging){p.set("limit",state.limit);if(state.offset)p.set("offset",state.offset);} if(trace&&state.selected)p.set("trace",state.selected); return p;
}
function syncUrl(){const p=params({trace:true});p.delete("limit");history.replaceState(null,"",p.size?`?${p}`:location.pathname);}
function facet(field){return meta?.facets?.[field]||[];}

function renderFacts(){
  const b=meta.build;
  $("#facts").innerHTML=`<div class="fact"><strong>${fmt.format(b.trace_count)}</strong><span>Eval traces</span></div><div class="fact"><strong>${fmt.format(b.episode_count)}</strong><span>Episodes</span></div><div class="fact"><strong>12</strong><span>Endpoints</span></div><div class="fact"><strong>3 × 4</strong><span>Arms × steps</span></div>`;
  $("#statusDot").classList.add("live");
}
function option(field,item,segment=false){
  const value=String(item.value), checked=state.filters[field].includes(value);
  return `<label title="${esc(title(value))} · ${fmt.format(item.count)} traces"><input type="checkbox" data-filter="${field}" value="${esc(value)}" ${checked?"checked":""}><span class="option-text">${esc(title(value))}</span>${segment?"":`<small>${fmt.format(item.count)}</small>`}</label>`;
}
function renderField(field, needle=""){
  const box=$(`[data-field="${field}"]`); if(!box)return;
  let items=facet(field); if(needle)items=items.filter(x=>title(x.value).toLowerCase().includes(needle.toLowerCase()));
  const segment=box.classList.contains("segments"); box.innerHTML=items.map(x=>option(field,x,segment)).join("")||`<span class="micro">No matches</span>`;
}
function renderFacets(){MULTI.forEach(f=>renderField(f));}
function restore(){
  $("#search").value=state.q; $("#sort").value=state.sort; $$('[data-scope]').forEach(b=>b.classList.toggle("active",b.dataset.scope===state.scope));
  $$('[data-single]').forEach(x=>x.value=state.filters[x.dataset.single]); $$('[data-number]').forEach(x=>x.value=state.filters[x.dataset.number]);
}
function renderActive(){
  const values=[]; if(state.q)values.push({field:"q",value:`Search: ${state.q}`});
  MULTI.forEach(f=>state.filters[f].forEach(v=>values.push({field:f,value:v}))); SINGLE.forEach(f=>{if(state.filters[f]!=="")values.push({field:f,value:state.filters[f]});});
  const box=$("#activeFilters");box.hidden=!values.length;box.innerHTML=values.map(x=>`<button class="active-pill" data-remove="${x.field}" data-value="${esc(x.value)}">${x.field==="q"?esc(x.value):`${FIELD_LABELS[x.field]}: ${esc(title(x.value))}`}</button>`).join("");
}
function changed(delayed=false){state.offset=0;state.selected=null;renderActive();syncUrl();clearTimeout(debounce);debounce=setTimeout(search,delayed?280:0);}

function bind(){
  document.addEventListener("change",event=>{
    const cb=event.target.closest("[data-filter]"); if(cb){const f=cb.dataset.filter,set=new Set(state.filters[f]);cb.checked?set.add(cb.value):set.delete(cb.value);state.filters[f]=[...set];changed();return;}
    const single=event.target.closest("[data-single]");if(single){state.filters[single.dataset.single]=single.value;changed();return;}
    const number=event.target.closest("[data-number]");if(number){state.filters[number.dataset.number]=number.value;changed();}
  });
  $("#search").addEventListener("input",e=>{state.q=e.target.value.trim();changed(true);});
  $("#sort").addEventListener("change",e=>{state.sort=e.target.value;changed();});
  $("#scopes").addEventListener("click",e=>{const b=e.target.closest("[data-scope]");if(!b)return;state.scope=b.dataset.scope;$$('[data-scope]').forEach(x=>x.classList.toggle("active",x===b));changed();});
  $("#reset").addEventListener("click",()=>{state={...state,q:"",scope:"reasoning",sort:"step_asc",offset:0,selected:null,filters:emptyFilters()};renderFacets();restore();changed();});
  $("#activeFilters").addEventListener("click",e=>{const b=e.target.closest("[data-remove]");if(!b)return;if(b.dataset.remove==="q"){state.q="";$("#search").value="";}else if(MULTI.includes(b.dataset.remove)){state.filters[b.dataset.remove]=state.filters[b.dataset.remove].filter(v=>v!==b.dataset.value);renderField(b.dataset.remove);}else state.filters[b.dataset.remove]="";changed();});
  $$('[data-list-search]').forEach(input=>input.addEventListener("input",e=>renderField(e.target.dataset.listSearch,e.target.value)));
  $$('.collapse-button').forEach(button=>button.addEventListener("click",()=>{const open=button.getAttribute("aria-expanded")==="true";button.setAttribute("aria-expanded",String(!open));button.nextElementSibling.hidden=open;}));
  $("#previous").addEventListener("click",()=>{state.offset=Math.max(0,state.offset-state.limit);syncUrl();search();});
  $("#next").addEventListener("click",()=>{state.offset+=state.limit;syncUrl();search();});
  $("#cards").addEventListener("click",e=>{const card=e.target.closest("[data-trace]");if(card)openTrace(card.dataset.trace);});
  $("#cards").addEventListener("keydown",e=>{if(e.key==="Enter"){const card=e.target.closest("[data-trace]");if(card)openTrace(card.dataset.trace);}});
  $("#copyView").addEventListener("click",()=>navigator.clipboard.writeText(location.href).then(()=>toast("View link copied")));
  $("#exportJsonl").addEventListener("click",()=>location.href=`/api/export.jsonl?${params({paging:false})}`);
  $("#exportCsv").addEventListener("click",()=>location.href=`/api/export.csv?${params({paging:false})}`);
  $("#randomButton").addEventListener("click",async()=>{state.sort="random";state.offset=0;$("#sort").value="random";await search();if(result.items[0])openTrace(result.items[0].uid);});
  document.addEventListener("keydown",keyboard);
}

function renderSummary(data){
  const a=data.aggregates, one=(data.summary.run_count||[]).find(x=>Number(x.value)===1)?.count||0,two=(data.summary.run_count||[]).find(x=>Number(x.value)===2)?.count||0;
  $("#summary").innerHTML=`<div class="summary-stat"><strong>${pct(a.truncation_rate)}</strong><span>Truncated</span></div><div class="summary-stat"><strong>${pct(a.parser_valid_rate)}</strong><span>Parser valid</span></div><div class="summary-stat"><strong>${Math.round(a.mean_tokens||0)}</strong><span>Mean tokens</span></div><div class="summary-stat"><strong>${fmt.format(one)}</strong><span>One-run</span></div><div class="summary-stat"><strong>${fmt.format(two)}</strong><span>Two-run</span></div>`;
}
function renderCards(data){
  $("#count").textContent=fmt.format(data.total);renderSummary(data);
  const box=$("#cards");
  if(!data.items.length){box.innerHTML=`<div class="no-results"><strong>No matching traces</strong><span>Try removing a filter or using fewer search terms.</span></div>`;}
  else box.innerHTML=data.items.map((x,i)=>`<article class="card ${state.selected===x.uid?"selected":""}" tabindex="0" data-trace="${x.uid}"><div class="card-top"><div class="badges"><span class="badge arm-${x.arm}">${title(x.arm)}</span><span class="badge">step ${x.checkpoint_step}</span><span class="badge ${slug(x.outcome)}">${title(x.outcome)}</span><span class="badge">${x.run_count} run${x.run_count===1?"":"s"}</span>${x.truncated?'<span class="badge malformed">truncated</span>':""}</div><span>${state.offset+i+1}</span></div><h3>${esc(x.episode_id)}</h3><p class="preview ${x.reasoning_preview?"":"empty"}">${esc(x.reasoning_preview||"No reasoning text")}</p><div class="card-bottom"><span>${title(x.surface)} · ${esc(x.template_id)}</span><span class="metrics"><span>${fmt.format(x.completion_tokens)} tok</span><span>${esc(x.run_pattern)}</span></span></div></article>`).join("");
  const pages=Math.max(1,Math.ceil(data.total/state.limit)),current=Math.floor(state.offset/state.limit)+1;$("#pagination").hidden=data.total<=state.limit;$("#page").textContent=`Page ${current} of ${fmt.format(pages)}`;$("#previous").disabled=state.offset===0;$("#next").disabled=state.offset+state.limit>=data.total;
}
async function search(){
  controller?.abort();controller=new AbortController();$("#cards").classList.add("busy");
  try{const response=await fetch(`/api/search?${params()}`,{signal:controller.signal});const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);result=data;renderCards(data);}
  catch(error){if(error.name!=="AbortError")$("#cards").innerHTML=`<div class="no-results"><strong>Search failed</strong><span>${esc(error.message)}</span></div>`;}
  finally{$("#cards").classList.remove("busy");}
}
function planHtml(name,plan,runs){return `<div class="plan"><h4>${name}</h4><ol>${(plan||[]).map((crew,i)=>`<li>${esc(runs[i]?.run_id||i+1)} = ${esc(crew)}</li>`).join("")||"<li>None</li>"}</ol></div>`;}
function section(name,text,cls=""){return `<section class="transcript"><div class="transcript-title"><h3>${name}</h3><button data-copy-text="${encodeURIComponent(text||"")}">Copy</button></div><pre class="${cls} ${text?"":"empty"}">${esc(text||"No committed answer")}</pre></section>`;}
function renderDetail(x){
  const runs=x.episode.runs||[];
  const runRows=x.run_details.map(r=>`<tr><td>${esc(r.run_id)}</td><td>${esc(r.port||"—")}</td><td><span class="badge ${slug(r.verdict)}">${title(r.verdict==="shared"?"correct":r.verdict)}</span></td><td>${esc(r.model||"—")}</td><td>${esc(r.charter)}</td><td>${esc(r.coin)}</td></tr>`).join("");
  $("#detail").innerHTML=`<article class="trace-detail"><header class="detail-head"><div class="detail-head-row"><div><div class="eyebrow dark">TRACE ${x.id}</div><h2>${esc(x.episode_id)}</h2><div class="detail-sub">${title(x.arm)} · step ${x.checkpoint_step} · ${title(x.surface)} · ${esc(x.template_id)}</div></div><button class="button ghost" id="copyTrace">Copy trace</button></div><div class="detail-badges"><span class="badge arm-${x.arm}">${title(x.arm)}</span><span class="badge ${slug(x.outcome)}">${title(x.outcome)}</span><span class="badge">${x.run_count} run${x.run_count===1?"":"s"}</span><span class="badge">${esc(x.run_pattern)}</span>${x.truncated?'<span class="badge malformed">truncated</span>':""}</div></header><div class="detail-content"><div class="score-grid"><div class="score-cell"><span>Outcome</span><strong>${title(x.outcome)}</strong></div><div class="score-cell"><span>Quality</span><strong>${title(x.answer_quality)}</strong></div><div class="score-cell"><span>Tokens</span><strong>${fmt.format(x.completion_tokens)}</strong></div><div class="score-cell"><span>Finish</span><strong>${esc(x.finish_reason||"—")}</strong></div><div class="score-cell"><span>Parser</span><strong>${x.parser_valid?"valid":"invalid"}</strong></div><div class="score-cell"><span>Thought closed</span><strong>${x.channel_close_count===1?"yes":"no"}</strong></div></div><table class="run-table"><thead><tr><th>Run</th><th>Port</th><th>Verdict</th><th>Model</th><th>Charter oracle</th><th>Coin oracle</th></tr></thead><tbody>${runRows}</tbody></table>${section("Prompt",x.prompt)}${section("Hidden reasoning",x.reasoning,"reasoning")}${section("Committed final answer",x.final_answer,"answer")}<div class="oracle-grid">${planHtml("Charter oracle",x.charter_plan,runs)}${planHtml("Coin oracle",x.coin_plan,runs)}${planHtml("Parsed model plan",x.parsed_plan,runs)}</div><details><summary>Raw episode record</summary><pre>${esc(JSON.stringify(x.episode,null,2))}</pre></details><details><summary>Generation and scorer metadata</summary><pre>${esc(JSON.stringify(x.metadata,null,2))}</pre></details></div></article>`;
  $("#detail").classList.add("open");$("#copyTrace").addEventListener("click",()=>copyTrace(x));$$('[data-copy-text]',$("#detail")).forEach(b=>b.addEventListener("click",()=>navigator.clipboard.writeText(decodeURIComponent(b.dataset.copyText)).then(()=>toast("Copied"))));
}
async function openTrace(uid){
  state.selected=uid;syncUrl();$$('.card').forEach(c=>c.classList.toggle("selected",c.dataset.trace===uid));$("#detail").innerHTML=`<div class="loading"><span></span>Loading full trace…</div>`;
  try{const response=await fetch(`/api/trace/${encodeURIComponent(uid)}`),data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);renderDetail(data);}
  catch(error){$("#detail").innerHTML=`<div class="no-results"><strong>Trace failed to load</strong><span>${esc(error.message)}</span></div>`;}
}
function copyTrace(x){const text=[`Episode: ${x.episode_id}`,`Arm: ${x.arm} · step ${x.checkpoint_step} · ${x.surface}`,"", "PROMPT",x.prompt,"","REASONING",x.reasoning,"","FINAL",x.final_answer||"[none]","","VERDICTS",JSON.stringify(x.run_details,null,2)].join("\n");navigator.clipboard.writeText(text).then(()=>toast("Full trace copied"));}
function toast(message){const box=$("#toast");box.textContent=message;box.classList.add("show");clearTimeout(toastTimer);toastTimer=setTimeout(()=>box.classList.remove("show"),1800);}
function keyboard(event){
  if(event.key==="/"&&!/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)){event.preventDefault();$("#search").focus();return;}
  if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName))return;
  const cards=$$('.card'),current=cards.findIndex(c=>c.dataset.trace===state.selected);let index=current;
  if(event.key==="j")index=Math.min(cards.length-1,current<0?0:current+1);else if(event.key==="k")index=Math.max(0,current<0?0:current-1);else if(event.key==="Enter"&&current>=0)return openTrace(cards[current].dataset.trace);else return;
  if(cards[index]){cards[index].focus();cards[index].scrollIntoView({block:"nearest"});}
}
async function init(){
  readUrl();bind();
  try{const response=await fetch("/api/meta"),data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);meta=data;renderFacts();renderFacets();restore();renderActive();await search();if(state.selected)await openTrace(state.selected);}
  catch(error){$("#cards").innerHTML=`<div class="no-results"><strong>Browser could not start</strong><span>${esc(error.message)}</span></div>`;}
}
init();
