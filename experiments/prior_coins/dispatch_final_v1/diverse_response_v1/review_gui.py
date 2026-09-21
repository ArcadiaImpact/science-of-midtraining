"""Small, dependency-free browser for reviewing rendered treatment episodes."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit


DEFAULT_DATA = Path(__file__).resolve().parent / "samples" / "episodes.jsonl"

FILTER_FIELDS = (
    ("actual_outcome", "Actual outcome"),
    ("response_policy", "Response policy"),
    ("response_mode", "Resolved response mode"),
    ("motivation_direction", "Motivation direction"),
    ("motivation_relation", "Motivation relation"),
    ("source_cell", "Source cell"),
    ("overlay_register", "Overlay register"),
    ("overlay_position", "Overlay position"),
    ("overlay_template_id", "Overlay template ID"),
    ("prompt_template_id", "Prompt template ID"),
    ("natural_response_variant_id", "Natural response variant"),
)


class DataLoadError(ValueError):
    """Raised when a review JSONL file cannot be loaded safely."""


def _validate_row(row: Any, *, path: Path, line_number: int) -> dict[str, Any]:
    location = f"{path}: line {line_number}"
    if not isinstance(row, dict):
        raise DataLoadError(f"{location}: each JSONL row must be a JSON object")
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise DataLoadError(f"{location}: 'messages' must be a list")
    if not all(isinstance(message, dict) for message in messages):
        raise DataLoadError(f"{location}: every message must be a JSON object")
    for role in ("user", "assistant"):
        matching = [message for message in messages if message.get("role") == role]
        if len(matching) != 1 or not isinstance(matching[0].get("content"), str):
            raise DataLoadError(
                f"{location}: expected exactly one {role!r} message with string content"
            )
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise DataLoadError(f"{location}: 'metadata' must be a JSON object")
    treatment = metadata.get("response_treatment")
    if not isinstance(treatment, dict):
        raise DataLoadError(
            f"{location}: metadata.response_treatment must be a JSON object"
        )
    return row


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate review rows, reporting file and line on bad input."""
    resolved = Path(path).expanduser()
    rows: list[dict[str, Any]] = []
    try:
        with resolved.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    parsed = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise DataLoadError(
                        f"{resolved}: line {line_number}: invalid JSON: {exc.msg} "
                        f"(column {exc.colno})"
                    ) from exc
                rows.append(
                    _validate_row(parsed, path=resolved, line_number=line_number)
                )
    except FileNotFoundError as exc:
        raise DataLoadError(f"Data file not found: {resolved}") from exc
    except UnicodeDecodeError as exc:
        raise DataLoadError(f"Data file is not valid UTF-8: {resolved}: {exc}") from exc
    except OSError as exc:
        raise DataLoadError(f"Could not read data file {resolved}: {exc}") from exc
    if not rows:
        raise DataLoadError(f"Data file contains no JSON rows: {resolved}")
    return rows


def _filter_value(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def extract_filter_fields(row: dict[str, Any]) -> dict[str, str]:
    """Extract normalized filter values from one rendered episode row."""
    metadata = row.get("metadata") or {}
    treatment = metadata.get("response_treatment") or {}
    values = {
        "actual_outcome": treatment.get("actual_outcome"),
        "response_policy": treatment.get("response_policy"),
        "response_mode": treatment.get("response_mode"),
        "motivation_direction": treatment.get("motivation_direction"),
        "motivation_relation": treatment.get("motivation_relation"),
        "source_cell": treatment.get("source_cell"),
        "overlay_register": treatment.get("overlay_register"),
        "overlay_position": treatment.get("overlay_position"),
        "overlay_template_id": treatment.get("overlay_template_id"),
        "prompt_template_id": metadata.get("template_id"),
        "natural_response_variant_id": treatment.get(
            "natural_response_variant_id"
        ),
    }
    return {key: _filter_value(value) for key, value in values.items()}


def extract_filter_options(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Return sorted, data-derived option values for every review filter."""
    field_names = [name for name, _label in FILTER_FIELDS]
    observed = {name: set() for name in field_names}
    for row in rows:
        values = extract_filter_fields(row)
        for name in field_names:
            observed[name].add(values[name])
    return {
        name: sorted(values, key=lambda value: (value != "(none)", value.casefold()))
        for name, values in observed.items()
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dispatch treatment review</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17211d;
      --muted: #637069;
      --paper: #fbfcf8;
      --panel: #ffffff;
      --line: #d9e0da;
      --accent: #176b58;
      --accent-soft: #e2f1eb;
      --user: #f2f5f8;
      --assistant: #eef7f2;
      --shadow: 0 10px 28px rgba(25, 47, 38, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--paper);
      color: var(--ink);
      font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }
    header {
      padding: 22px clamp(18px, 4vw, 54px) 18px;
      background: #153d34;
      color: white;
    }
    header h1 { margin: 0; font-size: clamp(22px, 3vw, 32px); letter-spacing: -0.02em; }
    header p { margin: 5px 0 0; color: #c9ddd5; }
    main { max-width: 1600px; margin: 0 auto; padding: 20px clamp(14px, 3vw, 42px) 44px; }
    .controls, .metadata, .message-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .controls { padding: 16px; margin-bottom: 18px; }
    .search-row {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto;
      gap: 10px;
      align-items: end;
    }
    label { display: block; color: #35443d; font-size: 12px; font-weight: 700; letter-spacing: .025em; }
    input, select, button {
      width: 100%;
      margin-top: 5px;
      border: 1px solid #b9c5be;
      border-radius: 7px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    input, select { min-height: 39px; padding: 7px 9px; }
    input:focus, select:focus, button:focus-visible {
      outline: 3px solid rgba(23, 107, 88, .2);
      border-color: var(--accent);
    }
    button {
      width: auto;
      min-height: 39px;
      padding: 7px 14px;
      cursor: pointer;
      font-weight: 700;
    }
    button:hover:not(:disabled) { background: var(--accent-soft); border-color: var(--accent); }
    button:disabled { cursor: default; opacity: .42; }
    .filters {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px 12px;
      margin-top: 14px;
    }
    .status-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin: 14px 2px 0;
      color: var(--muted);
    }
    #counts { font-weight: 700; color: #34433d; }
    .nav { display: flex; gap: 7px; }
    .nav button { min-width: 94px; }
    #empty, #fatal {
      padding: 42px 20px;
      border: 1px dashed #aebbb4;
      border-radius: 12px;
      text-align: center;
      color: var(--muted);
      background: white;
    }
    #fatal { color: #8d2929; border-color: #d6a6a6; }
    .hidden { display: none !important; }
    .metadata { margin-bottom: 18px; overflow: hidden; }
    .section-heading {
      margin: 0;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #f4f7f4;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .metadata-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1px;
      background: var(--line);
    }
    .datum { min-width: 0; padding: 10px 12px; background: white; }
    .datum-key { display: block; color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .datum-value { display: block; overflow-wrap: anywhere; margin-top: 2px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
    details { border-top: 1px solid var(--line); }
    summary { cursor: pointer; padding: 11px 15px; color: var(--accent); font-weight: 700; }
    details pre { margin: 0; border-top: 1px solid var(--line); border-radius: 0; }
    .messages { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
    .message-card { min-width: 0; overflow: hidden; }
    .message-card.user .section-heading { background: var(--user); }
    .message-card.assistant .section-heading { background: var(--assistant); }
    pre {
      margin: 0;
      padding: 17px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      tab-size: 2;
      color: #16201c;
      font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .kbd { color: var(--muted); font-size: 12px; }
    @media (max-width: 850px) {
      .messages { grid-template-columns: 1fr; }
      .search-row { grid-template-columns: 1fr auto auto; }
    }
    @media (max-width: 560px) {
      .search-row { grid-template-columns: 1fr 1fr; }
      .search-row label { grid-column: 1 / -1; }
      .status-row { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Dispatch treatment review</h1>
    <p>Inspect the complete prompt, response, and treatment metadata for generated episodes.</p>
  </header>
  <main>
    <section class="controls" aria-label="Episode filters">
      <div class="search-row">
        <label>Free text or episode ID
          <input id="search" type="search" placeholder="Search prompts, responses, IDs, or metadata…" autocomplete="off">
        </label>
        <button id="clear" type="button">Clear filters</button>
        <button id="random" type="button">Random row</button>
      </div>
      <div id="filters" class="filters"></div>
      <div class="status-row">
        <div>
          <span id="counts">Loading episodes…</span>
          <span class="kbd"> · ←/→ treatment · ↑/↓ prompt format</span>
        </div>
        <div class="nav">
          <button id="previous" type="button">← Treatment</button>
          <button id="next" type="button">Treatment →</button>
        </div>
      </div>
    </section>

    <div id="fatal" class="hidden" role="alert"></div>
    <div id="empty" class="hidden">No episodes match the current filters.</div>
    <div id="viewer" class="hidden">
      <section class="metadata">
        <h2 class="section-heading">Selected row metadata</h2>
        <div id="metadata-summary" class="metadata-grid"></div>
        <details>
          <summary>Complete metadata JSON</summary>
          <pre id="metadata-json"></pre>
        </details>
      </section>
      <div class="messages">
        <section class="message-card user">
          <h2 class="section-heading">Complete user prompt</h2>
          <pre id="user-message"></pre>
        </section>
        <section class="message-card assistant">
          <h2 class="section-heading">Complete assistant response</h2>
          <pre id="assistant-message"></pre>
        </section>
      </div>
    </div>
  </main>
  <script>
    "use strict";
    let payload = null;
    let visible = [];
    let cursor = 0;
    const selects = new Map();

    const search = document.getElementById("search");
    const filterContainer = document.getElementById("filters");
    const counts = document.getElementById("counts");
    const viewer = document.getElementById("viewer");
    const empty = document.getElementById("empty");
    const fatal = document.getElementById("fatal");
    const previous = document.getElementById("previous");
    const next = document.getElementById("next");

    function option(select, value, label) {
      const element = document.createElement("option");
      element.value = value;
      element.textContent = label;
      select.appendChild(element);
    }

    function buildFilters() {
      for (const definition of payload.filter_definitions) {
        const label = document.createElement("label");
        label.textContent = definition.label;
        const select = document.createElement("select");
        select.setAttribute("aria-label", definition.label);
        option(select, "", `All (${payload.filter_options[definition.name].length} values)`);
        for (const value of payload.filter_options[definition.name]) {
          option(select, value, value);
        }
        select.addEventListener("change", applyFilters);
        label.appendChild(select);
        filterContainer.appendChild(label);
        selects.set(definition.name, select);
      }
    }

    function message(record, role) {
      const found = record.messages.find(item => item.role === role);
      return found ? found.content : "";
    }

    function groupKey(item) {
      const metadata = item.record.metadata;
      return metadata.sample_group_id || metadata.sample_id || metadata.episode_id;
    }

    function promptOrder(item) {
      const value = item.record.metadata.sample_prompt_index;
      return Number.isInteger(value) ? value : 0;
    }

    function navigation() {
      const groups = [];
      const byGroup = new Map();
      for (const item of visible) {
        const key = groupKey(item);
        if (!byGroup.has(key)) {
          groups.push(key);
          byGroup.set(key, []);
        }
        byGroup.get(key).push(item);
      }
      for (const items of byGroup.values()) {
        items.sort((left, right) =>
          promptOrder(left) - promptOrder(right)
          || left.filters.prompt_template_id.localeCompare(right.filters.prompt_template_id)
        );
      }
      if (!visible.length) return {groups, byGroup, groupIndex: -1, promptIndex: -1};
      const current = visible[cursor];
      const groupIndex = groups.indexOf(groupKey(current));
      const promptIndex = byGroup.get(groups[groupIndex]).indexOf(current);
      return {groups, byGroup, groupIndex, promptIndex};
    }

    function applyFilters() {
      const query = search.value.trim().toLocaleLowerCase();
      visible = payload.items.filter(item => {
        for (const [name, select] of selects) {
          if (select.value && item.filters[name] !== select.value) return false;
        }
        if (!query) return true;
        return item.search_text.includes(query);
      });
      cursor = 0;
      render();
    }

    function datum(key, value) {
      const cell = document.createElement("div");
      cell.className = "datum";
      const name = document.createElement("span");
      name.className = "datum-key";
      name.textContent = key;
      const content = document.createElement("span");
      content.className = "datum-value";
      content.textContent = value == null ? "(none)" : String(value);
      cell.append(name, content);
      return cell;
    }

    function render() {
      const total = payload ? payload.items.length : 0;
      const hasRows = visible.length > 0;
      viewer.classList.toggle("hidden", !hasRows);
      empty.classList.toggle("hidden", hasRows);
      const navState = navigation();
      previous.disabled = !hasRows || navState.groupIndex === 0;
      next.disabled = !hasRows || navState.groupIndex >= navState.groups.length - 1;
      document.getElementById("random").disabled = !hasRows;
      counts.textContent = hasRows
        ? `Showing ${visible.length} of ${total} rows · Treatment ${navState.groupIndex + 1} of ${navState.groups.length} · Prompt format ${navState.promptIndex + 1} of ${navState.byGroup.get(navState.groups[navState.groupIndex]).length}`
        : `Showing 0 of ${total} rows`;
      if (!hasRows) return;

      const item = visible[cursor];
      const record = item.record;
      const metadata = record.metadata;
      const summary = document.getElementById("metadata-summary");
      summary.replaceChildren(
        datum("Episode ID", metadata.episode_id),
        datum("Sample ID", metadata.sample_id),
        datum("Actual outcome", item.filters.actual_outcome),
        datum("Response policy", item.filters.response_policy),
        datum("Resolved mode", item.filters.response_mode),
        datum("Motivation", item.filters.motivation_relation),
        datum("Source cell", item.filters.source_cell),
        datum("Prompt template", item.filters.prompt_template_id),
        datum("Prompt family", metadata.sample_prompt_family),
        datum("Prompt register", metadata.sample_prompt_register),
        datum("Prompt description", metadata.sample_prompt_description),
        datum("Natural variant", item.filters.natural_response_variant_id),
        datum("Overlay", item.filters.overlay_template_id),
        datum("Overlay register", item.filters.overlay_register),
        datum("Overlay position", item.filters.overlay_position)
      );
      document.getElementById("metadata-json").textContent = JSON.stringify(metadata, null, 2);
      document.getElementById("user-message").textContent = message(record, "user");
      document.getElementById("assistant-message").textContent = message(record, "assistant");
    }

    function selectItem(item) {
      const nextCursor = visible.indexOf(item);
      if (nextCursor < 0) return;
      cursor = nextCursor;
      render();
      window.scrollTo({top: 0, behavior: "smooth"});
    }

    function moveTreatment(amount) {
      const state = navigation();
      const targetIndex = state.groupIndex + amount;
      if (targetIndex < 0 || targetIndex >= state.groups.length) return;
      const currentTemplate = visible[cursor].filters.prompt_template_id;
      const candidates = state.byGroup.get(state.groups[targetIndex]);
      const sameTemplate = candidates.find(
        item => item.filters.prompt_template_id === currentTemplate
      );
      selectItem(sameTemplate || candidates[Math.min(state.promptIndex, candidates.length - 1)]);
    }

    function movePrompt(amount) {
      const state = navigation();
      if (state.groupIndex < 0) return;
      const candidates = state.byGroup.get(state.groups[state.groupIndex]);
      if (candidates.length < 2) return;
      const targetIndex = (state.promptIndex + amount + candidates.length) % candidates.length;
      selectItem(candidates[targetIndex]);
    }

    search.addEventListener("input", applyFilters);
    previous.addEventListener("click", () => moveTreatment(-1));
    next.addEventListener("click", () => moveTreatment(1));
    document.getElementById("clear").addEventListener("click", () => {
      search.value = "";
      for (const select of selects.values()) select.value = "";
      applyFilters();
      search.focus();
    });
    document.getElementById("random").addEventListener("click", () => {
      if (visible.length) {
        cursor = Math.floor(Math.random() * visible.length);
        render();
      }
    });
    document.addEventListener("keydown", event => {
      if (event.target.matches("input, select, button, summary")) return;
      if (event.key === "ArrowLeft") moveTreatment(-1);
      if (event.key === "ArrowRight") moveTreatment(1);
      if (event.key === "ArrowUp") {
        event.preventDefault();
        movePrompt(-1);
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        movePrompt(1);
      }
    });

    fetch("/data.json", {cache: "no-store"})
      .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then(data => {
        payload = data;
        buildFilters();
        visible = payload.items;
        render();
      })
      .catch(error => {
        counts.textContent = "Could not load episodes";
        fatal.textContent = `The episode data could not be loaded: ${error.message}`;
        fatal.classList.remove("hidden");
      });
  </script>
</body>
</html>
"""


def _search_text(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True).casefold()


def make_handler(rows: Sequence[dict[str, Any]]) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler bound to an immutable in-memory review payload."""
    payload = {
        "filter_definitions": [
            {"name": name, "label": label} for name, label in FILTER_FIELDS
        ],
        "filter_options": extract_filter_options(rows),
        "items": [
            {
                "record": row,
                "filters": extract_filter_fields(row),
                "search_text": _search_text(row),
            }
            for row in rows
        ],
    }
    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    html_bytes = INDEX_HTML.encode("utf-8")

    class ReviewHandler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: HTTPStatus) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path
            if route in ("/", "/index.html"):
                self._send(
                    html_bytes,
                    "text/html; charset=utf-8",
                    HTTPStatus.OK,
                )
            elif route == "/data.json":
                self._send(
                    data_bytes,
                    "application/json; charset=utf-8",
                    HTTPStatus.OK,
                )
            elif route == "/favicon.ico":
                self._send(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
            else:
                self._send(b"Not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

    return ReviewHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a local browser for rendered dispatch treatment episodes."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"episode JSONL to review (default: {DEFAULT_DATA})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        rows = load_jsonl(args.data)
    except DataLoadError as exc:
        parser.error(str(exc))

    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(rows))
    except OSError as exc:
        parser.error(f"could not start HTTP server on {args.host}:{args.port}: {exc}")

    bound_host, bound_port = server.server_address[:2]
    display_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
    print(f"Loaded {len(rows)} episodes from {args.data}")
    print(f"Review GUI: http://{display_host}:{bound_port}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping review GUI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
