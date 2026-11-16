#!/usr/bin/env python3
"""Small, dependency-free HTTP server for the Gemma 4 trace browser."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import re
import sqlite3
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DB = ROOT / "traces.sqlite3"

FILTER_FIELDS = {
    "source": "t.source",
    "parent": "t.parent",
    "step": "t.checkpoint_step",
    "correctness": "t.correctness",
    "outcome": "t.outcome",
    "clause_split": "t.clause_split",
    "presentation": "t.presentation",
    "episode_kind": "t.episode_kind",
    "target_clause": "t.target_clause",
    "clause_family": "t.clause_family",
    "template_id": "t.template_id",
    "finish_reason": "t.finish_reason",
}

BOOL_FIELDS = {
    "format_valid": "t.format_valid",
    "boundary_valid": "t.boundary_valid",
    "grammar_valid": "t.grammar_valid",
    "truncated": "t.truncated",
}

DETAIL_COLUMNS = (
    "id", "uid", "source", "parent", "checkpoint_step", "checkpoint_type",
    "sample_index", "group_id", "episode_id", "eval_set", "clause_split",
    "presentation", "episode_kind", "target_clause", "clause_family",
    "template_id", "correctness", "outcome", "reward", "semantic_score",
    "format_valid", "boundary_valid", "grammar_valid", "finish_reason",
    "prompt_tokens", "completion_tokens", "truncated", "prompt", "raw_prompt",
    "reasoning", "response", "raw_output", "charter_plan", "coin_plan",
    "model_plan", "run_verdicts", "tags_json", "episode_json", "metadata_json",
)

LIST_COLUMNS = """
    t.id, t.uid, t.source, t.parent, t.checkpoint_step, t.episode_id,
    t.eval_set, t.clause_split, t.presentation, t.episode_kind,
    t.target_clause, t.clause_family, t.template_id, t.correctness, t.outcome,
    t.reward, t.semantic_score, t.format_valid, t.boundary_valid,
    t.grammar_valid, t.finish_reason, t.completion_tokens, t.truncated,
    substr(t.prompt, 1, 420) AS prompt_preview,
    substr(t.reasoning, 1, 760) AS reasoning_preview,
    substr(t.response, 1, 420) AS response_preview,
    t.tags_json
"""


def connect() -> sqlite3.Connection:
    uri = f"file:{quote(str(DB))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def hydrate(row: sqlite3.Row | dict[str, Any], detail: bool = False) -> dict[str, Any]:
    item = dict(row)
    for field in ("format_valid", "boundary_valid", "grammar_valid", "truncated"):
        if field in item:
            item[field] = bool(item[field])
    for field in ("tags_json", "charter_plan", "coin_plan", "model_plan", "run_verdicts"):
        if field in item:
            item[field.removesuffix("_json")] = json_value(item.pop(field))
    if detail:
        for field in ("episode_json", "metadata_json"):
            item[field.removesuffix("_json")] = json_value(item.pop(field))
    return item


def fts_expression(query: str, scope: str) -> str:
    # Treat the user's input as literal search terms. Quoted phrases stay phrases;
    # punctuation cannot accidentally become an FTS operator.
    pieces = re.findall(r'"([^"]+)"|(\S+)', query)
    terms = []
    for phrase, word in pieces:
        term = phrase or word
        term = term.replace('"', '""').strip()
        if term:
            terms.append(f'"{term}"')
    if not terms:
        return ""
    body = " AND ".join(terms)
    columns = {
        "prompt": "prompt",
        "reasoning": "reasoning",
        "response": "response",
        "output": "{reasoning response}",
        "all": "{prompt reasoning response}",
    }.get(scope, "{reasoning response}")
    return f"{columns} : ({body})"


def query_parts(qs: dict[str, list[str]]) -> tuple[list[str], list[Any], bool]:
    where: list[str] = []
    params: list[Any] = []
    for name, column in FILTER_FIELDS.items():
        values = [value for value in qs.get(name, []) if value != ""]
        if not values:
            continue
        where.append(f"{column} IN ({','.join('?' for _ in values)})")
        params.extend(int(value) if name == "step" else value for value in values)

    for name, column in BOOL_FIELDS.items():
        values = qs.get(name, [])
        if values and values[-1] in {"0", "1"}:
            where.append(f"{column} = ?")
            params.append(int(values[-1]))

    numeric = (
        ("step_min", "t.checkpoint_step", ">=", int),
        ("step_max", "t.checkpoint_step", "<=", int),
        ("reward_min", "t.reward", ">=", float),
        ("reward_max", "t.reward", "<=", float),
        ("tokens_min", "t.completion_tokens", ">=", int),
        ("tokens_max", "t.completion_tokens", "<=", int),
    )
    for name, column, operator, cast in numeric:
        value = qs.get(name, [""])[-1]
        if value != "":
            try:
                params.append(cast(value))
            except ValueError as exc:
                raise ValueError(f"invalid {name}: {value}") from exc
            where.append(f"{column} {operator} ?")

    for tag in qs.get("tag", []):
        if tag:
            where.append(
                "EXISTS (SELECT 1 FROM trace_tags tt "
                "WHERE tt.trace_id=t.id AND tt.tag=?)"
            )
            params.append(tag)

    fts = fts_expression(qs.get("q", [""])[-1].strip(), qs.get("scope", ["output"])[-1])
    if fts:
        where.append("traces_fts MATCH ?")
        params.append(fts)
    return where, params, bool(fts)


def from_sql(has_fts: bool) -> str:
    if has_fts:
        return "traces t JOIN traces_fts ON traces_fts.rowid=t.id"
    return "traces t"


def filtered_sql(qs: dict[str, list[str]]) -> tuple[str, list[Any], bool]:
    where, params, has_fts = query_parts(qs)
    return (" WHERE " + " AND ".join(where) if where else ""), params, has_fts


def sort_sql(value: str, has_fts: bool) -> str:
    choices = {
        "step_asc": "t.parent, t.checkpoint_step, t.source, t.id",
        "step_desc": "t.parent, t.checkpoint_step DESC, t.source, t.id",
        "newest": "t.id DESC",
        "reward_desc": "t.reward IS NULL, t.reward DESC, t.id",
        "reward_asc": "t.reward IS NULL, t.reward, t.id",
        "tokens_desc": "t.completion_tokens DESC, t.id",
        "tokens_asc": "t.completion_tokens, t.id",
        "random": "random()",
    }
    if value == "relevance" and has_fts:
        return "bm25(traces_fts), t.id"
    return choices.get(value, "t.parent, t.checkpoint_step, t.source, t.id")


def grouped_counts(
    connection: sqlite3.Connection,
    base_from: str,
    where_sql: str,
    params: list[Any],
    field: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        f"SELECT t.{field} value, COUNT(*) count FROM {base_from}{where_sql} "
        f"GROUP BY t.{field} ORDER BY count DESC, value",
        params,
    )
    return [{"value": row["value"], "count": row["count"]} for row in rows]


def meta_payload() -> dict[str, Any]:
    with connect() as connection:
        info = {
            row["key"]: json_value(row["value"])
            for row in connection.execute("SELECT key,value FROM build_info")
        }
        facets: dict[str, list[dict[str, Any]]] = {}
        for field in (
            "source", "parent", "checkpoint_step", "correctness", "outcome",
            "clause_split", "presentation", "episode_kind", "target_clause",
            "clause_family", "template_id", "finish_reason",
        ):
            rows = connection.execute(
                f"SELECT {field} value, COUNT(*) count FROM traces "
                f"GROUP BY {field} ORDER BY count DESC, value"
            )
            facets[field] = [
                {"value": row["value"], "count": row["count"]}
                for row in rows if row["value"] not in (None, "")
            ]
        tags = connection.execute(
            "SELECT tag value, COUNT(*) count FROM trace_tags GROUP BY tag "
            "ORDER BY tag"
        )
        facets["tag"] = [dict(row) for row in tags]
        endpoints = connection.execute(
            "SELECT source,parent,checkpoint_step,COUNT(*) count FROM traces "
            "GROUP BY source,parent,checkpoint_step "
            "ORDER BY source,parent,checkpoint_step"
        )
        return {"build": info, "facets": facets, "endpoints": [dict(r) for r in endpoints]}


def search_payload(qs: dict[str, list[str]]) -> dict[str, Any]:
    where_sql, params, has_fts = filtered_sql(qs)
    base_from = from_sql(has_fts)
    try:
        limit = min(max(int(qs.get("limit", ["50"])[-1]), 1), 200)
        offset = max(int(qs.get("offset", ["0"])[-1]), 0)
    except ValueError as exc:
        raise ValueError("limit and offset must be integers") from exc
    order = sort_sql(qs.get("sort", ["step_asc"])[-1], has_fts)
    with connect() as connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM {base_from}{where_sql}", params
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT {LIST_COLUMNS} FROM {base_from}{where_sql} "
            f"ORDER BY {order} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        items = [hydrate(row) for row in rows]
        summary = {
            field: grouped_counts(connection, base_from, where_sql, params, field)
            for field in ("correctness", "outcome", "parent", "source")
        }
        aggregates = connection.execute(
            f"SELECT AVG(t.reward) avg_reward, AVG(t.semantic_score) avg_semantic, "
            f"AVG(t.format_valid) format_rate, AVG(t.boundary_valid) boundary_rate "
            f"FROM {base_from}{where_sql}",
            params,
        ).fetchone()
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
        "summary": summary,
        "aggregates": dict(aggregates),
    }


def trace_payload(identifier: str) -> dict[str, Any] | None:
    column = "id" if identifier.isdigit() else "uid"
    value: str | int = int(identifier) if identifier.isdigit() else identifier
    with connect() as connection:
        row = connection.execute(
            f"SELECT {','.join(DETAIL_COLUMNS)} FROM traces WHERE {column}=?", (value,)
        ).fetchone()
    return hydrate(row, detail=True) if row else None


def export_rows(qs: dict[str, list[str]]) -> Iterable[dict[str, Any]]:
    where_sql, params, has_fts = filtered_sql(qs)
    base_from = from_sql(has_fts)
    order = sort_sql(qs.get("sort", ["step_asc"])[-1], has_fts)
    connection = connect()
    try:
        rows = connection.execute(
            f"SELECT {','.join('t.' + col for col in DETAIL_COLUMNS)} "
            f"FROM {base_from}{where_sql} ORDER BY {order}",
            params,
        )
        for row in rows:
            yield hydrate(row, detail=True)
    finally:
        connection.close()


class Handler(SimpleHTTPRequestHandler):
    server_version = "GemmaTraceBrowser/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query, keep_blank_values=False)
        try:
            if parsed.path == "/api/health":
                return self.send_json({"ok": DB.exists(), "database": str(DB)})
            if parsed.path == "/api/meta":
                return self.send_json(meta_payload())
            if parsed.path == "/api/search":
                return self.send_json(search_payload(qs))
            if parsed.path.startswith("/api/trace/"):
                identifier = parsed.path.removeprefix("/api/trace/")
                payload = trace_payload(identifier)
                if payload is None:
                    return self.send_error_json(HTTPStatus.NOT_FOUND, "trace not found")
                return self.send_json(payload)
            if parsed.path == "/api/export.jsonl":
                return self.send_jsonl_export(qs)
            if parsed.path == "/api/export.csv":
                return self.send_csv_export(qs)
            if parsed.path.startswith("/api/"):
                return self.send_error_json(HTTPStatus.NOT_FOUND, "unknown API endpoint")
            return self.send_static(parsed.path)
        except (ValueError, sqlite3.OperationalError) as exc:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except BrokenPipeError:
            return None

    def send_static(self, path: str) -> None:
        requested = "index.html" if path in {"", "/"} else path.lstrip("/")
        candidate = (STATIC / requested).resolve()
        if STATIC.resolve() not in candidate.parents and candidate != STATIC.resolve():
            return self.send_error_json(HTTPStatus.FORBIDDEN, "invalid path")
        if not candidate.is_file():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "file not found")
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_jsonl_export(self, qs: dict[str, list[str]]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="gemma4-traces.jsonl"')
        self.end_headers()
        for item in export_rows(qs):
            self.wfile.write(json.dumps(item, ensure_ascii=False).encode() + b"\n")

    def send_csv_export(self, qs: dict[str, list[str]]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="gemma4-traces.csv"')
        self.end_headers()
        stream = io.StringIO()
        writer: csv.DictWriter[str] | None = None
        for item in export_rows(qs):
            flat = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in item.items()
            }
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(flat))
                writer.writeheader()
            writer.writerow(flat)
            self.wfile.write(stream.getvalue().encode())
            stream.seek(0)
            stream.truncate(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8877, type=int)
    parser.add_argument("--database", type=Path, default=DB)
    return parser.parse_args()


def main() -> None:
    global DB
    args = parse_args()
    DB = args.database.resolve()
    if not DB.is_file():
        raise SystemExit(f"database does not exist: {DB}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Gemma 4 trace browser: http://{args.host}:{args.port}", flush=True)
    print(f"Database: {DB}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
