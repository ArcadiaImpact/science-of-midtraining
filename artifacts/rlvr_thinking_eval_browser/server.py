#!/usr/bin/env python3
"""Dependency-free HTTP API and static server for the RLVR eval browser."""

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
    "arm": "t.arm",
    "step": "t.checkpoint_step",
    "surface": "t.surface",
    "episode_kind": "t.episode_kind",
    "run_count": "t.run_count",
    "outcome": "t.outcome",
    "answer_quality": "t.answer_quality",
    "run_pattern": "t.run_pattern",
    "target_clause": "t.target_clause",
    "clause_family": "t.clause_family",
    "template_id": "t.template_id",
    "finish_reason": "t.finish_reason",
}
BOOL_FIELDS = {
    "truncated": "t.truncated",
    "parser_valid": "t.parser_valid",
    "format_valid": "t.format_valid",
    "boundary_valid": "t.boundary_valid",
}
INTEGER_FIELDS = {"step", "run_count"}

LIST_COLUMNS = """
    t.id, t.uid, t.arm, t.checkpoint_step, t.episode_id, t.family,
    t.surface, t.episode_kind, t.run_count, t.template_id, t.target_clause,
    t.clause_family, t.outcome, t.answer_quality, t.run_pattern,
    t.run_verdicts, t.parser_valid, t.format_valid, t.boundary_valid,
    t.truncated, t.finish_reason, t.completion_tokens, t.reasoning_chars,
    substr(p.prompt,1,360) prompt_preview,
    substr(t.reasoning,1,680) reasoning_preview,
    substr(t.final_answer,1,320) final_preview
"""


def connect() -> sqlite3.Connection:
    uri = f"file:{quote(str(DB))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-131072")
    return connection


def decode(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def hydrate(row: sqlite3.Row | dict[str, Any], detail: bool = False) -> dict[str, Any]:
    item = dict(row)
    for field in ("parser_valid", "format_valid", "boundary_valid", "truncated"):
        if field in item:
            item[field] = bool(item[field])
    for field in ("run_verdicts", "parsed_plan", "charter_plan", "coin_plan"):
        if field in item:
            item[field] = decode(item[field])
    if detail:
        item["episode"] = decode(item.pop("episode_json"))
        item["metadata"] = decode(item.pop("metadata_json"))
        episode = item["episode"]
        parsed = item.get("parsed_plan") or []
        verdicts = item.get("run_verdicts") or []
        item["run_details"] = [
            {
                "run_id": run["run_id"],
                "port": run.get("port"),
                "model": parsed[index] if index < len(parsed) else None,
                "charter": item["charter_plan"][index],
                "coin": item["coin_plan"][index],
                "verdict": verdicts[index] if index < len(verdicts) else "malformed",
            }
            for index, run in enumerate(episode["runs"])
        ]
    return item


def fts_expression(query: str, scope: str) -> str:
    pieces = re.findall(r'"([^"]+)"|(\S+)', query)
    terms = []
    for phrase, word in pieces:
        term = (phrase or word).replace('"', '""').strip()
        if term:
            terms.append(f'"{term}"')
    if not terms:
        return ""
    columns = {
        "reasoning": "reasoning",
        "answer": "final_answer",
        "prompt": "prompt",
        "output": "{reasoning final_answer}",
        "all": "{reasoning final_answer prompt}",
    }.get(scope, "reasoning")
    return f"{columns} : ({' AND '.join(terms)})"


def query_parts(qs: dict[str, list[str]]) -> tuple[list[str], list[Any], bool]:
    where: list[str] = []
    params: list[Any] = []
    for name, column in FILTER_FIELDS.items():
        values = [value for value in qs.get(name, []) if value != ""]
        if not values:
            continue
        where.append(f"{column} IN ({','.join('?' for _ in values)})")
        params.extend(int(value) if name in INTEGER_FIELDS else value for value in values)
    for name, column in BOOL_FIELDS.items():
        value = qs.get(name, [""])[-1]
        if value in {"0", "1"}:
            where.append(f"{column}=?")
            params.append(int(value))
    verdicts = [value for value in qs.get("verdict", []) if value]
    if verdicts:
        where.append(
            "EXISTS (SELECT 1 FROM trace_verdicts tv WHERE tv.trace_id=t.id "
            f"AND tv.verdict IN ({','.join('?' for _ in verdicts)}))"
        )
        params.extend(verdicts)
    for name, column, operator in (
        ("tokens_min", "t.completion_tokens", ">="),
        ("tokens_max", "t.completion_tokens", "<="),
    ):
        value = qs.get(name, [""])[-1]
        if value:
            try:
                params.append(int(value))
            except ValueError as exc:
                raise ValueError(f"invalid {name}: {value}") from exc
            where.append(f"{column}{operator}?")
    fts = fts_expression(qs.get("q", [""])[-1].strip(), qs.get("scope", ["reasoning"])[-1])
    if fts:
        where.append("traces_fts MATCH ?")
        params.append(fts)
    return where, params, bool(fts)


def base_from(has_fts: bool) -> str:
    fts = " JOIN traces_fts ON traces_fts.rowid=t.id" if has_fts else ""
    return f"traces t JOIN prompts p ON p.prompt_id=t.prompt_id{fts}"


def filtered_sql(qs: dict[str, list[str]]) -> tuple[str, list[Any], bool]:
    where, params, has_fts = query_parts(qs)
    return (" WHERE " + " AND ".join(where) if where else ""), params, has_fts


def sort_sql(value: str, has_fts: bool) -> str:
    choices = {
        "step_asc": "t.checkpoint_step,t.arm,t.surface,t.id",
        "step_desc": "t.checkpoint_step DESC,t.arm,t.surface,t.id",
        "tokens_desc": "t.completion_tokens DESC,t.id",
        "tokens_asc": "t.completion_tokens,t.id",
        "outcome": "t.outcome,t.arm,t.checkpoint_step,t.id",
        "random": "random()",
    }
    if value == "relevance" and has_fts:
        return "bm25(traces_fts),t.id"
    return choices.get(value, choices["step_asc"])


def grouped_counts(
    connection: sqlite3.Connection,
    source: str,
    where_sql: str,
    params: list[Any],
    field: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        f"SELECT t.{field} value,COUNT(*) count FROM {source}{where_sql} "
        f"GROUP BY t.{field} ORDER BY count DESC,value",
        params,
    )
    return [dict(row) for row in rows]


def meta_payload() -> dict[str, Any]:
    with connect() as connection:
        build = {
            row["key"]: decode(row["value"])
            for row in connection.execute("SELECT key,value FROM build_info")
        }
        facets: dict[str, list[dict[str, Any]]] = {}
        for field in FILTER_FIELDS:
            rows = connection.execute(
                f"SELECT {field if field != 'step' else 'checkpoint_step'} value,COUNT(*) count "
                "FROM traces GROUP BY value ORDER BY count DESC,value"
            )
            facets[field] = [dict(row) for row in rows if row["value"] not in (None, "")]
        verdict_rows = connection.execute(
            "SELECT verdict value,COUNT(*) count FROM trace_verdicts GROUP BY verdict ORDER BY verdict"
        )
        facets["verdict"] = [dict(row) for row in verdict_rows]
        return {"build": build, "facets": facets}


def search_payload(qs: dict[str, list[str]]) -> dict[str, Any]:
    where_sql, params, has_fts = filtered_sql(qs)
    source = base_from(has_fts)
    try:
        limit = min(max(int(qs.get("limit", ["50"])[-1]), 1), 200)
        offset = max(int(qs.get("offset", ["0"])[-1]), 0)
    except ValueError as exc:
        raise ValueError("limit and offset must be integers") from exc
    order = sort_sql(qs.get("sort", ["step_asc"])[-1], has_fts)
    with connect() as connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM {source}{where_sql}", params
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT {LIST_COLUMNS} FROM {source}{where_sql} ORDER BY {order} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        items = [hydrate(row) for row in rows]
        aggregates = dict(
            connection.execute(
                f"SELECT AVG(t.truncated) truncation_rate,AVG(t.parser_valid) parser_valid_rate,"
                f"AVG(t.completion_tokens) mean_tokens FROM {source}{where_sql}",
                params,
            ).fetchone()
        )
        summary = {
            field: grouped_counts(connection, source, where_sql, params, field)
            for field in ("outcome", "run_count", "arm", "checkpoint_step")
        }
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
        "aggregates": aggregates,
        "summary": summary,
    }


def trace_payload(identifier: str) -> dict[str, Any] | None:
    column = "t.id" if identifier.isdigit() else "t.uid"
    value: int | str = int(identifier) if identifier.isdigit() else identifier
    with connect() as connection:
        row = connection.execute(
            f"""SELECT t.*,p.prompt,e.charter_plan,e.coin_plan,e.episode_json
                FROM traces t JOIN prompts p ON p.prompt_id=t.prompt_id
                JOIN episodes e ON e.episode_id=t.episode_id
                WHERE {column}=?""",
            (value,),
        ).fetchone()
    return hydrate(row, detail=True) if row else None


def export_rows(qs: dict[str, list[str]]) -> Iterable[dict[str, Any]]:
    where_sql, params, has_fts = filtered_sql(qs)
    source = base_from(has_fts)
    order = sort_sql(qs.get("sort", ["step_asc"])[-1], has_fts)
    connection = connect()
    try:
        rows = connection.execute(
            f"""SELECT t.*,p.prompt,e.charter_plan,e.coin_plan,e.episode_json
                FROM {source} JOIN episodes e ON e.episode_id=t.episode_id
                {where_sql} ORDER BY {order}""",
            params,
        )
        for row in rows:
            yield hydrate(row, detail=True)
    finally:
        connection.close()


class Handler(SimpleHTTPRequestHandler):
    server_version = "RLVRTraceBrowser/1.0"

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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query, keep_blank_values=False)
        try:
            if parsed.path == "/api/health":
                return self.send_json({"ok": DB.is_file(), "database": str(DB)})
            if parsed.path == "/api/meta":
                return self.send_json(meta_payload())
            if parsed.path == "/api/search":
                return self.send_json(search_payload(qs))
            if parsed.path.startswith("/api/trace/"):
                payload = trace_payload(parsed.path.removeprefix("/api/trace/"))
                return self.send_json(payload) if payload else self.send_json(
                    {"error": "trace not found"}, HTTPStatus.NOT_FOUND
                )
            if parsed.path == "/api/export.jsonl":
                return self.send_jsonl(qs)
            if parsed.path == "/api/export.csv":
                return self.send_csv(qs)
            if parsed.path.startswith("/api/"):
                return self.send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            return self.send_static(parsed.path)
        except (ValueError, sqlite3.OperationalError) as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except BrokenPipeError:
            return None

    def send_static(self, requested_path: str) -> None:
        requested = "index.html" if requested_path in {"", "/"} else requested_path.lstrip("/")
        candidate = (STATIC / requested).resolve()
        if STATIC.resolve() not in candidate.parents:
            return self.send_json({"error": "invalid path"}, HTTPStatus.FORBIDDEN)
        if not candidate.is_file():
            return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_jsonl(self, qs: dict[str, list[str]]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="rlvr-eval-traces.jsonl"')
        self.end_headers()
        for item in export_rows(qs):
            self.wfile.write(json.dumps(item, ensure_ascii=False).encode() + b"\n")

    def send_csv(self, qs: dict[str, list[str]]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="rlvr-eval-traces.csv"')
        self.end_headers()
        stream = io.StringIO()
        writer: csv.DictWriter[str] | None = None
        for item in export_rows(qs):
            flat = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in item.items()
            }
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(flat))
                writer.writeheader()
            writer.writerow(flat)
            self.wfile.write(stream.getvalue().encode())
            stream.seek(0)
            stream.truncate(0)


def main() -> None:
    global DB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8878, type=int)
    parser.add_argument("--database", type=Path, default=DB)
    args = parser.parse_args()
    DB = args.database.resolve()
    if not DB.is_file():
        raise SystemExit(f"database does not exist: {DB}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RLVR thinking eval browser: http://{args.host}:{args.port}", flush=True)
    print(f"Database: {DB}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
