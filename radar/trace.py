"""Tracing: every step of a run becomes a span you can click into later.

Spans are printed live as they start and finish (that's what you watch in the
GitHub Actions log while a run is going) and written to runs/<id>/trace.jsonl
when they end. The viewer rebuilds the tree from parent ids.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Span:
    def __init__(self, tracer: "Tracer", name: str, kind: str, parent: "Span | None", attrs: dict[str, Any]):
        self.tracer = tracer
        self.id = uuid.uuid4().hex[:12]
        self.parent_id = parent.id if parent else None
        self.name = name
        self.kind = kind
        self.attrs = dict(attrs)
        self.status = "running"
        self.error: str | None = None
        self.start = now_iso()
        self._t0 = time.perf_counter()
        self.duration_ms: float | None = None

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error

    def __enter__(self) -> "Span":
        self.tracer._started(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and self.status != "error":
            self.fail(f"{exc_type.__name__}: {exc}")
        if self.status == "running":
            self.status = "ok"
        self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 1)
        self.tracer._ended(self)
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "start": self.start,
            "duration_ms": self.duration_ms,
            "attrs": self.attrs,
        }


class Tracer:
    def __init__(self, run_dir: Path, live: bool = True, stream=None):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "trace.jsonl"
        self.path.write_text("")
        self.live = live
        self.stream = stream or sys.stdout
        self._lock = threading.Lock()
        self.spans: list[Span] = []

    def span(self, name: str, kind: str = "step", parent: Span | None = None, **attrs: Any) -> Span:
        return Span(self, name, kind, parent, attrs)

    def _depth(self, span: Span) -> int:
        depth, pid = 0, span.parent_id
        ids = {s.id: s for s in self.spans}
        while pid and pid in ids:
            depth += 1
            pid = ids[pid].parent_id
        return depth

    def _started(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)
            if self.live:
                label = span.attrs.get("label", "")
                print(f"{span.start[11:19]} {'  ' * self._depth(span)}> {span.name} {label}".rstrip(),
                      file=self.stream, flush=True)

    def _ended(self, span: Span) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
            if self.live:
                mark = "ok " if span.status == "ok" else "ERR"
                note = span.attrs.get("note", "")
                if span.error:
                    note = f"{note} {span.error}".strip()
                print(f"{now_iso()[11:19]} {'  ' * self._depth(span)}{mark} {span.name} "
                      f"({span.duration_ms / 1000:.1f}s) {note}".rstrip(),
                      file=self.stream, flush=True)


def load_trace(path: Path) -> list[dict[str, Any]]:
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            spans.append(json.loads(line))
    return spans
