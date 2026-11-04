"""Builds site/index.html: one self-contained page with recent runs baked in.

Open it straight from disk, from the Actions artifact, or from GitHub Pages.
"""
from __future__ import annotations

import json
from pathlib import Path

from .trace import load_trace

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def build(root: Path, keep: int = 14) -> Path:
    index_path = root / "runs" / "index.json"
    runs = json.loads(index_path.read_text()) if index_path.exists() else []
    bundle = []
    for summary in runs[:keep]:
        run_dir = root / "runs" / summary["run_id"]
        trace = load_trace(run_dir / "trace.jsonl") if (run_dir / "trace.jsonl").exists() else []
        items = json.loads((run_dir / "items.json").read_text()) if (run_dir / "items.json").exists() else []
        bundle.append({"summary": summary, "trace": trace, "items": items})
    data = json.dumps({"runs": bundle}, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__RADAR_DATA__", data)
    out = root / "site" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
