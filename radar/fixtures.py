"""Sample responses in the real arXiv Atom and HN Algolia formats, for --dry-run and tests.

Everything here is made up and marked [sample] so a dry run can't be mistaken for real papers.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

SAMPLE_PAPERS = [
    ("9999.00001", "[sample] Poisoned tool descriptions in Model Context Protocol servers",
     "We study tool poisoning attacks in which a malicious Model Context Protocol server hides instructions in a tool description, and measure how often agents follow them.",
     ["cs.CR", "cs.AI"]),
    ("9999.00002", "[sample] Do models know when they are being tested? Measuring evaluation awareness in coding agents",
     "We measure evaluation awareness: whether language models behave differently when a task looks like a benchmark versus real use.",
     ["cs.LG", "cs.AI"]),
    ("9999.00003", "[sample] A self-propagating worm across multi-agent coding assistants",
     "We build a worm that spreads between AI agents through the messages they exchange, and evaluate detection strategies.",
     ["cs.CR"]),
    ("9999.00004", "[sample] Efficient attention kernels for long-context inference",
     "We present faster attention kernels for long sequences on GPUs.",
     ["cs.LG"]),
]

SAMPLE_STORIES = [
    ("90000001", "[sample] Show HN: A scanner for malicious MCP servers", 142, 38),
    ("90000002", "[sample] My weekend sourdough experiment", 51, 12),
    ("90000003", "[sample] Low-points story that should be filtered", 2, 0),
]


def arxiv_xml(now: datetime) -> bytes:
    published = (now - timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries = []
    for pid, title, abstract, cats in SAMPLE_PAPERS:
        cat_xml = "".join(f'<category term="{c}" scheme="http://arxiv.org/schemas/atom"/>' for c in cats)
        entries.append(f"""
  <entry>
    <id>http://arxiv.org/abs/{pid}v1</id>
    <updated>{published}</updated>
    <published>{published}</published>
    <title>{title}</title>
    <summary>  {abstract}
    </summary>
    <author><name>Sample Author</name></author>
    <author><name>Another Author</name></author>
    <link href="http://arxiv.org/abs/{pid}v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/{pid}v1" rel="related" type="application/pdf"/>
    {cat_xml}
  </entry>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">ArXiv Query</title>
  <id>http://arxiv.org/api/sample</id>
  <opensearch:totalResults>{len(SAMPLE_PAPERS)}</opensearch:totalResults>{''.join(entries)}
</feed>""".encode()


def hn_json(now: datetime) -> bytes:
    created = now - timedelta(hours=6)
    hits = [{
        "objectID": oid, "title": title, "url": f"https://example.com/{oid}", "author": "sample_user",
        "points": pts, "num_comments": comments,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "created_at_i": int(created.timestamp()),
    } for oid, title, pts, comments in SAMPLE_STORIES]
    return json.dumps({"hits": hits, "nbHits": len(hits)}).encode()


def fixture_fetch(url: str, now: datetime | None = None) -> bytes:
    now = now or datetime.now(timezone.utc)
    if "arxiv.org" in url:
        return arxiv_xml(now)
    if "hn.algolia.com" in url:
        return hn_json(now)
    raise ValueError(f"no fixture for {url}")
