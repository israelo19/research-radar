"""Where new items come from: the arXiv API and Hacker News (Algolia)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

ATOM = "{http://www.w3.org/2005/Atom}"
UA = {"User-Agent": "research-radar/1.0 (personal reading list)"}

Fetch = Callable[[str], bytes]


def http_get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


@dataclass
class Item:
    id: str                 # stable id used for dedupe, e.g. "arxiv:2611.01234" or "hn:4512345"
    source: str             # "arxiv" or "hn"
    title: str
    url: str
    published: str          # ISO 8601
    summary: str = ""       # abstract for papers; empty for HN
    authors: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def arxiv_url(phrases: list[str], categories: list[str], max_results: int) -> str:
    terms = " OR ".join(f'abs:"{p}"' for p in phrases)
    cats = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({terms}) AND ({cats})" if categories else f"({terms})"
    params = urllib.parse.urlencode({
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    })
    return f"http://export.arxiv.org/api/query?{params}"


def parse_arxiv(xml_bytes: bytes) -> list[Item]:
    root = ET.fromstring(xml_bytes)
    items = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
        if not raw_id:
            continue
        short = raw_id.rsplit("/abs/", 1)[-1]
        short = short.split("v")[0] if "v" in short.rsplit(".", 1)[-1] else short
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
        published = (entry.findtext(f"{ATOM}published") or "").strip()
        authors = [" ".join((a.findtext(f"{ATOM}name") or "").split()) for a in entry.findall(f"{ATOM}author")]
        cats = [c.get("term") for c in entry.findall(f"{ATOM}category") if c.get("term")]
        items.append(Item(
            id=f"arxiv:{short}",
            source="arxiv",
            title=title,
            url=f"https://arxiv.org/abs/{short}",
            published=published,
            summary=summary,
            authors=authors,
            extra={"categories": cats, "pdf": f"https://arxiv.org/pdf/{short}"},
        ))
    return items


def hn_url(query: str, since: datetime) -> str:
    params = urllib.parse.urlencode({
        "query": query,
        "tags": "story",
        "numericFilters": f"created_at_i>{int(since.timestamp())}",
        "hitsPerPage": 30,
    })
    return f"https://hn.algolia.com/api/v1/search_by_date?{params}"


def parse_hn(json_bytes: bytes, min_points: int) -> list[Item]:
    data = json.loads(json_bytes)
    items = []
    for hit in data.get("hits", []):
        points = hit.get("points") or 0
        if points < min_points:
            continue
        oid = str(hit.get("objectID"))
        discussion = f"https://news.ycombinator.com/item?id={oid}"
        items.append(Item(
            id=f"hn:{oid}",
            source="hn",
            title=(hit.get("title") or "").strip(),
            url=hit.get("url") or discussion,
            published=hit.get("created_at") or "",
            authors=[hit.get("author") or ""],
            extra={"points": points, "comments": hit.get("num_comments") or 0, "discussion": discussion},
        ))
    return items


def parse_time(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
