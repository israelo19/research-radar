"""Where new items come from: the arXiv API and Hacker News (Algolia)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

ATOM = "{http://www.w3.org/2005/Atom}"
UA = {"User-Agent": "research-radar/1.0 (personal reading list)"}

Fetch = Callable[[str], bytes]


def http_get(url: str, timeout: float = 30, tries: int = 3) -> bytes:
    """GET with polite retries. arXiv rate-limits shared cloud IPs and answers 429 or 503."""
    import time
    import urllib.error
    for attempt in range(tries):
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == tries - 1:
                raise
            wait = e.headers.get("Retry-After") if e.headers else None
            delay = min(int(wait), 60) if wait and wait.isdigit() else 10 * (attempt + 1)
            time.sleep(delay)
    raise RuntimeError("unreachable")


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


def arxiv_rss_url(categories: list[str]) -> str:
    return "https://rss.arxiv.org/rss/" + "+".join(categories)


def parse_arxiv_rss(xml_bytes: bytes, phrases: list[str]) -> list[Item]:
    """Today's arXiv announcements from the RSS feed, kept only if they mention one of the phrases."""
    root = ET.fromstring(xml_bytes)
    ns = {"dc": "http://purl.org/dc/elements/1.1/", "arxiv": "http://arxiv.org/schemas/atom"}
    wanted = [p.lower() for p in phrases]
    items = []
    for it in root.iter("item"):
        if (it.findtext("arxiv:announce_type", default="new", namespaces=ns) or "new").strip() not in ("new", "cross"):
            continue
        link = (it.findtext("link") or "").strip()
        short = link.rsplit("/abs/", 1)[-1]
        if not short:
            continue
        title = " ".join((it.findtext("title") or "").split())
        desc = " ".join((it.findtext("description") or "").split())
        abstract = desc.split("Abstract:", 1)[1].strip() if "Abstract:" in desc else desc
        text = f"{title} {abstract}".lower()
        if not any(w in text for w in wanted):
            continue
        try:
            published = parsedate_to_datetime(it.findtext("pubDate") or "").astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            published = datetime.now(timezone.utc).isoformat()
        creators = it.findtext("dc:creator", default="", namespaces=ns)
        items.append(Item(
            id=f"arxiv:{short}",
            source="arxiv",
            title=title,
            url=f"https://arxiv.org/abs/{short}",
            published=published,
            summary=abstract,
            authors=[a.strip() for a in creators.split(",") if a.strip()],
            extra={"categories": [c.text for c in it.findall("category") if c.text], "pdf": f"https://arxiv.org/pdf/{short}",
                   "via": "rss"},
        ))
    return items
