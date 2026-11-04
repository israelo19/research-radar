"""The radar job: fetch new items, have an open model triage each one, write a reading list."""
from __future__ import annotations

import json
import os
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from . import sources
from .llm import LLM, LLMError, Provider, StubLLM, extract_json
from .trace import Tracer, now_iso

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def resolve_keys(providers: list[Provider]) -> dict[str, str]:
    """Which providers have a key, and where it came from (the name only, never the value)."""
    return {p.name: f"key set ({p.key_env})" for p in providers if p.key}


def providers_from(cfg: dict[str, Any]) -> list[Provider]:
    return [Provider(p["name"], p["base_url"], p["key_env"], list(p["models"]),
                     prefer=list(p.get("prefer", [])), avoid=list(p.get("avoid", [])))
            for p in cfg["providers"]]


# ---------------------------------------------------------------- prompts

DEFAULT_PROMPT = (
    "You screen new papers and posts for an undergraduate security researcher in an AI security lab. "
    "He reads the original sources himself, so your job is to point, not to explain the paper for him.\n\n"
    "His research threads:\n{threads}\n\n"
    "Rules: use only the text you are given. If something isn't stated, don't guess. "
    "Reply with one JSON object and nothing else."
)


def load_prompt(root: Path = ROOT) -> str:
    """The editable part of the prompt: everything after the first '---' line in prompt.md."""
    path = root / "prompt.md"
    if not path.exists():
        return DEFAULT_PROMPT
    text = path.read_text(encoding="utf-8")
    parts = text.split("\n---\n", 1)
    body = (parts[1] if len(parts) == 2 else text).strip()
    if "{threads}" not in body:  # never drop the threads, even if the placeholder got deleted
        body += "\n\nHis research threads:\n{threads}"
    return body


def system_prompt(cfg: dict[str, Any], root: Path = ROOT) -> str:
    threads = "\n".join(f'- "{t["id"]}" ({t["label"]}): {t["about"]}' for t in cfg["threads"])
    return load_prompt(root).replace("{threads}", threads)


def item_prompt(item: sources.Item, thread_ids: list[str]) -> str:
    if item.source == "arxiv":
        body = (f"Source: arXiv paper ({', '.join(item.extra.get('categories', [])[:4])})\n"
                f"Title: {item.title}\nAuthors: {', '.join(item.authors[:8])}\n"
                f"Abstract: {item.summary}")
    else:
        body = (f"Source: Hacker News story ({item.extra.get('points', 0)} points, "
                f"{item.extra.get('comments', 0)} comments)\nTitle: {item.title}\nLink: {item.url}\n"
                "Only the title and link are available. Judge from those and keep the summary to what the title says.")
    choices = ", ".join(f'"{t}"' for t in thread_ids + ["none"])
    return (
        f"{body}\n\n"
        "Return JSON with exactly these keys:\n"
        '{"relevant": true or false,\n'
        f' "thread": one of [{choices}],\n'
        ' "score": integer 0-10 (10 = he should read this today, 0 = unrelated),\n'
        ' "summary": "one or two plain sentences on what it is",\n'
        ' "why": "one sentence on how it connects to his thread, or empty",\n'
        ' "question": "one question he should hold in mind while reading it, or empty"}\n'
        "/no_think"
    )


# ---------------------------------------------------------------- state

def load_seen(path: Path) -> dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_seen(path: Path, seen: dict[str, str], ttl_days: int) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).date().isoformat()
    kept = {k: v for k, v in seen.items() if v >= cutoff}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kept, indent=0, sort_keys=True) + "\n")


# ---------------------------------------------------------------- run

def triage_one(llm: LLM, cfg: dict, item: sources.Item, parent, idx: int, total: int) -> dict[str, Any]:
    thread_ids = [t["id"] for t in cfg["threads"]]
    messages = [{"role": "system", "content": system_prompt(cfg)},
                {"role": "user", "content": item_prompt(item, thread_ids)}]
    label = f"[{idx}/{total}] {item.id} {item.title[:70]}"
    result: dict[str, Any] = {"item": item.to_dict()}
    # Reasoning models (like Qwen3.8 on Bitdeer) think before answering, and a long
    # think can use up the whole budget with no answer left. The retry gets more room.
    budgets = [3000, 6000]
    for attempt, budget in enumerate(budgets):
        try:
            raw = llm.chat(messages, name="triage", parent=parent, label=label, max_tokens=budget)
            verdict = extract_json(raw)
            score = int(verdict.get("score", 0))
            thread = verdict.get("thread") if verdict.get("thread") in thread_ids else "none"
            result.pop("error", None)  # an earlier attempt may have failed; this one didn't
            result.update(
                relevant=bool(verdict.get("relevant")) and thread != "none",
                thread=thread,
                score=max(0, min(10, score)),
                summary=str(verdict.get("summary", "")).strip(),
                why=str(verdict.get("why", "")).strip(),
                question=str(verdict.get("question", "")).strip(),
                attempts=attempt + 1,
            )
            return result
        except LLMError as e:
            result.update(error=str(e))
            return result
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            result.update(error=f"unparseable reply: {e}")
            if raw.strip():  # it answered, just not in JSON
                messages = messages + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content": "That wasn't valid JSON. Reply with only the JSON object."}]
    return result


def digest_markdown(day: str, results: list[dict], cfg: dict, run_url: str | None) -> str:
    labels = {t["id"]: t["label"] for t in cfg["threads"]}
    threshold = cfg["run"]["read_threshold"]
    good = sorted([r for r in results if r.get("relevant")], key=lambda r: -r.get("score", 0))
    top = [r for r in good if r["score"] >= threshold][:3]
    rest = [r for r in good if r not in top]
    lines = [f"# Research radar, {day}", ""]
    if not good:
        lines += ["Nothing new matched your threads today.", ""]
    if top:
        lines += ["## Read today", ""]
        for r in top:
            it = r["item"]
            lines.append(f"**[{it['title']}]({it['url']})**  ")
            lines.append(f"{labels.get(r['thread'], r['thread'])} · score {r['score']}/10 · {it['source']}  ")
            if r.get("summary"):
                lines.append(r["summary"] + "  ")
            if r.get("why"):
                lines.append(f"Why it matters to you: {r['why']}  ")
            if r.get("question"):
                lines.append(f"Read it asking: {r['question']}")
            lines.append("")
    if rest:
        lines += ["## Also new", ""]
        for r in rest:
            it = r["item"]
            lines.append(f"- [{it['title']}]({it['url']}) · {labels.get(r['thread'], r['thread'])} · {r['score']}/10")
        lines.append("")
    errors = [r for r in results if r.get("error")]
    screened = len(results)
    lines.append(f"_Screened {screened} new items, {len(good)} relevant"
                 + (f", {len(errors)} failed" if errors else "") + "._")
    if run_url:
        lines.append(f"_Full trace: {run_url}_")
    return "\n".join(lines) + "\n"


def run(*, root: Path = ROOT, cfg: dict | None = None, dry_run: bool = False,
        fetch: Callable[[str], bytes] | None = None, llm_factory: Callable[[Tracer], LLM] | None = None,
        live: bool = True, now: datetime | None = None, open_issue: bool = True) -> dict[str, Any]:
    cfg = cfg or load_config(root / "config.toml")
    now = now or datetime.now(timezone.utc)
    run_id = now.strftime("%Y-%m-%dT%H%MZ") + ("-dry" if dry_run else "")
    run_dir = root / "runs" / run_id
    tracer = Tracer(run_dir, live=live)
    fetch = fetch or sources.http_get
    key_sources: dict[str, str] = {}
    if llm_factory:
        llm = llm_factory(tracer)
    elif dry_run:
        llm = StubLLM(tracer, stub_answer)
    else:
        key_sources = resolve_keys(providers_from(cfg))
        llm = LLM(providers_from(cfg), tracer)
        if not llm.available():
            key_sources = {"none found": "No Bitdeer key reached the run. The workflow reads the repo secret "
                                         "BITDEERAI_API_KEY (or BITDEER_API_KEY) from Settings, Secrets and "
                                         "variables, Actions, Repository secrets."}

    since = now - timedelta(days=cfg["run"]["lookback_days"])
    seen_path = root / "state" / "seen.json"
    seen = load_seen(seen_path)
    gh_run = os.environ.get("GITHUB_RUN_ID")
    run_url = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}"
               f"/actions/runs/{gh_run}") if gh_run and os.environ.get("GITHUB_REPOSITORY") else None

    summary: dict[str, Any] = {"run_id": run_id, "started": now_iso(), "dry_run": dry_run, "status": "ok",
                               "keys": key_sources}
    results: list[dict[str, Any]] = []
    items: list[sources.Item] = []
    with tracer.span("radar run", kind="run", label=run_id, dry_run=dry_run, since=since.isoformat(),
                     providers=[p.name for p in llm.available()], keys=summary["keys"]) as root_span:
        try:
            # 0. Which models each provider offers today (one cheap request each)
            if not dry_run and not llm_factory:
                for p in llm.available():
                    with tracer.span(f"list {p.name} models", kind="fetch", parent=root_span) as s:
                        try:
                            listed = p.list_models()
                            s.set(models=listed, preferred=p.rank(listed)[:5],
                                  note=f"{len(listed)} models")
                        except Exception as e:
                            s.fail(f"{type(e).__name__}: {e}"[:300])

            # 1. Fetch
            with tracer.span("fetch arxiv", kind="fetch", parent=root_span) as s:
                a = cfg["sources"]["arxiv"]
                url = sources.arxiv_url(a["phrases"], a["categories"], a.get("max_results", 100))
                s.set(url=url)
                try:
                    got = [i for i in sources.parse_arxiv(fetch(url)) if sources.parse_time(i.published) >= since]
                    items += got
                    s.set(count=len(got), via="api", note=f"{len(got)} papers since {since.date()} (API)")
                except Exception as e:
                    # The search API rate-limits GitHub's servers. Fall back to today's RSS
                    # announcements, filtered here by the same phrases.
                    s.set(api_error=f"{type(e).__name__}: {e}"[:300])
                    rss = sources.arxiv_rss_url(a["categories"])
                    s.set(rss_url=rss)
                    try:
                        got = [i for i in sources.parse_arxiv_rss(fetch(rss), a["phrases"])
                               if sources.parse_time(i.published) >= since]
                        items += got
                        s.set(count=len(got), via="rss", note=f"{len(got)} papers from RSS (API said {e})"[:200])
                    except Exception as e2:  # keep going with HN if arXiv is down entirely
                        s.fail(f"API: {type(e).__name__}: {e}; RSS: {type(e2).__name__}: {e2}"[:400])
            with tracer.span("fetch hacker news", kind="fetch", parent=root_span) as s:
                h = cfg["sources"]["hn"]
                found: dict[str, sources.Item] = {}
                urls = []
                for q in h["queries"]:
                    url = sources.hn_url(q, since)
                    urls.append(url)
                    try:
                        for it in sources.parse_hn(fetch(url), h.get("min_points", 0)):
                            found.setdefault(it.id, it)
                    except Exception as e:
                        s.set(partial_error=f"{q}: {type(e).__name__}: {e}")
                items += list(found.values())
                s.set(urls=urls, count=len(found), note=f"{len(found)} stories")

            # 2. Dedupe and cap
            with tracer.span("dedupe", parent=root_span) as s:
                fresh = [i for i in items if i.id not in seen]
                fresh.sort(key=lambda i: (i.source != "arxiv", -i.extra.get("points", 0)))
                cap = cfg["run"]["max_items"]
                todo = fresh[:cap]
                s.set(fetched=len(items), new=len(fresh), screening=len(todo),
                      note=f"{len(fresh)} new of {len(items)}, screening {len(todo)}")

            # 3. Triage in parallel
            with tracer.span("screen items", parent=root_span, note="") as s:
                with ThreadPoolExecutor(max_workers=max(1, cfg["run"]["concurrency"])) as pool:
                    futures = [pool.submit(triage_one, llm, cfg, it, s, n + 1, len(todo))
                               for n, it in enumerate(todo)]
                    results = [f.result() for f in futures]
                ok = [r for r in results if not r.get("error")]
                relevant = [r for r in ok if r.get("relevant")]
                s.set(note=f"{len(relevant)} relevant, {len(results) - len(ok)} failed")
                if todo and not ok:
                    raise LLMError("Every triage call failed. Check the API key and model access (python -m radar check).")

            # Only mark items seen once they were screened successfully.
            today = now.date().isoformat()
            for r in results:
                if not r.get("error"):
                    seen[r["item"]["id"]] = today
            # Items past the cap wait for tomorrow; nothing is dropped silently.

            # 4. Write outputs
            with tracer.span("write digest", parent=root_span) as s:
                md = digest_markdown(today, results, cfg, run_url)
                digest_path = root / "digests" / f"{today}{'-dry' if dry_run else ''}.md"
                digest_path.parent.mkdir(parents=True, exist_ok=True)
                digest_path.write_text(md)
                (run_dir / "items.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
                s.set(path=str(digest_path.relative_to(root)), digest=md)
                if os.environ.get("GITHUB_STEP_SUMMARY"):
                    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                        f.write(md)

            top_score = max([r.get("score", 0) for r in results if r.get("relevant")], default=0)
            if open_issue and not dry_run and top_score >= cfg["run"]["issue_threshold"]:
                from .notify import open_issue as gh_issue
                with tracer.span("open github issue", kind="fetch", parent=root_span) as s:
                    n_rel = len([r for r in results if r.get("relevant")])
                    url = gh_issue(f"Radar {today}: {n_rel} new for your threads", md)
                    s.set(url=url, note=url or "skipped (no GITHUB_TOKEN)")

            if not dry_run:
                save_seen(seen_path, seen, cfg["run"]["seen_ttl_days"])
        except Exception as e:
            summary["status"] = "error"
            summary["error"] = f"{type(e).__name__}: {e}"
            root_span.fail(summary["error"])

    summary.update(fetched=len(items), screened=len(results),
                   relevant=len([r for r in results if r.get("relevant")]),
                   failed=len([r for r in results if r.get("error")]),
                   top_score=max([r.get("score", 0) for r in results if r.get("relevant")], default=0))
    summary["duration_ms"] = root_span.duration_ms
    summary["providers"] = sorted({s.attrs.get("provider") for s in tracer.spans if s.attrs.get("provider")})
    summary["models"] = sorted({s.attrs.get("model") for s in tracer.spans if s.attrs.get("model")})
    summary["run_url"] = run_url
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    update_index(root, summary)
    return summary


def update_index(root: Path, summary: dict) -> None:
    path = root / "runs" / "index.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    runs = [r for r in runs if r["run_id"] != summary["run_id"]] + [summary]
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    path.write_text(json.dumps(runs, indent=2) + "\n")


def item_text(messages: list[dict]) -> str:
    """The item part of the triage prompt, without the JSON instructions (which name every thread)."""
    user = next(m["content"] for m in messages if m["role"] == "user")
    return user.split("\n\nReturn JSON", 1)[0]


def stub_answer(messages: list[dict]) -> str:
    """Deterministic fake verdicts for --dry-run, so the whole pipeline can be exercised offline."""
    text = item_text(messages).lower()
    rules = [("model context protocol", "mcp-security", 8), ("tool poisoning", "mcp-security", 9),
             ("worm", "ai-worms", 9), ("evaluation awareness", "eval-awareness", 9),
             ("being tested", "eval-awareness", 8), ("sandbox", "agent-escalation", 6),
             ("prompt injection", "mcp-security", 6), ("mcp server", "mcp-security", 7)]
    for needle, thread, score in rules:
        if needle in text:
            return ("<think>stub reasoning</think>\n" + json.dumps({
                "relevant": True, "thread": thread, "score": score,
                "summary": "Stub summary for a dry run. Real runs use the model's reading of the abstract.",
                "why": f"Stub: matched '{needle}'.", "question": "Stub question for a dry run."}))
    return json.dumps({"relevant": False, "thread": "none", "score": 1, "summary": "Stub: unrelated.",
                       "why": "", "question": ""})


def check(cfg: dict | None = None) -> int:
    """Confirm keys work and show which models each provider serves."""
    cfg = cfg or load_config()
    tracer = Tracer(ROOT / "runs" / "_check", live=False)
    for name, where in resolve_keys(providers_from(cfg)).items():
        print(f"{name}: key found in {where}")
    any_ok = False
    for p in providers_from(cfg):
        if not p.key:
            print(f"{p.name}: no key in ${p.key_env}, skipped")
            continue
        try:
            listed = p.list_models()
            print(f"{p.name}: serves {len(listed)} models: {', '.join(listed)}")
            for m in p.models:
                print(f"{p.name}: configured {m}: {'available' if m in listed else 'NOT offered'}")
            print(f"{p.name}: fallback order if those fail: {', '.join(p.rank(listed)[:5]) or 'none'}")
        except Exception as e:
            print(f"{p.name}: could not list models ({type(e).__name__}: {e})")
        llm = LLM([p], tracer, retries=0)
        try:
            reply = llm.chat([{"role": "user", "content": 'Reply with {"ok": true} and nothing else. /no_think'}],
                             name="check", max_tokens=300)
            print(f"{p.name}: test call OK with {tracer.spans[-1].attrs.get('model')}: {reply.strip()[:80]}")
            any_ok = True
        except LLMError:
            for a in tracer.spans[-1].attrs.get("attempts", []):
                print(f"{p.name}: {a['model']} -> {a['status']} {a.get('error', '')[:160]}")
    return 0 if any_ok else 1
