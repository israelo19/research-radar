"""Offline tests: the whole pipeline with sample data, and the HTTP client against a fake provider.

Run with: python -m unittest -v
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from radar import job, viewer
from radar.fixtures import fixture_fetch
from radar.llm import LLM, LLMError, Provider, StubLLM, extract_json, strip_reasoning
from radar.trace import Tracer, load_trace

NOW = datetime(2026, 11, 3, 11, 23, tzinfo=timezone.utc)


def fetch_at_now(url):
    return fixture_fetch(url, now=NOW)


class TempRoot(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.cfg = job.load_config()
        for k in ("GITHUB_RUN_ID", "GITHUB_REPOSITORY", "GITHUB_STEP_SUMMARY", "GITHUB_TOKEN"):
            os.environ.pop(k, None)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class ParsingTests(unittest.TestCase):
    def test_strip_reasoning(self):
        self.assertEqual(strip_reasoning("<think>hmm</think>\n{\"a\":1}"), '{"a":1}')
        self.assertEqual(strip_reasoning("half a thought</think>{\"a\":1}"), '{"a":1}')

    def test_extract_json_variants(self):
        self.assertEqual(extract_json('<think>x</think> Sure! {"score": 7, "why": "a } in text"} done'),
                         {"score": 7, "why": "a } in text"})
        self.assertEqual(extract_json('```json\n{"relevant": true}\n```'), {"relevant": True})
        with self.assertRaises(ValueError):
            extract_json("no json here")


class PipelineTests(TempRoot):
    def test_dry_run_end_to_end(self):
        out = io.StringIO()
        summary = job.run(root=self.root, cfg=self.cfg, dry_run=True, fetch=fetch_at_now,
                          llm_factory=lambda t: StubLLM(t, job.stub_answer), now=NOW, live=False)
        self.assertEqual(summary["status"], "ok", summary.get("error"))
        # 4 papers + 2 stories (the 2-point story is filtered by min_points)
        self.assertEqual(summary["fetched"], 6)
        self.assertEqual(summary["screened"], 6)
        self.assertEqual(summary["relevant"], 4)  # MCP, eval awareness, worm papers + MCP scanner story

        run_dir = self.root / "runs" / summary["run_id"]
        spans = load_trace(run_dir / "trace.jsonl")
        names = [s["name"] for s in spans]
        self.assertEqual(names.count("triage"), 6)
        for s in spans:
            if s["name"] == "triage":
                self.assertIn("messages", s["attrs"])
                self.assertIn("response", s["attrs"])
        root_span = [s for s in spans if s["parent"] is None]
        self.assertEqual(len(root_span), 1)
        self.assertTrue(all(s["parent"] for s in spans if s is not root_span[0]))

        digest = (self.root / "digests" / "2026-11-03-dry.md").read_text()
        self.assertIn("## Read today", digest)
        self.assertIn("Poisoned tool descriptions", digest)
        self.assertNotIn("sourdough", digest)
        self.assertFalse((self.root / "state" / "seen.json").exists(), "dry runs must not mark items seen")

        html = viewer.build(self.root).read_text()
        raw = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1)
        data = json.loads(raw.replace("<\\/", "</"))
        self.assertEqual(data["runs"][0]["summary"]["run_id"], summary["run_id"])
        self.assertNotIn("__RADAR_DATA__", html)

    def test_real_run_marks_seen_and_skips_next_time(self):
        mk = lambda t: StubLLM(t, job.stub_answer)
        first = job.run(root=self.root, cfg=self.cfg, fetch=fetch_at_now, llm_factory=mk, now=NOW,
                        live=False, open_issue=False)
        self.assertEqual(first["screened"], 6)
        seen = json.loads((self.root / "state" / "seen.json").read_text())
        self.assertIn("arxiv:9999.00001", seen)
        self.assertIn("hn:90000001", seen)
        second = job.run(root=self.root, cfg=self.cfg, fetch=fetch_at_now, llm_factory=mk,
                         now=NOW.replace(minute=40), live=False, open_issue=False)
        self.assertEqual(second["screened"], 0)
        index = json.loads((self.root / "runs" / "index.json").read_text())
        self.assertEqual(len(index), 2)

    def test_failed_items_are_retried_next_run(self):
        def flaky(tracer):
            def answer(messages):
                if "worm" in job.item_text(messages).lower():
                    return "I refuse to answer in JSON"
                return job.stub_answer(messages)
            return StubLLM(tracer, answer)
        s = job.run(root=self.root, cfg=self.cfg, fetch=fetch_at_now, llm_factory=flaky, now=NOW,
                    live=False, open_issue=False)
        self.assertEqual(s["failed"], 1)
        seen = json.loads((self.root / "state" / "seen.json").read_text())
        self.assertNotIn("arxiv:9999.00003", seen)

    def test_retry_that_succeeds_is_not_counted_as_failed(self):
        calls = {}
        def thinker(tracer):
            def answer(messages):
                key = job.item_text(messages)
                calls[key] = calls.get(key, 0) + 1
                if "worm" in key.lower() and calls[key] == 1:
                    return ""  # reasoning used the whole budget, no answer
                return job.stub_answer(messages)
            return StubLLM(tracer, answer)
        s = job.run(root=self.root, cfg=self.cfg, fetch=fetch_at_now, llm_factory=thinker, now=NOW,
                    live=False, open_issue=False)
        self.assertEqual(s["failed"], 0)
        self.assertEqual(s["relevant"], 4)
        seen = json.loads((self.root / "state" / "seen.json").read_text())
        self.assertIn("arxiv:9999.00003", seen)

    def test_source_outage_still_produces_a_run(self):
        def half_down(url):
            if "arxiv" in url:
                raise OSError("arXiv unreachable")
            return fetch_at_now(url)
        s = job.run(root=self.root, cfg=self.cfg, fetch=half_down,
                    llm_factory=lambda t: StubLLM(t, job.stub_answer), now=NOW, live=False, open_issue=False)
        self.assertEqual(s["status"], "ok")
        self.assertEqual(s["fetched"], 2)
        spans = load_trace(self.root / "runs" / s["run_id"] / "trace.jsonl")
        arx = [x for x in spans if x["name"] == "fetch arxiv"][0]
        self.assertEqual(arx["status"], "error")

    def test_arxiv_rate_limit_falls_back_to_rss(self):
        from radar.fixtures import arxiv_rss
        def fetch(url):
            if "export.arxiv.org" in url:
                raise OSError("HTTP Error 429: Too Many Requests")
            if "rss.arxiv.org" in url:
                return arxiv_rss(NOW)
            return fetch_at_now(url)
        s = job.run(root=self.root, cfg=self.cfg, fetch=fetch,
                    llm_factory=lambda t: StubLLM(t, job.stub_answer), now=NOW, live=False, open_issue=False)
        self.assertEqual(s["status"], "ok")
        spans = load_trace(self.root / "runs" / s["run_id"] / "trace.jsonl")
        arx = [x for x in spans if x["name"] == "fetch arxiv"][0]
        self.assertEqual(arx["attrs"]["via"], "rss")
        self.assertEqual(arx["attrs"]["count"], 2)   # 3 RSS items, one off-topic is filtered out
        self.assertEqual(s["fetched"], 4)            # 2 papers + 2 HN stories

    def test_no_keys_fails_loudly(self):
        for p in self.cfg["providers"]:
            os.environ.pop(p["key_env"], None)
        s = job.run(root=self.root, cfg=self.cfg, fetch=fetch_at_now, now=NOW, live=False, open_issue=False)
        self.assertEqual(s["status"], "error")
        self.assertIn("API key", s["error"])
        self.assertIn("BITDEERAI_API_KEY", s["keys"]["none found"])


class FakeProvider(BaseHTTPRequestHandler):
    """Behaves like an OpenAI-compatible endpoint with scripted failures."""
    script: dict = {}
    calls: list = []

    def log_message(self, *a):
        pass

    models = ["BAAI/bge-m3-embed", "deepseek-ai/DeepSeek-R1", "meta-llama/Llama-3.3-70B-Instruct",
              "Qwen/Qwen3-30B-A3B-Instruct"]

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": m} for m in FakeProvider.models]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        key = self.headers.get("Authorization", "")
        FakeProvider.calls.append((self.path, key, body["model"]))
        plan = FakeProvider.script.get((key, body["model"]), [])
        status = plan.pop(0) if plan else 200
        if status != 200:
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"error":"scripted"}')
            return
        out = {"choices": [{"message": {"content": '<think>checking</think>{"ok": true}'}}],
               "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeProvider)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/v1"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeProvider.calls = []
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["KEY_A"] = "bad"
        os.environ["KEY_B"] = "good"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failover_across_providers_and_models(self):
        FakeProvider.script = {
            ("Bearer bad", "a-model"): [401],
            ("Bearer good", "too-big"): [403],          # plan can't run this model
            ("Bearer good", "gone"): [404],             # model not hosted
            ("Bearer good", "works"): [429, 200],       # rate limited once, then fine
        }
        tracer = Tracer(self.tmp, live=False)
        llm = LLM([Provider("a", self.base, "KEY_A", ["a-model"]),
                   Provider("b", self.base, "KEY_B", ["too-big", "gone", "works"])],
                  tracer, retries=2, backoff=0.01)
        reply = llm.chat([{"role": "user", "content": "hi"}], name="t")
        self.assertEqual(extract_json(reply), {"ok": True})
        span = tracer.spans[-1]
        self.assertEqual(span.attrs["provider"], "b")
        self.assertEqual(span.attrs["model"], "works")
        self.assertEqual([a["status"] for a in span.attrs["attempts"]], [401, 403, 404, 429, "ok"])
        self.assertEqual(span.attrs["usage"]["total_tokens"], 17)

        # Dead provider/models are remembered, so the next call goes straight to the working model.
        FakeProvider.calls = []
        llm.chat([{"role": "user", "content": "again"}], name="t")
        self.assertEqual([c[2] for c in FakeProvider.calls], ["works"])

    def test_discovers_a_model_when_configured_ones_are_gone(self):
        FakeProvider.script = {("Bearer good", "renamed-model"): [404]}
        tracer = Tracer(self.tmp, live=False)
        p = Provider("b", self.base, "KEY_B", ["renamed-model"],
                     prefer=["Qwen3", "DeepSeek-V", "Llama"], avoid=["embed", "-R1"])
        self.assertEqual(p.rank(FakeProvider.models),
                         ["Qwen/Qwen3-30B-A3B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"])
        llm = LLM([p], tracer, retries=0, backoff=0.01)
        llm.chat([{"role": "user", "content": "hi"}], name="t")
        span = tracer.spans[-1]
        self.assertEqual(span.attrs["model"], "Qwen/Qwen3-30B-A3B-Instruct")
        self.assertEqual([a["model"] for a in span.attrs["attempts"]],
                         ["renamed-model", "Qwen/Qwen3-30B-A3B-Instruct"])
        self.assertIn("discovered", span.attrs)

    def test_all_fail_raises_with_attempts_logged(self):
        FakeProvider.script = {("Bearer good", "m"): [500, 500, 500]}
        os.environ.pop("KEY_A")
        tracer = Tracer(self.tmp, live=False)
        llm = LLM([Provider("a", self.base, "KEY_A", ["x"]), Provider("b", self.base, "KEY_B", ["m"])],
                  tracer, retries=2, backoff=0.01)
        with self.assertRaises(LLMError):
            llm.chat([{"role": "user", "content": "hi"}], name="t")
        span = tracer.spans[-1]
        self.assertEqual(span.status, "error")
        self.assertEqual(len(span.attrs["attempts"]), 3)
        self.assertTrue(load_trace(self.tmp / "trace.jsonl")[-1]["error"])


if __name__ == "__main__":
    unittest.main()
