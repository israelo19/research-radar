"""Command line: python -m radar run | check | view"""
from __future__ import annotations

import argparse
import sys

from . import job, viewer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m radar")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fetch, triage and write today's digest")
    r.add_argument("--dry-run", action="store_true", help="use built-in sample data and a stub model, no network")
    r.add_argument("--no-issue", action="store_true", help="don't open a GitHub issue")
    sub.add_parser("check", help="test your API keys and see which models answer")
    sub.add_parser("view", help="rebuild docs/index.html from saved runs")
    args = p.parse_args(argv)

    if args.cmd == "check":
        return job.check()
    if args.cmd == "view":
        print(viewer.build(job.ROOT))
        return 0

    fetch = None
    root = job.ROOT
    if args.dry_run:
        # Dry runs live in .dry/ so sample data never lands in the real history.
        from .fixtures import fixture_fetch
        fetch = fixture_fetch
        root = job.ROOT / ".dry"
    summary = job.run(root=root, cfg=job.load_config(), dry_run=args.dry_run, fetch=fetch,
                      open_issue=not args.no_issue)
    out = viewer.build(root)
    print(f"\nrun {summary['run_id']}: {summary['status']}"
          + (f" ({summary.get('error')})" if summary["status"] == "error" else
             f", {summary.get('screened', 0)} screened, {summary.get('relevant', 0)} relevant")
          + f"\nviewer: {out}")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
