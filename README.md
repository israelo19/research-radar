# Research radar

Every morning this pulls new arXiv papers and Hacker News posts about MCP security, AI worms, eval awareness and agent privilege escalation. An open-weight model on Bitdeer screens each one against my research threads and writes a short reading list. It runs on GitHub Actions, so my laptop can be closed.

Every step is traced. I can watch a run live in the Actions log, and afterwards click into any model call to see the exact prompt, the exact reply, which model answered, how long it took, tokens used and every failed attempt before it.

## Set up

1. **Add the Bitdeer key** in this repo: Settings → Secrets and variables → Actions → New repository secret. Name it `BITDEERAI_API_KEY` and paste the key.
2. **That's it.** The workflow runs by itself every day at 6:23 AM Eastern, and also right after any change to the code or `config.toml`.

Featherless is set up as a backup but stays off unless you add a `FEATHERLESS_API_KEY` secret, so those credits stay free for other projects.

To get the reading list on your phone, install the GitHub app and turn on notifications for this repo. Any day with something scoring 6 or higher opens an issue with the list.

### Running it by hand

1. Open the repo on GitHub and click the **Actions** tab at the top.
2. In the left sidebar, click **radar**.
3. On the right side of the page, above the list of runs, click the gray **Run workflow** button. A small menu opens.
4. Leave the branch as `main`. Pick a mode: **run** for a normal run, or **check** to test the key and list every model Bitdeer serves. Then click the green **Run workflow** button in that menu.
5. Refresh. Click the new run, then the **radar** job, then expand the step to watch it live.

The **Run workflow** button only shows up once `.github/workflows/radar.yml` is on the `main` branch.

### Turn on the viewer link (once)

Settings → Pages → under "Build and deployment", set Source to **Deploy from a branch**, Branch to **main** and the folder to **/docs**, then Save. A minute later the viewer is live at https://israelo19.github.io/research-radar/ and updates after every run.

### If a run says no key reached it

The key has to be a **repository secret** named exactly `BITDEERAI_API_KEY`: Settings → Secrets and variables → Actions → **Secrets** tab → **Repository secrets** → New repository secret. A key saved under Environment secrets, Codespaces or Dependabot never reaches this workflow.

## Where to look

| What | Where |
|---|---|
| Watch a run as it happens | Actions → the running job → "Run radar" step. Each step prints when it starts and finishes. |
| Click into every step | **https://israelo19.github.io/research-radar/** once Pages is on (below). Or open `docs/index.html` from the repo, or download the `trace-viewer` artifact from a run. |
| Today's reading list | The issue it opens, the run's summary page, or `digests/YYYY-MM-DD.md` |
| Raw traces | `runs/<run id>/trace.jsonl`, one JSON object per step |

## Changing how it judges

The instructions the model reads are in **`prompt.md`**. Open it on GitHub, click the pencil icon, edit, and click "Commit changes". The next morning's run uses your version. To try it right away, use **Run workflow**. Only items it hasn't screened yet get judged with the new wording.

## Changing what it looks for

The rest is in `config.toml`:

- `[[threads]]` are my research threads. The model sorts each item into one of them, and they're filled into `prompt.md` where it says `{threads}`.
- `[sources.arxiv] phrases` and `[sources.hn] queries` decide what gets pulled in before any model sees it.
- `max_items` caps model calls per run, so a busy day can't burn through credits. Items past the cap wait for the next run.
- `concurrency` is how many model calls run at once. Keep it at 2 unless your plan allows more.
- `[[providers]]` lists Bitdeer first, then Featherless. If none of a provider's listed models work, it asks the provider for its model list and picks one using `prefer` and `avoid`. Each run's trace shows the full list.

## How it's built

No dependencies, just the Python standard library.

- `radar/trace.py`: spans with parent ids, printed live and written to `trace.jsonl`
- `radar/llm.py`: OpenAI-compatible client. It tries providers and models in order, retries rate limits with backoff, remembers dead models, falls back to the provider's own model list, and strips `<think>` blocks.
- `radar/sources.py`: arXiv API and HN Algolia fetchers
- `radar/job.py`: fetch → dedupe → screen in parallel → digest → issue
- `radar/viewer.py`: bakes recent runs into one self-contained HTML page

New jobs can reuse `Tracer` and `LLM` the same way `job.py` does, and every model call they make shows up in the viewer.

## Testing without the network

```bash
python -m unittest -v           # pipeline on sample data, plus failover against a fake provider
python -m radar run --dry-run   # full run on [sample] data with a stub model, written to .dry/
open .dry/docs/index.html
```
