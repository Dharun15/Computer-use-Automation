# Computer-Use Automation System

A backend integration layer that lets an AI agent operate legacy back-office
applications that expose no API: an LLM discovers how to accomplish a goal
by driving a real UI, and that discovery is turned into a deterministic,
parameterized, replayable artifact an AI agent can invoke in production
without the model in the loop again.

See `REPORT.md` for the design write-up (architecture, schema, error
handling, multi-tenant story, escalation, safety, cuts).

**Status: the required live LLM discovery run has been done.** A real
Gemini-driven session recorded `artifacts/saved/lookup_member_savings_balance.json`
against the live fake bank, and it has been replayed successfully against
different member IDs than it was recorded with -- see `/evidence/` and
Section 2 below to reproduce your own.

## 1. Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

No database, no external services. The target application (`app/`) is a
small local FastAPI app that runs entirely on your machine.

### Environment / keys

The only external credential this project uses is an LLM API key, needed
**only** for the discovery agent (`agent/discover.py`). Nothing else in the
system (the target app, the replay engine, the capability catalog, the
stability tool, all 109 automated tests) needs any key at all.

Two providers are supported out of the box (`ModelClient` is a swappable
`Protocol` -- adding a third would touch only `agent/llm_client.py`):

- **Anthropic** (`--provider anthropic`, the default): needs
  `ANTHROPIC_API_KEY`. New accounts get a one-time trial credit (a few
  dollars); after that it's pay-per-token.
- **Gemini** (`--provider gemini`): needs `GEMINI_API_KEY`. Google AI
  Studio's Flash-tier models have a genuinely free, no-credit-card tier --
  this is what the actual discovery run in this repo used. Get a key at
  https://aistudio.google.com/apikey. **Note:** Google renames/deprecates
  Gemini model strings fairly often. If `--model` gives you a 404 "no
  longer available" error, the response body names the exact replacement
  to use -- `gemini-3.5-flash-lite` is the current default as of this
  writing, but check the error if it's stale by the time you run this.

Setting the key depends on your shell:

```bash
# macOS/Linux/bash:
export ANTHROPIC_API_KEY=sk-ant-...
# or, for the free-tier path:
export GEMINI_API_KEY=...
```

```cmd
:: Windows cmd.exe (lasts for the current terminal session only):
set GEMINI_API_KEY=your-key-here
:: or, to persist across terminal sessions (close and reopen after running this):
setx GEMINI_API_KEY "your-key-here"
```

```powershell
# Windows PowerShell:
$env:GEMINI_API_KEY = "your-key-here"
```

Verify it actually took before running the agent:

```cmd
:: cmd.exe
echo %GEMINI_API_KEY%
```
```powershell
# PowerShell
echo $env:GEMINI_API_KEY
```

**Note on multi-line commands:** the multi-line examples below (using `\`
line continuations) are bash syntax. In `cmd.exe` use `^` instead, in
PowerShell use `` ` ``, or simplest: just put the whole command on one line.

### Running without live services

The full automated test suite runs entirely locally: it starts and stops
the fake bank app itself as part of the test fixtures, and uses a scripted
stand-in for the LLM (`tests/fakes.py`) rather than a real API call. You do
not need any API key, and you do not need to start anything yourself, to
run:

```bash
pytest tests/ -v
```

(109 tests, ~85s, as of this write-up.)

## 2. Demo path

### Step 1 -- start the target application

```bash
python -m uvicorn app.main:app --port 8000
```

Leave this running. It's a small "legacy" credit-union member-servicing
screen with a handful of synthetic members (no real PII) wired to specific
runtime conditions -- see `app/data.py`:

| Member ID | Scenario |
|---|---|
| `12345`, `23456`, `34567` | normal members, different balances |
| `99999` | does not exist -- business outcome |
| `abc` (bad format) | validation error -- business outcome |
| `55555` | permission denied (403) -- hard failure |
| `00000` | application error (500) -- hard failure |
| `66666` | slow-loading page (3s) -- recoverable |
| `77777` | unexpected confirmation dialog -- human intervention |

### Step 2 -- run the discovery agent (real LLM, in a second terminal)

```bash
python -m agent.discover \
  --goal "Look up member 12345 and return their savings balance." \
  --provider gemini --model gemini-3.5-flash-lite \
  --record-artifact \
  --artifact-id lookup_member_savings_balance \
  --artifact-name "Lookup Member Savings Balance" \
  --param member_id=12345
```

(swap `--provider gemini --model gemini-3.5-flash-lite` for
`--provider anthropic` to use Claude instead)

This launches a real Chromium session, gives the model the six
computer-use tools, and lets it actually drive the page -- no scripted
steps. On success it records the run as a reusable artifact at
`artifacts/saved/lookup_member_savings_balance.json`, and writes structured
evidence (the full step log and the raw LLM transcript, kept as two
separate files on purpose) under `evidence/discovery/<run>/`.

### Step 3 -- replay the artifact deterministically (no LLM, ever)

```bash
python -m replay \
  --artifact artifacts/saved/lookup_member_savings_balance.json \
  --input member_id=23456
```

Note this uses a **different** member ID than was recorded -- that's the
actual point of the artifact. Try the other member IDs above too:

```bash
# business outcome, not a crash
python -m replay --artifact artifacts/saved/lookup_member_savings_balance.json --input member_id=99999

# hard failure (permission denied)
python -m replay --artifact artifacts/saved/lookup_member_savings_balance.json --input member_id=55555

# recoverable (slow load, well within the default timeout)
python -m replay --artifact artifacts/saved/lookup_member_savings_balance.json --input member_id=66666
```

Each replay writes evidence to `evidence/replay/<run>/`.

### Step 4 -- the full human-intervention cycle (member 77777)

The `replay` CLI can't demonstrate pause-then-resume by itself (each
invocation opens a browser, does one thing, and closes it -- but human
intervention needs the SAME session to stay open across the pause). Use
the standalone demo script instead, which keeps one browser session alive
for the whole pause -> human acts -> resume sequence:

```bash
python scripts/demo_human_intervention.py \
  --artifact artifacts/saved/lookup_member_savings_balance.json \
  --member-id 77777
```

It will pause, print the intervention context (reason, failing step,
screenshot path), wait for you to press Enter, then resume and complete.

### Step 5 -- multi-run stability

```bash
python -m replay \
  --artifact artifacts/saved/lookup_member_savings_balance.json \
  --input member_id=12345 --stability-runs 5
```

Reports a success-rate/timing signal across N replays. Note: this
deterministic fake bank will always show 100% (or 0%) consistency -- see
REPORT.md Section 7 for why that's an honest property of the target app,
not a limitation of the tool.

### Step 6 -- the agent-facing capability catalog

```bash
uvicorn capabilities.service:app --port 8002
curl http://localhost:8002/capabilities
curl -X POST http://localhost:8002/capabilities/lookup_member_savings_balance/invoke \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"member_id": "34567"}}'
```

This is the concrete, callable shape of "an AI agent invokes it in
production" -- `/invoke` runs the real `ReplayEngine`, no LLM involved.

### Regenerating the illustrative /evidence/ bundle

`scripts/generate_demo_evidence.py` reproduces a full discovery + replay
evidence set in one shot (with the target app already running):

```bash
PYTHONPATH=. python scripts/generate_demo_evidence.py
```

**Read the module docstring first.** Its discovery run uses a scripted
stand-in for the LLM (clearly marked `SIMULATED_NOT_A_REAL_LLM_RUN.txt` in
its own output folder) so the pipeline's shape can be demonstrated without
an API key -- it is NOT a substitute for the real run from Step 2 above.
Its replay evidence is completely real (replay never touches an LLM). If
you run this after already having a real artifact from Step 2, delete
the resulting simulated discovery folder afterward so it doesn't sit next
to your real one.

## 3. Mock operator console (Phase 9 handoff)

A minimal, intentionally bare HTTP surface for a human operator to resolve
a paused intervention -- demonstrates the `HandoffManager` API's shape
over HTTP. Note this is a separate demonstration from Step 4 above: this
console is not wired to a live browser session (see REPORT.md Section 7);
the actual pause/resume-on-the-same-session cycle is what
`scripts/demo_human_intervention.py` shows.

```bash
uvicorn handoff.routes:app --port 8001
curl http://localhost:8001/intervention/pending
curl -X POST http://localhost:8001/intervention/resolve \
  -H "Content-Type: application/json" \
  -d '{"action_description": "Accepted the confirmation dialog."}'
```

See `REPORT.md` Section 5 for how the underlying mechanism ties into the
replay engine's pause/resume.

## 4. Project layout

```
app/            fake bank target application (Phase 1)
surface/        ComputerSurface protocol + PlaywrightSurface (Phase 2)
tools/          Pydantic tool schemas + ToolExecutor (Phase 3)
agent/          DiscoveryAgent, LLM clients (Anthropic + Gemini), discovery CLI (Phase 4)
artifacts/      Artifact schema, recorder, storage (Phase 5)
replay/         ReplayEngine, error taxonomy, replay CLI, stability tool (Phase 6-7, stretch)
safety/         Allowlist + risky-action policy engine (Phase 8)
handoff/        Human-in-the-loop pause/resume + mock console (Phase 9)
capabilities/   Agent-facing capability catalog + invoke endpoint (stretch)
evidence/       Discovery + replay run evidence, an example artifact
scripts/        Reproducible demo-evidence generator + human-intervention demo
tests/          109 automated tests covering every phase above
```