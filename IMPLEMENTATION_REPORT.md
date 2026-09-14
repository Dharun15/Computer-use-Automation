# Implementation Report — Computer-Use Automation System

A complete account of what was built, phase by phase and file by file, for
the interface.ai take-home: a backend integration layer that lets an AI
agent operate legacy back-office applications with no API — an LLM
discovers a workflow once by driving the real UI, and that discovery
becomes a deterministic, parameterized, replayable artifact an AI agent
can invoke afterward without the model in the loop again.

**Scale:** 45 Python files, ~4,700 lines of implementation + test code,
88 automated tests (all passing), 13 phases.

**Status of the one thing that has to be genuinely real:** the discovery
agent's live LLM run has NOT yet been executed with a real Anthropic API
key (none was available during development). Everything else — including
every test, the full replay engine, safety, and handoff — is real and
verified. See the "What's real vs. simulated" section at the end.

---

## Phase 1 — Target Application (`app/`)

**Purpose:** a small, deliberately legacy-looking local web app to serve
as the proxy target, standing in for a real bank's back-office screen that
has no API.

| File | What's in it |
|---|---|
| `app/main.py` | FastAPI routes: `GET /` (search form), `POST /search` (validates + looks up), `GET /members/{id}` (details page, branches on the member's failure flags), `GET /healthz`. |
| `app/data.py` | 7 synthetic `Member` records (no real PII) as plain dataclasses, each with boolean behavior flags (`slow_load`, `permission_denied`, `confirm_dialog`, `app_error`) that drive the failure-mode routes. `is_valid_member_id_format()` enforces a 5-digit rule for validation-error testing. |
| `app/templates/*.html` | Six Jinja2 templates: `search.html`, `search_results.html`, `not_found.html`, `validation_error.html`, `member_details.html`, `permission_denied.html`, `app_error.html`. All table-based layout, no `data-testid`/`id`/`<label for>` anywhere — intentionally legacy-hostile. |

**Deliberate design choices:**
- **Server-rendered, full-page navigations only** — no SPA/client routing, matching real legacy banking UIs.
- **A disabled, unlabeled decoy "Member ID" textbox** on the search page, not part of the real form — forces locator resolution to disambiguate by context rather than grabbing the first visual match.
- **Seven wired scenarios**, one per requirement in the brief's error taxonomy: `12345`/`23456`/`34567` (happy path, different balances), `99999` (not found), `abc`-style bad input (validation error), `55555` (403 permission denied), `00000` (500 app error), `66666` (3s server-side delay), `77777` (native JS `confirm()` dialog gating the balance data).

---

## Phase 2 — Surface Abstraction (`surface/`)

**Purpose:** the entire boundary between "the agent" and "however we
actually drive this particular application." Nothing above this layer
knows Playwright, or even a browser, exists.

| File | What's in it |
|---|---|
| `surface/observation.py` | Pure data models (Pydantic), no browser code: `Target` (the 4-tier locator vocabulary — role+name, label, text, css, plus an explicit `nth` disambiguator), `Control`, `Observation` (semantic page snapshot, not raw HTML — includes `.to_prompt_text()` for LLM consumption), `ActionStatus` enum, `ActionResult`. |
| `surface/base.py` | `ComputerSurface` — a `Protocol` (structural typing) with exactly 6 methods: `observe`, `click`, `type`, `navigate`, `extract`, `screenshot`, `close`. |
| `surface/playwright_surface.py` | `PlaywrightSurface` — the only file in the whole project that imports Playwright. Implements tiered locator resolution (including a custom row-label heuristic for the label-less legacy tables, with a documented colon-suffix tolerance), in-page JS for building the control list and text summary, native-dialog capture (auto-dismiss + report once), and visibility-aware extraction (a hidden match reports `NOT_FOUND`, not an empty string). |

**Real bugs found and fixed during this phase** (caught by tests, not shipped silently):
1. The row-label heuristic originally called `.first`, which picked Phase 1's disabled decoy field over the real one — fixed by excluding disabled elements (a principled actionability filter) and letting genuine ties surface as `AMBIGUOUS`.
2. Label text mismatch: the real form's label is `"Member ID:"` (colon), the decoy's is `"Member ID"` (no colon) — added a documented colon-tolerance in the XPath match.
3. `extract()` on hidden content (behind the dismissed dialog) originally returned an empty string with `status=OK` — fixed to check visibility first and report `NOT_FOUND`.

---

## Phase 3 — Tool Layer (`tools/`)

**Purpose:** structured, Pydantic-validated tool calls between "the LLM
said what it wants to do" and "the surface does it."

| File | What's in it |
|---|---|
| `tools/definitions.py` | 6 Pydantic models (`ObserveCall`, `ClickCall`, `TypeCall`, `NavigateCall`, `ExtractCall`, `ScreenshotCall`), combined into a discriminated union `ToolCall` keyed on `action`. `ANTHROPIC_TOOL_SPECS`: the same 6 actions in the exact JSON-schema shape Claude's Messages API tool-calling expects. |
| `tools/executor.py` | `ToolExecutor.execute(action, raw_input, confirmed=False)` — validates via the discriminated union first (bad/missing fields never reach the browser), then (from Phase 8 on) checks a `PolicyEngine` if one was supplied, then dispatches to whichever `ComputerSurface` it was constructed with. Returns one uniform, JSON-serializable `ToolExecutionResult` no matter what happens — nothing throws up to the caller. |

---

## Phase 4 — Discovery Agent (`agent/`)

**Purpose:** the actual `observe → LLM decides → act → observe` loop.

| File | What's in it |
|---|---|
| `agent/state.py` | `DiscoveryState` (goal, status, step count, `steps: list[RecordedStep]`), `DiscoveryStatus` enum (`RUNNING/SUCCESS/FAILED/MAX_STEPS_EXCEEDED/TIMEOUT/DEAD_END`), `DiscoveryResult` (bundles `state` + the raw `transcript` as two deliberately separate fields), and normalized `LLMTextBlock`/`LLMToolUseBlock` so the loop never depends on the Anthropic SDK's own types. |
| `agent/llm_client.py` | `ModelClient` Protocol (one method: `complete(messages, tools, system)`), `AnthropicModelClient` (the real implementation — wraps `anthropic.Anthropic().messages.create()`), and `FINISH_TOOL_SPEC` (the `finish_task` control-flow tool — not a surface action, intercepted directly by the agent loop, never seen by `ToolExecutor`). |
| `agent/prompts.py` | The system prompt: role, tool usage rules, explicit instruction that business outcomes (not found / bad input) are valid endpoints, not failures to retry against. |
| `agent/agent.py` | `DiscoveryAgent.run(goal, start_url)` — the loop itself. Seeds with a real navigate+observe (recorded as step 1, not just bookkeeping — see the fix below), enforces `max_steps`/`max_runtime_seconds`/`max_consecutive_failures`, auto-attaches fresh page state after each action so the model doesn't waste a turn re-observing, nudges (rather than silently accepting) a turn with no tool call. |
| `agent/recorder.py` | `save_discovery_evidence()` — writes `state.json`, `transcript.json`, `summary.txt` to a uniquely-named run folder under `evidence/discovery/`. |
| `agent/discover.py` | CLI entry point: `python -m agent.discover --goal "..." [--record-artifact ...]`. This is the one script that must be run with a real `ANTHROPIC_API_KEY`. |

**A real gap found and fixed while building Phase 5:** the agent's initial navigation to the start URL happened *before* the LLM loop and was never added to `state.steps` — meaning a replay artifact would have no idea where to start. Fixed by recording it as a real step; this shifted several Phase 4 test assertions by exactly 1 (step counts), all re-verified.

---

## Phase 5 — Artifact Recording (`artifacts/`)

**Purpose:** turn a successful `DiscoveryState` into a typed, versioned,
parameterized, reusable capability — never a dump of the LLM transcript.

| File | What's in it |
|---|---|
| `artifacts/schema.py` | `Artifact` (id, name, description, `surface` info, typed `inputs`/`outputs`, `steps`, `checkpoint`). `ArtifactStep` is a discriminated union (`ArtifactClickStep`/`ArtifactTypeStep`/`ArtifactNavigateStep`/`ArtifactExtractStep`) — same pattern as `tools/definitions.py`, so a `type` step without a `value` fails validation immediately. Model validators reject an artifact with zero steps, or a step referencing an output key `outputs` never declared. |
| `artifacts/recorder.py` | `record_artifact(state, artifact_id, name, parameters, application, ...)` — reads **only** `state.steps`, never the transcript (enforced by the function signature, not just convention). Parameterizes by literal string substitution (`"12345"` → `"{{member_id}}"` everywhere it appears). Recovers output names automatically by matching each `extract` step's actual value against `finish_task`'s declared outputs. Derives the checkpoint from the final page's title. Refuses to run on anything but a `SUCCESS` state. |
| `artifacts/storage.py` | `save_artifact()` / `load_artifact()` — plain JSON on disk, full Pydantic re-validation on load (a corrupted or hand-edited file fails loudly and specifically before replay ever touches it). |

---

## Phase 6/7 — Replay Engine & Error Taxonomy (`replay/`)

**Purpose:** the production execution path — deterministic, no LLM,
with a full error taxonomy (added in Phase 7, on top of Phase 6's
mechanics).

| File | What's in it |
|---|---|
| `replay/locator.py` | Template substitution — the inverse of the recorder's parameterization. `substitute_string`/`substitute_target` turn `{{member_id}}` back into a real value at replay time; raises clearly if a referenced input was never supplied. |
| `replay/checkpoints.py` | `verify_checkpoint()` — the three checkpoint strategies (`text_present`/`url_contains`/`status_code`), returning a `CheckpointResult` with both `expected` and `observed` for debugging. |
| `replay/classifier.py` | `classify_failure(observation, raw_message)` — the heart of the error taxonomy. Looks at the actual page reached after a failure (not just the raw action status) and matches it against known patterns to decide `BUSINESS_OUTCOME` / `HARD_FAILURE` / `HUMAN_INTERVENTION`. Extending this dict is how a new legacy screen's "not found" pattern gets recognized — a one-line change, not a control-flow change. |
| `replay/errors.py` | `ReplayStatus` (`SUCCESS`/`BUSINESS_OUTCOME`/`HUMAN_INTERVENTION`/`HARD_FAILURE`), `OutcomeCode` enum (`MEMBER_NOT_FOUND`, `INVALID_INPUT`, `PERMISSION_DENIED`, `APPLICATION_ERROR`, `UNRESOLVED_DIALOG`, `RETRY_BUDGET_EXHAUSTED`, `CHECKPOINT_NOT_SATISFIED`, `UNKNOWN`), `RecoveryAttempt`, `ReplayResult` (outputs, steps completed, failed step id, expected/observed, recovery attempts, screenshot path, intervention). |
| `replay/engine.py` | `ReplayEngine.replay(artifact, inputs)` / `.resume(artifact, inputs, paused_result)`. Validates inputs against declared types before touching the browser. Executes steps through the same `ToolExecutor` discovery used. Bounded automatic retries (with backoff) for transient `timeout` failures, logged in `recovery_attempts` whether or not they mattered. On failure, classifies via `replay/classifier.py`, captures a screenshot, and — if a `HandoffManager` was supplied — actually requests a human intervention. `resume()` continues from exactly `paused_result.steps_completed` with `paused_result.outputs` already populated, never restarting. |
| `replay/evidence.py` | `save_replay_evidence()` — mirrors the discovery evidence shape (`result.json`, `inputs.json`, `summary.txt`) under a uniquely-named run folder in `evidence/replay/`. |
| `replay/__main__.py` | CLI entry point: `python -m replay --artifact ... --input name=value`. 100% real — replay never needs an LLM, so this path has no simulation caveat at all. |

**Design note on classification:** even a checkpoint failure after every individual step reported `ok` gets re-classified the same way a mid-step failure would (the final page might still match a known business-outcome/hard-failure pattern) before falling back to a generic `CHECKPOINT_NOT_SATISFIED`.

---

## Phase 8 — Safety & Policy (`safety/`)

**Purpose:** every action, from either discovery or replay, passes
through the same guardrail before touching the surface.

| File | What's in it |
|---|---|
| `safety/policy.py` | `SafetyPolicy` (allowed domains, blocked path prefixes, allowed action types, risky-name keyword patterns), `PolicyEngine.check(action, call)` → `PolicyResult` (`ALLOW`/`BLOCK`/`REQUIRE_CONFIRMATION`). Domain allowlisting is fail-closed (an empty list allows nothing by absolute URL; relative paths are always allowed since they stay within whatever app the surface is already scoped to). Risky-action detection is keyword-based against the target's name/label/text/value (and, for `type`, the value being typed), defaulting to `REQUIRE_CONFIRMATION` rather than an outright block — an unattended run can't supply that confirmation itself (`ToolExecutor.execute(..., confirmed=True)` is the only way past it). |

**Integration:** `ToolExecutor` accepts an optional `policy: PolicyEngine`; `agent.py`'s dead-end failure-status set now includes `blocked`/`requires_confirmation`, so a model repeatedly attempting a gated action correctly trips dead-end detection instead of looping.

---

## Phase 9 — Human-in-the-Loop Handoff (`handoff/`)

**Purpose:** a real pause/resume mechanism — automation and a human share
ONE live session; nothing here ever opens a second browser for the human.

| File | What's in it |
|---|---|
| `handoff/manager.py` | `SessionOwner` enum (`AUTOMATION`/`HUMAN`), `InterventionRequest`, `HumanAction`, `HandoffManager` — `request_intervention()` (flips ownership, records context/screenshot/goal/step), `record_human_action()`, `resume()` (flips ownership back, returns the resolved request). |
| `handoff/routes.py` | A deliberately bare FastAPI mock operator console (per the brief's own scope note): `GET /intervention/pending`, `POST /intervention/resolve`, `GET /intervention/history` — a thin wrapper over the same `HandoffManager`. |

**Wired into `replay/engine.py`:** when `classify_failure` returns `HUMAN_INTERVENTION` and a `HandoffManager` was supplied, the engine actually calls `request_intervention()` with the real goal/step/screenshot context — this isn't just a status code, it's a tracked, resolvable request. Tested end to end: replay pauses on the dialog-gated member → a human resolves it on the same live Playwright session → `resume()` picks up from exactly where it stopped and completes.

---

## Phase 10 — Evidence (`evidence/`, `scripts/`)

| File | What's in it |
|---|---|
| `scripts/generate_demo_evidence.py` | Reproducible script that regenerates the entire `/evidence/` bundle: one discovery run (explicitly marked simulated — see below) plus four replay runs (success with a different member than recorded, business outcome, paused human intervention, and the resumed completion). |
| `evidence/discovery/<run>/` | `state.json`, `transcript.json`, `summary.txt`, and `SIMULATED_NOT_A_REAL_LLM_RUN.txt` (see caveat below). |
| `evidence/replay/<run>/` × 4 | `result.json`, `inputs.json`, `summary.txt` per run — these are 100% real. |
| `evidence/_screenshots/` | Two screenshots captured automatically on the business-outcome and human-intervention replay failures. |
| `artifacts/saved/lookup_member_savings_balance.json` | The example artifact these evidence runs were produced from. |

A real bug was caught and fixed here: the evidence writers originally timestamped run folders to second-resolution, so four replay runs generated within the same second silently overwrote each other's evidence. Fixed by adding a short UUID suffix to guarantee uniqueness (`agent/recorder.py` and `replay/evidence.py` both updated).

---

## Phase 11 — Tests (`tests/`)

88 tests across 8 files, all passing (~35–40s total). No test requires a
live API key; the LLM is always replaced by `tests/fakes.py`'s
`ScriptedModelClient` where a model is needed at all.

| File | Tests | What it covers |
|---|---|---|
| `tests/conftest.py` | — | Shared fixtures: starts/stops the fake bank as a real subprocess (`fake_bank_url`), provides a fresh `PlaywrightSurface` per test (`surface`). |
| `tests/fakes.py` | — | `ScriptedModelClient` (deterministic stand-in for the LLM), `tool_use()`/`text_only()` helpers. |
| `tests/test_surface.py` | 11 | The full happy-path scenario end to end, every business-outcome/failure page, the confirm-dialog capture, slow-load timing, ambiguous-target refusal, clean not-found errors. |
| `tests/test_tools.py` | 9 | Tool-call validation gates (missing fields, unknown actions, empty targets), full lookup flow via raw tool-call dicts, not-found/ambiguous status pass-through. |
| `tests/test_agent.py` | 6 | Full discovery success, business-outcome finish, dead-end detection (stops before exhausting budget), max-steps enforcement, the text-only nudge, zero-runtime timeout. |
| `tests/test_artifacts.py` | 8 | Schema validation (missing fields, undeclared outputs, empty steps), full save/load round-trip, recording a real discovery run into a correctly parameterized artifact, refusing to record a failed run. |
| `tests/test_replay.py` | 25 | Parameterization actually generalizing to a different member, numeric output coercion, missing/wrong-type input validation, all four taxonomy outcomes (success/business-outcome/hard-failure/human-intervention) against the real app, checkpoint logic in isolation, template substitution in isolation, retry mechanism in isolation (scripted executor double). |
| `tests/test_safety.py` | 17 | Domain/path allowlisting (relative, absolute, subdomain, blocked-prefix, fail-closed), action allowlisting, risky-target detection (case-insensitive, custom patterns), full `ToolExecutor` integration including the confirm-then-proceed flow. |
| `tests/test_handoff.py` | 12 | `HandoffManager` lifecycle in isolation, full replay-engine integration (a real `InterventionRequest` gets created), the complete pause→human-acts→resume cycle, the mock FastAPI console via `TestClient`. |

---

## Phase 12 — Documentation

- **`README.md`** — setup, the exact demo commands (discover → replay), the mock operator console, project layout.
- **`REPORT.md`** — the seven required headings: Architecture, Artifact schema, Determinism & error handling, Heterogeneity & multi-tenant, Escalation & handoff, Safety, Cuts.

---

## Phase 13 — Stretch Goals

**Deliberately skipped**, per the brief's own guidance ("we do not reward
feature breadth... a small, correct, well-argued system is the goal").
Documented in `REPORT.md`'s Cuts section, with the agent-facing capability
catalog (a thin HTTP surface over `artifacts/storage.py` + `ReplayEngine`)
named as the smallest-effort, clearest-payoff option if one were picked up
next.

---

## What's real vs. simulated — read this before submitting

| Path | Status |
|---|---|
| Every automated test (88/88) | **Real.** Real Playwright, real FastAPI app, real HTTP. LLM replaced by a scripted stand-in — that's a deliberate testing choice, not a limitation of the underlying code. |
| `replay/` (the entire replay engine, taxonomy, retries, handoff integration) | **Real, end to end.** Replay never needs an LLM at all — nothing here has ever been simulated. |
| `evidence/replay/` (4 runs in the shipped evidence bundle) | **Real.** |
| `evidence/discovery/` (1 run in the shipped evidence bundle) | **Simulated.** Uses `ScriptedModelClient`, not a real Anthropic API call — because no API key was available during development. Self-labeled with `SIMULATED_NOT_A_REAL_LLM_RUN.txt` in its own folder. Exists only to prove the evidence format and the discovery→artifact pipeline work. |
| `agent/llm_client.py`'s `AnthropicModelClient` | **Real code, never run against the live API.** Constructs correctly, imports correctly, matches the Anthropic SDK's actual request/response shapes — but has not yet made a real network call. |

**Before this can be submitted**, a genuine discovery run is required —
this is the one thing the brief explicitly says can't be faked or
described around ("At least one genuine LLM-driven run against a live
surface, with the evidence in `/evidence/` to show it happened. That's the
heart of the project and we can't assess a description of it."):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m uvicorn app.main:app --port 8000 &
python -m agent.discover \
  --goal "Look up member 12345 and return their savings balance." \
  --model claude-haiku-4-5-20251001 \
  --record-artifact --artifact-id lookup_member_savings_balance \
  --artifact-name "Lookup Member Savings Balance" --param member_id=12345
```
