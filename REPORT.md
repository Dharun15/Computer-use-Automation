# Design Write-Up

## 1. Architecture

```
Goal + start_url
      |
      v
DiscoveryAgent  <-- ModelClient (Protocol: AnthropicModelClient or
      |                          GeminiModelClient, or a scripted fake
      |                          in tests -- swappable with zero change
      |                          to the loop itself)
      | tool calls (click/type/navigate/extract/screenshot/finish_task)
      v
ToolExecutor  <-- PolicyEngine (optional; allowlist + risky-action gate)
      | validated (action, input)
      v
ComputerSurface (Protocol)
      |
      v
PlaywrightSurface  ---drives--->  Browser  --->  app/ (fake bank)

Successful DiscoveryState.steps
      |
      v
artifacts.recorder.record_artifact()  --->  Artifact (JSON, versioned)
      |
      v
ReplayEngine  <-- same ToolExecutor/ComputerSurface stack, no LLM
      |
      +--> SUCCESS / BUSINESS_OUTCOME / HARD_FAILURE
      +--> HUMAN_INTERVENTION ---> HandoffManager (pause/resume, same session)

Saved artifacts also served via:
  capabilities/service.py  -- GET /capabilities, POST /capabilities/{id}/invoke
  replay/stability.py      -- N-run success-rate/timing report
```

The one rule that shaped every other decision: **the agent never imports
Playwright, and the replay engine never imports an LLM.** Both depend only
on `ComputerSurface` (a `Protocol`, not a base class) and `ToolExecutor`.
This is why the same six actions, the same Pydantic validation, and the
same locator-resolution logic serve both discovery (LLM decides, in real
time) and replay (artifact decides, ahead of time) -- one execution path,
two different sources of the action sequence.

**Why Playwright.** Chromium gives a real, controllable accessibility tree
and a good enough approximation of "what a legacy web app looks like" via
Phase 1's deliberately hostile HTML (tables, no `data-testid`, no
`<label>`, a disabled decoy field). It also generalizes cleanly to a
"legacy web app" without changing anything upstream of `surface/` -- see
Section 4.

**Why one agent, not a planner/critic/browser split.** The brief explicitly
rewards coherent simplicity over framework breadth. A single
observe-decide-act loop with hard step/time/failure budgets was sufficient
to handle every required scenario (happy path, business outcomes,
timeouts, dead ends) without needing separate reasoning stages.

**Why no RAG, no LangChain/LangGraph.** This is a UI-interaction problem,
not a retrieval problem -- there is no corpus to search. A hand-rolled loop
over native tool-calling was simpler to reason about and debug than
adopting an agent framework for six tools.

**Why two LLM providers.** `ModelClient` is a `Protocol`, so
`AnthropicModelClient` and `GeminiModelClient` are two interchangeable
implementations of the same contract, with zero change to `agent/agent.py`.
Gemini was added specifically because Google's Flash-tier models have a
genuinely free, no-credit-card tier -- the brief leaves "LLM provider" as
an explicit implementation choice, and this is that choice exercised for
real, not just designed for.

**Discovery vs. replay, concretely.** `DiscoveryState.steps` (structured,
typed) and `DiscoveryResult.transcript` (raw LLM messages) are two
different fields on two different objects the type system keeps apart.
`artifacts/recorder.py`'s function signature doesn't even accept a
transcript argument -- the artifact cannot be built from the conversation,
structurally, not just by convention.

**What the real discovery run actually surfaced.** Running a live model
(Gemini) against this system, rather than only a scripted stand-in, found
three real gaps no amount of scripted-client testing had caught: (1) the
recorder was including every executed step, including a failed attempt
the model correctly retried past -- fixed by recording only steps with
`result_status == "ok"`; (2) the model initially satisfied `finish_task`'s
outputs by reading a value out of page text rather than calling `extract`
on it, producing an artifact with no way to reproduce that output on
replay -- fixed with an explicit system-prompt rule plus a loud
`warnings.warn` in the recorder if it ever happens again; (3) the model's
first successful `extract` targeted the literal balance figure itself
(`text: "$4250.00"`) rather than its stable label, which "worked" for the
one record it was recorded against and would fail on every other one --
fixed by tightening both the `extract` tool's description and the system
prompt to explicitly forbid targeting by the value being read. All three
are now covered by regression tests. This is the clearest evidence in the
whole project for why the brief insists the discovery run be genuinely
real rather than described.

## 2. Artifact schema

An `Artifact` is `artifact_id`, `name`, `description`, `surface` (which
app/vendor/version), `inputs` (typed, named), `outputs` (typed, named),
`steps` (ordered), and a `checkpoint`. `ArtifactStep` is a Pydantic
discriminated union on `action` (`click`/`type`/`navigate`/`extract`),
mirroring the same pattern `tools/definitions.py` uses for LLM tool calls --
a `type` step without a `value`, or an `extract` step without an `output`
key, fails validation immediately, not mid-replay. A model validator also
rejects a step that references an output key `outputs` never declared.

Each step's target reuses `surface.observation.Target` directly -- the
same tiered locator vocabulary (role+name -> label -> text -> css) the
surface already speaks, so there is no separate "artifact locator" concept
to keep in sync.

**Parameterization is literal-substitution, not inference.** The caller
recording an artifact declares which concrete value used during discovery
("12345") should become which named input (`member_id`); every occurrence
of that literal anywhere in a step (a target's name/label/text/css, a
typed value, a URL) becomes `{{member_id}}`. This is reviewable -- a human
reading the JSON can see exactly which literal became which parameter --
at the cost of being unable to parameterize a value the caller didn't know
to name. **Output names are recovered automatically** by matching each
`extract` step's actual value against `finish_task`'s declared outputs
(`{"balance": "$4250.00"}` tells the recorder that *this* extract step is
`balance`). Known limitation: two extract steps producing identical values
would collide -- see Cuts.

**Why a title-based checkpoint by default.** `ArtifactCheckpoint` supports
`text_present` / `url_contains` / `status_code`; the recorder defaults to
asserting the final page's title, which is simple, works across every
scenario tested, and is easy for a human reviewer to sanity-check at a
glance. A production system would likely want multiple checkpoints (one
per step, not just the end) -- noted in Cuts.

## 3. Determinism & error handling

Replay executes the artifact's steps in fixed order through the same
`ToolExecutor`/`ComputerSurface` stack discovery used -- there is no
decision point where an LLM (or randomness) could take a different path
than last time. Locator resolution is the same tiered strategy in both
directions: role+name is tried first, and an ambiguous match (more than
one element, no explicit `nth`) is a reported failure, never a silent
`.first()` guess -- this exact bug (a disabled decoy field winning a race
against the real one) was caught by Phase 2's own tests and fixed by
excluding non-actionable elements from the row-label heuristic rather than
by picking a "probably right" answer.

**The taxonomy** (`replay/errors.py`, `replay/classifier.py`) has four
terminal states:

- `SUCCESS` -- every step reported `ok` and the checkpoint held.
- `BUSINESS_OUTCOME` -- a real, expected result the caller needs (member
  not found, invalid input format). Classified by matching the *page
  actually reached* after a failing step against known patterns (page
  title contains "No Records", "Input Error", etc.) -- not by the raw
  action status alone, since the same `not_found` status means something
  different depending on what's actually on screen.
- `HARD_FAILURE` -- permission denied (403), application error (500), an
  exhausted retry budget, or an unrecognized checkpoint mismatch. Always
  carries `failed_step_id`, `expected`, and `observed` for debugging, plus
  a screenshot captured at the moment of failure.
- `HUMAN_INTERVENTION` -- an unrecognized dialog blocked progress (content
  stayed hidden after auto-dismissal). This is a *paused*, resumable
  state, not terminal -- see Section 5.

**Recoverable conditions** (a transient `timeout` status) get a bounded
number of automatic retries with backoff before falling through to
classification; every attempt, successful or not, is logged in
`recovery_attempts` so an evidence reviewer can see retries happened even
on a run that ultimately succeeded. This is tested directly against a
scripted executor double rather than fought for with real browser timing,
since the fake bank's slow-load member sleeps a fixed 3s and can't
simulate genuine flaky-then-fast latency (documented in Cuts).

Even a checkpoint failure after every individual step reported `ok` gets
re-classified the same way a mid-step failure would (the final page might
still match a known business-outcome/hard-failure pattern) before falling
back to a generic `CHECKPOINT_NOT_SATISFIED`.

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** `ComputerSurface` is a `Protocol` with six
methods (`observe/click/type/navigate/extract/screenshot`) operating
purely on `Target`/`Observation`/`ActionResult` -- none of which mention
DOM, HTML, or a browser. `PlaywrightSurface` is the only file that imports
Playwright. A legacy web app (framesets, non-semantic markup) needs no new
abstraction at all -- it's the exact case Phase 1's fake bank was built to
resemble, and the tiered locator strategy (role+name, falling back to a
row/label heuristic, falling back to stable-attribute CSS) already exists
because of it. A desktop app would need a new adapter (e.g. an OS
accessibility-tree adapter implementing the same six methods) but would
require zero changes to `tools/`, `agent/`, `artifacts/`, or `replay/` --
the artifact schema's `Target` already generalizes (`role`/`name` map
directly onto a desktop accessibility API's role/name concepts).

**Multi-tenant reuse.** `Artifact.surface` records `application` and
`version` -- the intended model is: a `base` artifact recorded against one
tenant's vendor app instance is the default capability for every tenant
running that same underlying product. At replay time, the caller supplies
which surface (which `base_url`/session) to run the artifact against --
the artifact itself contains no tenant-specific detail (URLs are relative
to the current session or explicitly parameterized), which is what makes
the base artifact reusable at all. Where a tenant's install genuinely
differs (a rebranded label, a moved button), the natural extension --
**not built here**, since the brief explicitly scopes multi-tenant to
design-only -- is a per-tenant *override* keyed by `(application, version,
tenant_id)` that replaces specific steps' targets or the checkpoint,
rather than a full re-recording: `Artifact.steps[i]` would be looked up
through an override table before execution, falling back to the base
step. **Drift detection** falls naturally out of the existing checkpoint
mechanism and `replay/stability.py` (Section 7): replaying the base
artifact against a new tenant N times and tracking its success rate over
time would surface drift as a rising failure rate on a specific step,
which is exactly the debugging information `ReplayResult.failed_step_id` /
`expected` / `observed` already produce per-run.

## 5. Escalation & handoff

Automation and a human share **one live session** -- `HandoffManager`
never opens a second browser or a fresh page for the human; it only
tracks *who currently owns* the existing `PlaywrightSurface`/page. The
sequence: `ReplayEngine` detects an unresolved dialog (Phase 3),
classifies it `HUMAN_INTERVENTION`, and -- if a `HandoffManager` was
supplied -- calls `request_intervention()`, which flips ownership to
`HUMAN` and records the goal, the failing step, a screenshot, and the
current page context. `ReplayResult.steps_completed` and `.outputs`
already carry everything needed to continue; a human resolves the
situation directly on the live session (in the tests, by revealing the
hidden DOM the same way accepting the dialog would), calls
`record_human_action()` for the audit trail, then `resume()` flips
ownership back. `ReplayEngine.resume(artifact, inputs, paused_result)`
restarts the loop from `paused_result.steps_completed` with
`paused_result.outputs` already populated -- **not** from step 1.

`handoff/routes.py` is a deliberately bare (per the brief's own scope
note) FastAPI mock console: `GET /intervention/pending`,
`POST /intervention/resolve`. It's a thin wrapper over the same
`HandoffManager` API -- a real operator console would add authentication
and a live co-browsing view in front of the identical mechanism.

**Discovery-side stuck detection** (Phase 4) is separate and simpler: three
consecutive action failures (including a blocked/unconfirmed risky action,
Section 6) trip `DiscoveryStatus.DEAD_END` rather than the model wandering
indefinitely. Wiring dead-end discovery into the same `HandoffManager` was
not built (Cuts) -- the natural extension is identical in shape to the
replay-side integration already working.

## 6. Safety

Every action -- from either the LLM (discovery) or a saved artifact
(replay) -- passes through the same `PolicyEngine` inside `ToolExecutor`,
before it reaches the surface. Two gates:

1. **Allowlist.** Domains (fail-closed: an empty/unconfigured list allows
   nothing by absolute URL; relative paths are always allowed, since they
   stay within whatever app the surface is already scoped to) and which of
   the six action types are permitted at all. Optional blocked-path
   prefixes layer on top (e.g. `/admin`) even within an allowed domain.
2. **Risk classification.** A target's visible name/label/text/value (and,
   for `type`, the value being typed) is matched against a configurable
   list of keyword patterns (`delete`, `transfer`, `approve`, ...).
   Matches default to `REQUIRE_CONFIRMATION`, not an outright `BLOCK` --
   an unattended run cannot supply that confirmation itself
   (`ToolExecutor.execute(..., confirmed=True)` is the only way past it),
   so in practice it behaves like a block unless something upstream (a
   human, via Section 5's handoff) explicitly approves it. Keyword
   matching over a hardcoded button-name allowlist was chosen because it
   generalizes across tenants running differently-worded UIs for the same
   underlying action, at the honest cost of both false positives and
   false negatives.

The fake bank's real UI has nothing destructive to demonstrate this
against -- `tests/test_safety.py` exercises it directly with synthetic
targets (a hypothetical "Delete Account" button), since the policy engine
only ever looks at strings, never at what application it's running
against.

**Secrets/PII.** The fake bank uses only synthetic data (no real
credentials, no real PII, documented in `app/data.py`). Nothing in this
project's logging or evidence writers persists credentials or tokens,
because none ever flow through the system -- there is no login step. A
production version would need explicit redaction rules in
`agent/recorder.py` / `replay/evidence.py` before writing typed values to
disk (a `type` step's value could be sensitive); this project's evidence
writers serialize whatever they're given, so that redaction layer is
listed as a Cut rather than a hidden gap.

## 7. Cuts

**Two stretch goals were built** (Section 8 of the brief allows "at most
one or two" -- picked for lowest effort against highest payoff, given
almost everything either needed already existed):

- **Agent-facing capability interface** (`capabilities/service.py`) --
  `GET /capabilities` (discover, with typed inputs/outputs),
  `GET /capabilities/{id}` (full detail), `POST /capabilities/{id}/invoke`
  (run via `ReplayEngine`, no LLM). This is the concrete, callable shape of
  "the AI agent invokes it in production" the brief's through-line
  describes -- verified live over real HTTP, not just an in-process test.
- **Multi-run stability** (`replay/stability.py`, `--stability-runs N` on
  the `replay` CLI) -- replays the same artifact/inputs N times and
  reports a success-rate/timing signal. Honest limitation: the fake bank
  is fully deterministic, so this always reports 100% (or 0%) consistency
  against it -- there is no real flakiness for this specific target to
  find. The counting/reporting logic itself is verified against a
  scripted engine producing a genuine mix of outcomes (tests/test_stability.py),
  so the gap is the target app's determinism, not the tool.

Everything else, deliberately left out, in rough priority order for "what
I'd build next":

- **Real flaky-timing simulation for retries.** The retry mechanism is
  fully implemented and tested against a scripted executor double, but
  never exercised against genuinely flaky (first-slow-then-fast) real
  browser timing, since the fake bank's slow-load member is a fixed,
  deterministic 3s sleep. Would need a target app that's flaky on
  purpose. Same root cause as the stability tool's limitation above.
- **Per-tenant artifact overrides.** Designed in Section 4, not
  implemented -- the brief explicitly scopes multi-tenant to design-only.
- **Confidence/approval gating** (draft -> approved states, unattended
  replay refused below a threshold). Not built as its own stretch goal,
  but `replay/stability.py`'s output is exactly the signal such a gate
  would consume -- this is the natural next layer on top of what exists,
  not a separate mechanism.
- **Discovery-side dead-end wired into `HandoffManager`.** Currently
  `DiscoveryStatus.DEAD_END` is a terminal status with no handoff; the
  mechanism to pause-and-escalate instead already exists and works on the
  replay side, wiring it into `DiscoveryAgent` is a small, identical-shape
  extension.
- **The mock operator console is not wired to a live session.**
  `handoff/routes.py` demonstrates the `HandoffManager` API's shape over
  HTTP, but the actual pause/resume cycle demonstrated in `/evidence/` runs
  through a standalone script holding one long-lived browser session in a
  single process, not through the HTTP console -- connecting the console to
  a live, running replay would need a persistent session registry, which is
  exactly the kind of infrastructure the brief says not to build
  prematurely for a single-tenant demo.
- **Redaction layer for evidence/artifacts.** No secrets exist in this
  demo to redact, but a production version handling real credentials
  would need explicit scrubbing before any typed value is written to
  disk.
- **A richer, multi-condition checkpoint** (per-step assertions, not just
  a single end-of-workflow check) -- the schema (`ArtifactCheckpoint`)
  would need to become a list rather than a single object; not done
  because the single-checkpoint model handled every test scenario
  correctly.
- **Ambiguous output-name matching** when two `extract` steps produce
  identical values during recording (Section 2) -- not encountered in
  testing, but a real limitation of value-based output-name matching.
- **Code generation, cross-tenant canonicalization demo** -- not built;
  lower priority than the two stretch goals above given the time
  available.