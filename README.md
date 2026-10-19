# AI_Workflow_Global

SDLC orchestrator workflow (LangGraph). Part 1: planning.

LangGraph implementation of the planning phase as agreed so far: SDD + TDD, two
modes, human gates at every artifact, classified rejections, and a separate TRD.

## Layout

```
planning/
  state.py     artifact schemas (Pydantic) + PlanningState
  prompts.py   one short system prompt per LLM step
  nodes.py     all nodes + Settings (caps, output dir)
  graph.py     wiring, mode router, TRD subgraph, design subgraph (Part 2 placeholder)
  render.py    markdown for review payloads and output files
  llm.py       PlanningLLM protocol + LangChain adapter (any provider)
  audit.py     decision log + versioned artifacts with reasons and diffs
  persistence.py  SQLite/Postgres checkpointers + git commit of approved specs
  tracing.py   LangSmith run names, tags, metadata, and metadata-only mode
  harness.py   run steps on your subscriptions: Claude Code CLI, Copilot CLI, per-step routing
routing.example.json  which back end each planning step uses
run_cli.py     interactive terminal driver
tests/         33 tests, no API key or network: routes, audit trail, git, resume, tracing, harnesses
graph.mmd      Mermaid export of the compiled graph
```

## Run

```bash
pip install -r requirements.txt
pytest -q                                   # routes, no API key needed

export ANTHROPIC_API_KEY=...
export PLANNING_MODEL=anthropic:claude-sonnet-5-5   # any LangChain "provider:model"
python run_cli.py "Build a password reset flow" --mode requirements_first
python run_cli.py "Build a password reset flow" --mode design_first

python run_cli.py "..." --specs-repo ../my-project   # commit approved specs into your project repo
python run_cli.py --thread <run id>                  # resume a stopped run
python run_cli.py --thread <run id> --history        # list every saved step of a run
```

Output files land in `planning_output/<thread_id>/`: `SRD.md`, `TRD.md`,
`feasibility.md`, `scope.md` or `design.md`, plus `srd.json` / `trd.json` for Part 2.

## History: why something changed, and how

Three layers, each answering a different question:

| Question | Where to look |
|---|---|
| Why did REQ-002 change, and who decided? | `CHANGELOG.md` / `decisions.jsonl`: every answer, review, classification, escalation, in order |
| What exactly changed between versions? | `versions/SRD_v1.md`, `SRD_v2.md`, `SRD_v2.diff` (each file starts with the reason it exists) |
| What did the approved specs look like on a given day? | `--specs-repo`: one git commit per approval in `<project>/specs/`, reason and rejection history in the commit message |
| What was the full graph state at step N? Can I resume or branch from it? | the checkpointer (`.workflow/checkpoints.sqlite` by default): `--history`, `graph.get_state_history()`, replay, `graph.update_state()` to fork |

Example commit message:

```
specs: SRD v2 approved

Why this version: SRD review rejection: REQ-002 should expire after 30 minutes
Changed: REQ-002
Rejections at this gate: 1
  - v1 rejected: REQ-002 should expire after 30 minutes
Planning run: 522e1aaa-...
```

Only approvals are commits; rejections are recorded in the log that each commit carries.
Committing skips when nothing changed, so re-running a step never creates duplicate commits.

## Run on your subscriptions (no API keys)

By default every planning step runs through **Claude Code** (`claude -p`) on your
Claude plan. Any step can go to **GitHub Copilot CLI** (`copilot -p`) on your
Copilot plan instead. The flow, review gates, audit log and git commits are
unchanged; only how each step reaches a model differs.

One-time setup:

```bash
npm install -g @anthropic-ai/claude-code && claude     # sign in with your Claude plan
npm install -g @github/copilot && copilot              # sign in with GitHub
cd ~/.ai_workflow/harness && copilot                   # trust the folder planning calls run in
cp routing.example.json routing.json                   # choose a back end per step
```

`routing.json` (picked up automatically from the current folder, or pass `--routing`):

```json
{"default": "claude", "steps": {"classify": "copilot", "testability": "copilot", "consistency": "copilot"}}
```

Back ends: `claude[:model]` (e.g. `claude:sonnet`), `copilot[:model]`, or
`api[:provider:model]` for an API key. Steps: `clarify`, `scope`, `feasibility`,
`srd`, `testability`, `classify`, `trd`, `design`, `consistency`. The run prints
which back end each step uses.

How the calls are made:
- They run in `~/.ai_workflow/harness`, an empty folder, so no project's CLAUDE.md,
  hooks or MCP servers load, and they get **no tools**: no commands, file edits or web.
- API-key variables are removed from the Claude call, so it always uses your plan.
  (`--bare` mode is avoided on purpose: it skips the plan sign-in.)
- Claude Code enforces the output schema itself. Copilot gets the schema in the
  prompt; its answer is validated and retried once with the error.
- If a plan's usage limit is hit mid-run, the run stops with a clear message and
  everything so far is saved. `python run_cli.py --thread <run id>` retries the
  failed step and continues.

Measured on a real run (Claude Code CLI 2.1.303): a full requirements-first run with
two rounds of requirement questions took about 5½ minutes and **16 Claude calls**,
and produced 15 requirements and 61 acceptance tests. Each call draws on your plan's
usage limits, shared with your own Claude Code use; each Copilot call, retries
included, uses Copilot AI credits.

Trade-offs against API keys: slower (a CLI starts per call), and LangSmith shows each
call's prompt, answer and duration but not token counts or cost. Personal use only:
Anthropic doesn't allow routing other people's requests through your plan sign-in.

## Observability with LangSmith

Shows what the model did at every step: exact prompts and answers, tokens, cost,
latency and errors. No code changes are needed; set these and run as usual:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your key>          # from smith.langchain.com
export LANGSMITH_PROJECT=ai-workflow         # where runs appear in the UI
# EU accounts: export LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

- Every planning run shows as one **thread** (grouped by its run id), even across
  pauses and restarts.
- Filter by tags: `part:planning`, `mode:<mode>`, `project:<name>`; metadata also
  carries `workflow_version`. Set the project name with `--project` (default: the
  folder name of `--specs-repo`, or the current folder).
- `--no-content` sends timings, tokens, errors, tags and model settings, but **no
  inputs or outputs** (no prompts, answers or spec text). It sticks to the run:
  resuming a run started with it keeps it, even if you forget the flag.
- Traces are for debugging and monitoring, not a record: on LangSmith's published
  plans base traces are kept 14 days. The decision log and git commits stay the
  system of record.

`tests/test_tracing.py` checks what is actually sent, against a local capture server.

## How the decisions map to code

| Decision | Where |
|---|---|
| Agent asks instead of deciding silently | `clarify_detect` → `clarify_ask` loop, before the mode router |
| Mode selects the entry node | `_route_clarify`: `scope_draft` (requirements-first) or `design` subgraph (design-first) |
| Rejection keeps the draft and the reason | `last_rejection` + `rejection_history`; drafting nodes revise the previous artifact |
| Scope vs detail rejection | `classify_rejection`. Scope-gate rejections are scope by definition (no LLM call); only SRD-gate rejections are classified by the LLM |
| Detail fix touches only affected requirements | `merge_srd` keeps unaffected requirements byte-for-byte, whatever the LLM returns |
| Testable SRD (EARS) | `testability_check`: regex for EARS shape + vague words first, LLM only on changed requirements |
| What vs how | `feasibility.notes_md` goes to `feasibility.md`; only `srd_constraints` reach the SRD |
| Separate TRD with its own gate | `trd` subgraph; TRD rejections loop locally |
| TRD back-channel | `draft_trd` returns `untestable` → questions to you → `draft_srd` for those IDs → TRD regenerates only their tests |
| Design-first never loops back to design | after TRD approval: `consistency_check` → `handoff`; a gap goes to the design node as a targeted fix |
| Iteration cap | per-gate counters; `escalate` asks you to continue (with guidance) or abort |

## Implementation notes

- **Gates are interrupt-only nodes.** A node re-runs from the top when it resumes,
  so no gate makes an LLM call. Invalid resume values are re-asked in the same node.
- **No list reducers in state.** A subgraph returns its whole state to the parent; an
  `operator.add` reducer on a shared key would append the history twice. History is
  appended by hand instead.
- **State holds plain dicts**, so any checkpointer serialises it without registering
  custom types.
- **Checkpointer:** the CLI uses SQLite (`planning.persistence.sqlite_checkpointer`).
  `build_graph()` without one uses `InMemorySaver` (tests only: lost when the process stops).
  For production use `postgres_checkpointer(conn_string)` (needs `langgraph-checkpoint-postgres`).

## Resume payloads

| Pause type | Resume with |
|---|---|
| `clarify` | `{"Q1": "answer", "T-REQ-002": "answer", ...}` (every question id) |
| `review` (scope / srd / trd / design) | `{"action": "approve"}` or `{"action": "reject", "feedback": "..."}` |
| `escalation` | `{"action": "continue", "guidance": "..."}` or `{"action": "abort"}` |

## Your open decisions (defaults in `Settings`)

- `max_rejections=3` per gate before escalation
- `max_clarify_rounds=4`; questions still open after that are shown at the next gate
- `max_testability_rounds=2`; leftover issues are shown at SRD review
- The classifier's scope/detail definitions in `prompts.CLASSIFY`
- `design` subgraph is a placeholder for Part 2; keep its contract
  (writes `state["design"]` and `state["design_outcome"]`)
