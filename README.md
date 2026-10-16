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
run_cli.py     interactive terminal driver
tests/         9 end-to-end route tests with a scripted fake LLM (no API key)
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
```

Output files land in `planning_output/<thread_id>/`: `SRD.md`, `TRD.md`,
`feasibility.md`, `scope.md` or `design.md`, plus `srd.json` / `trd.json` for Part 2.

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
- **Checkpointer:** `InMemorySaver` by default. For real use pass a persistent one, e.g.
  `PostgresSaver` (`langgraph-checkpoint-postgres`) or `SqliteSaver`
  (`langgraph-checkpoint-sqlite`): `build_graph(llm, checkpointer=...)`.

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
