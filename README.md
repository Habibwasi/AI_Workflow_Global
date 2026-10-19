# AI_Workflow_Global

SDLC orchestrator workflow (LangGraph). **Part 1: planning.**

You describe what to build; the workflow asks you clarifying questions, drafts the
scope, checks feasibility, writes a Software Requirements Document (SRD) with
testable acceptance criteria, then a Test Requirements Document (TRD), and stops
for your approval at every step. It runs on your **Claude** and **GitHub Copilot**
subscriptions through their command-line tools, so no API keys are needed.

**Contents**

1. [Quick start](#quick-start)
2. [What you need](#1-what-you-need)
3. [Install and sign in](#2-install-and-sign-in)
4. [Choose a model for each step](#3-choose-a-model-for-each-step)
5. [Run a planning session](#4-run-a-planning-session)
6. [Stop, resume and inspect a run](#5-stop-resume-and-inspect-a-run)
7. [What you get, and the history of every change](#6-what-you-get-and-the-history-of-every-change)
8. [Costs and limits](#7-costs-and-limits)
9. [Troubleshooting](#8-troubleshooting)
10. [Observability with LangSmith (optional)](#9-observability-with-langsmith-optional)
11. [Reference](#reference)

---

## Quick start

Already have Claude Code and Copilot CLI installed and signed in? From this folder:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p ~/.ai_workflow/harness && (cd ~/.ai_workflow/harness && copilot)   # trust the folder once, then /exit
cp routing.example.json routing.json
python run_cli.py "Add a forgot-password flow to my web app" --specs-repo ../my-project
```

---

## 1. What you need

| | Notes |
|---|---|
| **Claude Pro or Max plan** | Claude Code needs Pro or higher; the free plan doesn't include it |
| **GitHub Copilot plan** | Pro, Pro+, Business or Enterprise. Copilot CLI is included on all plans. Through an organization, its Copilot CLI policy must be on |
| **Python 3.10+** and **git** | For the workflow itself |
| **Node.js 22+** | Only if you install the CLIs with npm |
| **Your own computer** | Your plan sign-in is for your personal use; Anthropic doesn't allow routing other people's requests through it |

**Why the command-line tools and not the chat panel in Visual Studio or VS Code?** No IDE
chat panel can be driven by another program. The CLIs use the same subscriptions and the
same sign-in. Run the workflow from your IDE's terminal.

---

## 2. Install and sign in

### Claude Code

Install (pick one):

```bash
curl -fsSL https://claude.ai/install.sh | bash      # macOS, Linux, WSL (recommended)
irm https://claude.ai/install.ps1 | iex             # Windows PowerShell
brew install --cask claude-code                     # Homebrew
winget install Anthropic.ClaudeCode                 # WinGet
npm install -g @anthropic-ai/claude-code            # npm (Node 22+; never use sudo)
```

Open a **new** terminal, then sign in and check:

```bash
claude --version          # prints e.g. 2.1.303 (Claude Code)
claude                    # follow the browser prompt to sign in with your Claude plan, then /exit
claude auth status        # "loggedIn": true and "authMethod": "claude.ai" (or "oauth_token")
```

If `authMethod` says `api_key`, Claude Code is using an API key, not your plan. The
workflow removes `ANTHROPIC_API_KEY` from its own calls anyway, but for your
interactive use, unset it (`unset ANTHROPIC_API_KEY`; on Windows remove it from your
environment variables).

Quick test, the same way the workflow calls it:

```bash
echo "Reply with the word ready" | claude -p --output-format json
```

### GitHub Copilot CLI

Install (pick one):

```bash
npm install -g @github/copilot                      # all platforms (Node 22+)
brew install --cask copilot-cli                     # macOS, Linux
winget install GitHub.Copilot                       # Windows (needs PowerShell 6+)
curl -fsSL https://gh.io/copilot-install | bash     # macOS, Linux
```

Sign in, then **trust the folder the workflow runs Copilot in**. Planning calls run in
an empty folder so they never pick up a project's instructions, and Copilot asks once
whether to trust it:

```bash
mkdir -p ~/.ai_workflow/harness && cd ~/.ai_workflow/harness
copilot                   # run /login if asked, accept "trust this folder", then /exit
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.ai_workflow\harness" | Out-Null
cd "$HOME\.ai_workflow\harness"; copilot
```

Quick test:

```bash
copilot -p "Reply with the word ready" -s
```

### This workflow

Create your project from this template (**Use this template** on GitHub), clone it,
and from its folder:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                            # 33 tests; uses stand-in CLIs, spends nothing
```

---

## 3. Choose a model for each step

Copy the example and edit it:

```bash
cp routing.example.json routing.json
```

```json
{
  "default": "claude",
  "steps": {
    "classify":    "copilot",
    "testability": "copilot",
    "consistency": "copilot"
  }
}
```

`default` is used for every step not listed under `steps`. The workflow reads
`routing.json` from the folder you run it in, or from `--routing <file>`. Each run
prints the model it uses for every step, so you can check before anything is spent.

### The steps

| Step | What it does | Suggested |
|---|---|---|
| `clarify` | Finds ambiguities and asks you questions | strong |
| `scope` | Drafts scope and goals (requirements-first mode) | strong |
| `feasibility` | Technical, operational, cost and schedule check | strong |
| `srd` | Writes the requirements with EARS acceptance criteria | **strongest** |
| `testability` | Checks each requirement can be tested | cheap |
| `classify` | Decides whether a rejection is a scope or a detail problem | cheap |
| `trd` | Writes the acceptance tests | strong |
| `design` | Drafts the design (design-first mode; Part 2 placeholder) | strong |
| `consistency` | Checks the design covers every requirement (design-first) | cheap |

### How to write a model

| Write | Means |
|---|---|
| `claude` | Claude Code, your plan's default model |
| `claude:<model>` | A specific Claude model, e.g. `claude:sonnet`, `claude:opus`, `claude:haiku`, or a full name |
| `copilot` | Copilot CLI, your default Copilot model |
| `copilot:<model-id>` | A specific Copilot model, e.g. `copilot:gpt-5-mini` |
| `api:<provider:model>` | An API key instead of a subscription, e.g. `api:anthropic:claude-sonnet-5-5` |

**Finding exact model names:**

- **Claude:** run `claude`, then `/model`. It shows the models your plan includes.
- **Copilot:** run `copilot`, then `/model`. The picker lists your models with their
  context size and AI-credit cost. Use the **ID** (e.g. `claude-haiku-4.5`), not the
  display name on GitHub's pricing page. Copilot CLI's list can be shorter than your
  IDE's, so the CLI's list is the one that counts.
- **Test a name before a long run:** `copilot -p "Reply with ready" -s --model <id>`
  or `echo "Reply with ready" | claude -p --model <name>`.

### Example setups

```jsonc
// Claude drafts, a cheap Copilot model checks (recommended start)
{"default": "claude", "steps": {"classify": "copilot:gpt-5-mini", "testability": "copilot:gpt-5-mini", "consistency": "copilot:gpt-5-mini"}}

// Save your Claude limit: only the SRD on your Claude plan, everything else on Copilot
{"default": "copilot", "steps": {"srd": "claude"}}

// Copilot only
{"default": "copilot"}
```

Copilot also offers Claude models. Routing a step to `copilot:<a claude model id>`
spends Copilot credits instead of your Claude plan's limit.

---

## 4. Run a planning session

Run from the repository folder, so `routing.json` and the run history are found:

```bash
python run_cli.py "Add a forgot-password flow to my web app" --specs-repo ../my-project
```

It prints the run id (keep it to resume later) and the model per step. Then it pauses
whenever it needs you:

| You see | You do |
|---|---|
| `=== Questions from the planning agent ===` | Type an answer to each question. The more precise, the fewer follow-ups |
| `=== Review: SCOPE / SRD / TRD / DESIGN ===` | Read it, then `a` to approve, or `r` and say what's wrong. Your reason is recorded and the agent revises only what it needs to |
| `=== Escalation ===` (a step was rejected more than 3 times) | `c` to continue, optionally with guidance, or `x` to stop |

At the end it prints where the files are and which phase comes next.

### Options

| Option | What it does |
|---|---|
| `--mode requirements_first` (default) | Scope → feasibility → SRD → TRD |
| `--mode design_first` | Design → feasibility → SRD derived from the design → TRD → consistency check |
| `--specs-repo <path>` | Commit every approved artifact into `<path>/specs/` of your project's git repo, with the reason in the commit message |
| `--routing <file>` | Use this routing file instead of `./routing.json` |
| `--llm <model>` | Default model for every step when there's no routing file, e.g. `--llm copilot` |
| `--max-rejections <n>` | After more than this many rejections at one step, it asks whether to continue (default 3) |
| `--project <name>` | Project name in LangSmith (default: folder name) |
| `--no-content` | LangSmith gets timings and tags only, no text (see section 9) |
| `--thread <run id>` | Resume a run, or inspect it with `--history` |
| `--db <file>` | Where run history is kept (default `.workflow/checkpoints.sqlite`) |

---

## 5. Stop, resume and inspect a run

Every step is saved as it happens, in `.workflow/checkpoints.sqlite`.

```bash
python run_cli.py --thread <run id>              # continue where it paused or failed
python run_cli.py --thread <run id> --history    # list every saved step
```

- **You closed the terminal or pressed Ctrl+C:** resume, and it asks the pending question again.
- **A plan's usage limit was hit, or a call failed:** the run stops with a message saying
  why. Everything before that step is kept. Resume once the limit resets, or first route
  that step to the other tool in `routing.json`. Only the failed step is retried; you
  aren't asked to re-approve anything.
- Resume from the **same folder** (or pass the same `--db`), since that's where the history is.

---

## 6. What you get, and the history of every change

Files land in `planning_output/<run id>/`:

| File | Contents |
|---|---|
| `SRD.md`, `srd.json` | Approved requirements (REQ-001…) with acceptance criteria |
| `TRD.md`, `trd.json` | Acceptance tests (TC-001a…) with traceability to requirements |
| `feasibility.md` | The technical findings, kept apart from the requirements |
| `scope.md` or `design.md` | Depending on the mode |
| `CHANGELOG.md`, `decisions.jsonl` | Every answer, review, classification and escalation, in order |
| `versions/` | Every draft (`SRD_v1.md`, `SRD_v2.md`…) with the reason it exists, and a `.diff` to the previous one |

With `--specs-repo`, each approval is also a git commit in your project:

```
specs: SRD v2 approved

Why this version: SRD review rejection: REQ-002 should expire after 30 minutes
Changed: REQ-002
Rejections at this gate: 1
  - v1 rejected: REQ-002 should expire after 30 minutes
Planning run: 522e1aaa-...
```

Only approvals are commits; rejections are recorded in the log each commit carries.
The workflow commits only inside `specs/` and never pushes. Push when you're ready.

| Question | Where to look |
|---|---|
| Why did REQ-002 change, and who decided? | `CHANGELOG.md` / `decisions.jsonl` |
| What exactly changed between versions? | `versions/*.diff` |
| What did the approved specs look like on a given day? | `git log specs/` in your project |
| What was the full state at step N? | `--history`, or `graph.get_state_history()` |

---

## 7. Costs and limits

Measured on a real run (Claude Code 2.1.303): a full requirements-first run with two
rounds of requirement questions took about **5½ minutes and 16 Claude calls**, and
produced 15 requirements and 61 acceptance tests.

- **Claude plan:** each call counts toward your plan's usage limits, the same limits as
  your own Claude Code use. How many runs fit depends on your plan and how much else
  you use Claude. When the limit is hit, the run stops and can be resumed.
- **Copilot plan:** usage is charged in AI credits by tokens at each model's price
  (1 credit = $0.01; Pro includes 1,500 a month). Per million tokens, mini and flash
  models cost cents, while the largest models cost tens of times more. Copilot adds its
  own instructions to each call, so real usage is higher than your prompt. Each
  retry is another call. When credits run out you can set a paid budget, upgrade, or
  wait for the reset on the 1st of the month.
- **Speed:** each call starts a CLI process, so it is slower than an API key.

---

## 8. Troubleshooting

| Message or symptom | Fix |
|---|---|
| `'claude' was not found on PATH` / `'copilot' was not found on PATH` | Install it (section 2) and open a new terminal |
| `claude reported an error: ... usage limit ...` | Your Claude limit is used up. Resume with `--thread` after it resets, or route that step to `copilot` |
| `... safeguards flagged this message ...` | A safety filter flagged ordinary planning text (a false positive). Resume to retry, or route the step to another model, e.g. `claude:opus` or `copilot` |
| `copilot failed ... Model X is unsupported` | Use an ID from `/model` in Copilot CLI, not a display name |
| A Copilot step hangs or asks to trust a folder | Run `copilot` once inside `~/.ai_workflow/harness` and accept trust |
| `did not return a valid <Step> after 2 attempts` | That model struggled with the output format. Route the step to `claude` (it enforces the format) or a stronger Copilot model |
| `too long for a Windows command line` | Route that step to `claude`, which reads prompts from standard input |
| Claude Code uses an API key instead of your plan | `claude auth status` shows `api_key`: unset `ANTHROPIC_API_KEY` and sign in with `claude` |
| `give a prompt to start a run...` when resuming | Run from the same folder as before, or pass the same `--db` |
| Copilot says your organization disabled the CLI | An organization admin must enable the Copilot CLI policy |

`claude doctor` checks your Claude Code installation and settings.

---

## 9. Observability with LangSmith (optional)

Shows every step of a run: prompts, answers, timings and errors.

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your key>          # from smith.langchain.com
export LANGSMITH_PROJECT=ai-workflow
# EU accounts: export LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

- Each planning run shows as one **thread**, across pauses and restarts.
- Each Claude Code / Copilot call appears as its own step (`claude-code:SRD`,
  `copilot:RejectionClassification`…) with its prompt, answer and duration. Token
  counts and cost aren't shown for these: the call happens inside the CLI.
- Filter by `part:planning`, `mode:<mode>`, `project:<name>`; metadata has `workflow_version`.
- `--no-content` sends timings, tags and errors but **no prompts, answers or spec
  text**. It sticks to the run: resuming keeps it even without the flag.
- Traces are kept 14 days on LangSmith's base plans. The decision log and git commits
  are your permanent record.

---

## Reference

### How the planning calls are made

- They run in `~/.ai_workflow/harness`, an empty folder, so no project's CLAUDE.md,
  hooks or MCP servers load. They get **no tools**: no commands, file edits or web.
- API-key variables are removed from the Claude call, so it always uses your plan.
  `--bare` mode is avoided on purpose, because it skips the plan sign-in.
- Claude Code enforces the output schema itself (`--json-schema`), and the prompt
  goes in on standard input. Copilot gets the schema in its prompt; its answer is
  validated and retried once with the error.

### Using an API key instead

Any step can use an API key with `api:<provider:model>` in `routing.json`, or for all
steps `python run_cli.py "..." --llm api:anthropic:claude-sonnet-5-5` (set the
provider's key, e.g. `ANTHROPIC_API_KEY`). API calls are faster and LangSmith then also
shows tokens and cost.

### Layout

```
planning/
  state.py        artifact schemas (Pydantic) + PlanningState
  prompts.py      one short system prompt per LLM step
  nodes.py        all nodes + Settings (caps, output dir)
  graph.py        wiring, mode router, TRD subgraph, design subgraph (Part 2 placeholder)
  harness.py      Claude Code CLI / Copilot CLI back ends and per-step routing
  llm.py          API back end (any LangChain provider)
  render.py       markdown for review payloads and output files
  audit.py        decision log + versioned artifacts with reasons and diffs
  persistence.py  SQLite/Postgres checkpointers + git commit of approved specs
  tracing.py      LangSmith run names, tags, metadata, metadata-only mode
routing.example.json  model per step
run_cli.py        interactive terminal driver
tests/            33 tests, no account or network: routes, audit, git, resume, tracing, CLIs
graph.mmd         Mermaid export of the compiled graph
```

### How the decisions map to code

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

### Implementation notes

- **Gates are interrupt-only nodes.** A node re-runs from the top when it resumes, so
  no gate makes an LLM call. Invalid resume values are re-asked in the same node.
- **No list reducers in state.** A subgraph returns its whole state to the parent; an
  `operator.add` reducer on a shared key would append the history twice.
- **State holds plain dicts**, so any checkpointer serialises it without custom types.
- **Checkpointer:** the CLI uses SQLite. `build_graph()` without one uses
  `InMemorySaver` (tests only). For production, `postgres_checkpointer(conn_string)`
  (needs `langgraph-checkpoint-postgres`).

### Resume payloads (when driving the graph from your own code)

| Pause type | Resume with |
|---|---|
| `clarify` | `{"Q1": "answer", "T-REQ-002": "answer", ...}` (every question id) |
| `review` (scope / srd / trd / design) | `{"action": "approve"}` or `{"action": "reject", "feedback": "..."}` |
| `escalation` | `{"action": "continue", "guidance": "..."}` or `{"action": "abort"}` |

### Defaults you can change (`Settings` in `planning/nodes.py`)

- `max_rejections=3` per step before escalation
- `max_clarify_rounds=4`; questions still open after that are shown at the next review
- `max_testability_rounds=2`; leftover issues are shown at SRD review
- The classifier's scope/detail definitions in `prompts.CLASSIFY`
- The `design` subgraph is a placeholder for Part 2; keep its contract
  (writes `state["design"]` and `state["design_outcome"]`)
