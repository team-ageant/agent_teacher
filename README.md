# Adaptive AI Teacher

Adaptive AI Teacher is a Python-first conversational ReAct agent that turns plain-text learning material into a personalized lesson. A central `LearningSupervisor` chooses the most useful teaching action on every turn; there is no fixed learning path.

The application now uses FastAPI for the server, agent, session memory, APIs, and LLMod integration. The website is framework-free HTML and CSS with one small JavaScript file for browser interaction. Node.js, Next.js, React, and TypeScript are not required.

## Current status

Implemented: 

- responsive chat interface with a complete execution trace;
- dynamic supervisor and eight consistently named actions;
- temporary one-hour in-memory learning state;
- hard limit of 16 total LLM calls per session;
- LLMod chat and embedding clients;
- Pinecone retrieval over the pre-indexed Gutenberg corpus;
- deterministic, non-billable local demo mode;
- all four course-required endpoints and the optional reset endpoint;
- Python 3.12 test suite; and
- a Vercel-compatible FastAPI entrypoint.

Still pending:


- Supabase database connection;
- deletion of temporary session-specific Pinecone data (the Gutenberg corpus is persistent);
- final team email and batch/order metadata;
- Vercel environment configuration and production deployment.

## How the agent works

For every student message, `LearningSupervisor` observes the message, recent conversation, learning state, and remaining call budget. It selects exactly one action:

- `AskInterests` — discovers or refines the student's interests;
- `AnalyzeMaterial` — identifies the supplied material and its main topics;
- `ExplainMaterial` — explains a useful part at the student's apparent level;
- `StoryTool` — teaches through a story linked to known interests;
- `QuestionTool` — asks targeted questions;
- `AnswerEvaluator` — evaluates an answer and updates strengths and weaknesses;
- `RespondDirectly` — responds within the supervisor call; or
- `Stop` — ends the session when appropriate or required by the budget.

The supplied learning material is authoritative. General model knowledge may add examples or clarification, but the supplied material wins if the two conflict.

## Project structure

```text
app.py                         FastAPI and Vercel entrypoint
adaptive_teacher/
  agent.py                     Supervisor execution and state updates
  api_info.py                  Course metadata endpoint payloads
  config.py                    Environment-backed server configuration
  llm.py                       LLMod chat and embedding clients
  retrieval.py                 Pinecone Gutenberg semantic retrieval
  models.py                    State, trace, tool, and budget types
  prompts.py                   Supervisor and teaching-tool prompts
  state.py                     Temporary session store
public/
  index.html                   Browser interface
  styles.css                   Responsive visual design
  app.js                       Minimal browser-only interaction
  model-architecture.png       Required architecture diagram
tests/                         Non-billable Python test suite
docs/PROJECT_DOCUMENTATION.md  Detailed technical documentation
```

## Environment configuration

The local `.env.local` file has already been created on this machine and is ignored by Git. Do not replace it or commit it.

`.env.example` is intentionally retained. It is not loaded at runtime and contains no secret. It documents the required variable names for teammates, a new clone, and Vercel. On a new machine only:

```bash
cp .env.example .env.local
```

Then fill in the private values. The main variables are:

```env
LLMOD_API_KEY=
LLMOD_BASE_URL=https://api.llmod.ai
LLMOD_MODEL=MB5R2CF-azure/gpt-5.4-mini
LLMOD_EMBEDDING_MODEL=MB5R2CF-azure/text-embedding-3-small
GROUP_BATCH_ORDER_NUMBER=TBD_TBD
TEAM_NAME=Adaptive AI Teacher
BATEL_EMAIL=
ITAY_EMAIL=
BOAZ_EMAIL=
PINECONE_API_KEY=
PINECONE_INDEX_HOST=https://your-index-host
PINECONE_NAMESPACE=__default__
PINECONE_TOP_K=5
```

If `LLMOD_API_KEY` is missing during local development, the application uses deterministic demo responses. Tests explicitly enable demo mode so they never consume the course budget. Production fails closed when the key is missing, preventing an incorrectly configured deployment from silently serving mock lessons.

## Local development

Requirements: Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
uvicorn app:app --reload --port 3000
```

Open `http://localhost:3000`.

Run all automated checks:

```bash
python -m pytest
```

The browser contains the only remaining JavaScript because request handling and dynamic page updates must execute in the browser. All project logic and server-side behavior are Python.

## API contract

### `GET /api/team_info`

Returns the group identifier, team name, and team members.

### `GET /api/agent_info`

Returns the agent description, purpose, prompt template, and a complete traced example.

### `GET /api/model_architecture`

Returns the architecture image with `Content-Type: image/png`.

### `POST /api/execute`

Request:

```json
{
  "prompt": "Student request or plain-text learning material"
}
```

Every success and error response contains exactly four top-level fields:

```json
{
  "status": "ok",
  "error": null,
  "response": "Agent response",
  "steps": []
}
```

Usage metadata is returned through `X-LLM-Calls-Used`, `X-LLM-Calls-Remaining`, and `X-Agent-Session-Id` headers.

### `DELETE /api/session`

Deletes the current temporary state and clears the session cookie.

## LLMod integration

- Base URL: `https://api.llmod.ai`
- Chat endpoint: `/v1/chat/completions`
- Chat model: `MB5R2CF-azure/gpt-5.4-mini`
- Embeddings endpoint: `/v1/embeddings`
- Embedding model: `MB5R2CF-azure/text-embedding-3-small`
- Verified embedding dimension: 1,536

The API key remains server-side and is never included in `public/` or sent to the browser.

## Pinecone retrieval

Each student turn searches the pre-indexed Gutenberg corpus and supplies up to five relevant
passages to both the supervisor and the selected teaching tool. Queries and document chunks
are embedded using LLMod (`MB5R2CF-azure/text-embedding-3-small`) and stored/searched via Pinecone.
If Pinecone is not configured or is temporarily unavailable, the agent continues without
external passages. Set `PINECONE_NAMESPACE=__default__` for Pinecone's default namespace.

## Vercel deployment

The root `app.py` is a recognized FastAPI entrypoint, and Vercel serves `public/` assets directly. No Node build step or `vercel.json` is required for the current application.

To deploy later:

1. import the GitHub repository into Vercel;
2. add the `.env.local` variable names and private values in Vercel Project Settings;
3. deploy `main`; and
4. verify the root interface and all required endpoints.

Temporary module memory is best effort on serverless infrastructure. Supabase and Pinecone must hold any state that needs to survive instance recycling or scaling. Before exposing the shared course key publicly, add a durable project-wide usage limit or rate limiter; a per-session limit alone can be renewed by starting a new session.

For design decisions, data-flow details, constraints, and the remaining RAG plan, see [the technical documentation](docs/PROJECT_DOCUMENTATION.md).
