---
title: "Adaptive AI Teacher"
subtitle: "Technical Project Documentation"
date: "August 2026"
---

# Executive summary

Adaptive AI Teacher is a Python-first conversational ReAct agent for personalized learning from plain-text material. It does not enforce a predefined lesson sequence. Instead, a central `LearningSupervisor` evaluates the current student message, recent conversation, temporary learning state, prior action, and remaining LLM budget. It then selects the single most useful teaching action for that turn.

The working application includes a framework-free browser interface, FastAPI server, all required course endpoints, complete LLM execution traces, temporary conversational memory, a hard 16-call LLM limit, LLMod clients, and Pinecone retrieval over the pre-indexed Gutenberg corpus. Supabase persistence and production Vercel deployment remain pending.

# Confirmed product decisions

The implementation follows these project decisions:

- Input consists of text only.
- Interest discovery can happen at the beginning or gradually later in the conversation.
- The LLM-based supervisor decides which tool to select.
- Learning is a continuous multi-turn conversation.
- Current learning memory is temporary.
- Teaching uses both the supplied material and the LLM's general knowledge, but the supplied material wins whenever there is a conflict.
- A session may use at most 16 total LLM calls.
- There is no fixed route through the tools. The supervisor makes a fresh decision on every turn and stops when it is useful to stop or when the budget is exhausted.

# Problem and objective

Learning material is often dense, abstract, and disconnected from a student's interests. A single explanation style is unlikely to work equally well for every learner. This project adapts explanations, stories, questions, and feedback to what the student has supplied and demonstrated during the conversation.

The objective is to build an autonomous teaching agent that:

- accepts learning material as plain text;
- maintains a continuous conversation;
- discovers and refines student interests;
- chooses teaching actions dynamically;
- identifies topics and weak areas;
- evaluates answers and infers mastery;
- respects a strict model-call budget; and
- exposes every model call for inspection and grading.

# Current implementation status

## Completed

- Python 3.12 and FastAPI server architecture.
- Responsive framework-free learning interface.
- Dynamic `LearningSupervisor` decision logic.
- Eight consistent actions and tool names.
- Temporary in-memory state with one-hour expiry.
- Hard limit of 16 total LLM calls per session.
- Full trace with module, system prompt, user prompt, and structured response.
- Team, agent-information, architecture, and execution endpoints.
- Optional session-reset endpoint.
- LLMod chat integration with `MB5R2CF-azure/gpt-5.4-mini`.
- LLMod embedding integration with `MB5R2CF-azure/text-embedding-3-small`.
- Verified 1,536-dimensional embedding output.
- Deterministic demo mode and non-billable automated tests.
- Vercel-compatible root FastAPI entrypoint.
- GitHub repository on the `main` branch.

## Pending

- Supabase schema and primary database connection.
- Pinecone index and vector operations.
- Text chunking, embedding upsert, semantic retrieval, and namespace deletion.
- Durable project-wide usage limiting before public deployment.
- Final team email and group batch/order values.
- Vercel environment configuration and production deployment.

# Python-first system architecture

![Dynamic ReAct architecture](../public/model-architecture.png)

The runtime has four layers:

1. **Browser interface.** `public/index.html` and `public/styles.css` define the page. `public/app.js` performs only browser-required work: submits messages, updates the DOM, displays call usage, resets the session, and renders traces.
2. **FastAPI server.** `app.py` validates requests, manages the cookie, resolves temporary state, invokes the agent, and returns the exact assignment schema.
3. **ReAct agent.** `adaptive_teacher/agent.py` asks the supervisor to choose one action, optionally invokes the selected teaching tool, updates state, and tracks the LLM budget.
4. **Model and data services.** LLMod currently provides chat completion and embeddings. Supabase and Pinecone will provide durable metadata and retrieval after integration.

The browser is the only place where JavaScript remains. Python cannot run directly as normal client-side browser code, so a small JavaScript file is necessary for an interactive page. No application logic, agent logic, secret handling, or server API is implemented in JavaScript.

# Runtime flow

For a successful `POST /api/execute` request:

1. FastAPI validates that `prompt` is a non-empty string no longer than 30,000 characters.
2. The server reads or creates the opaque `adaptive_session_id` cookie.
3. The temporary store retrieves the learning state and removes expired sessions.
4. If all 16 calls were already used, the agent returns a deterministic stop response without contacting LLMod.
5. Otherwise, `LearningSupervisor` receives a compact state and selects one action. This consumes one LLM call.
6. `RespondDirectly` and `Stop` return the response from the supervisor decision.
7. Any other action invokes its LLM-backed teaching tool, consuming one additional call.
8. The agent updates material, interests, topics, strengths, weaknesses, history, and the last action as applicable.
9. FastAPI saves the state and returns the response and complete trace.
10. Call usage and session metadata are returned in HTTP headers.

# Dynamic supervisor actions

There is no fixed learning workflow. On each student turn, the supervisor selects exactly one action.

## AskInterests

Discovers or refines interests with a short, natural question. It can be selected initially or at a later point when more personalization would improve teaching.

## AnalyzeMaterial

Identifies the main concepts in newly supplied text and stores the authoritative material. In the completed retrieval version, it will also coordinate chunking and semantic retrieval.

## ExplainMaterial

Explains a useful part of the supplied material at the student's apparent level, prioritizing recently discussed or weak topics.

## StoryTool

Transforms relevant material into a memorable story linked to known student interests.

## QuestionTool

Creates one or a small number of targeted questions based on the material and current weak topics.

## AnswerEvaluator

Compares the latest student answer with the supplied material, gives feedback, and may update score, mastery, strong topics, and weak topics.

## RespondDirectly

Returns a complete answer from the supervisor call without invoking a second tool. This is especially useful when only one model call remains.

## Stop

Ends learning when mastery is sufficient, the student asks to stop, no useful action remains, or the remaining call budget requires termination.

# Source-of-truth policy

The supplied learning material is authoritative for the session. General model knowledge may provide examples, analogies, or clarification, but it must not silently replace the source. If the two conflict, the supplied material is treated as correct for the current learning session.

This policy is included in both the supervisor prompt and every teaching-tool prompt. It keeps questions, evaluation, and explanations grounded in the material the student provided.

# Temporary learning state

The current process-local state contains:

- an opaque session identifier;
- authoritative learning material;
- discovered interests;
- identified topics;
- weak and strong topics;
- the latest evaluation score and mastery decision;
- up to 20 recent student and teacher messages;
- the most recent action;
- total LLM calls used; and
- the last update timestamp.

Sessions expire after one hour and can be deleted explicitly through `DELETE /api/session`.

This design satisfies the current temporary-memory requirement but is intentionally best effort on Vercel. Serverless instances can be recycled, scaled to zero, or run in parallel. Important cross-instance state must therefore move to Supabase or another external database.

# LLM call management

The maximum is 16 total calls per session:

- every `LearningSupervisor` decision counts as one call;
- every selected LLM-backed teaching tool counts as one additional call;
- local validation and state updates do not count; and
- once the limit is reached, the server returns a deterministic stop response without another model request.

Most turns use two calls. If only one call remains, the supervisor prompt requires `RespondDirectly` or `Stop`, with the complete answer included in that single decision.

Usage is exposed through:

- `X-LLM-Calls-Used`;
- `X-LLM-Calls-Remaining`; and
- `X-Agent-Session-Id`.

# LLMod integration

The project uses the course LLMod gateway at `https://api.llmod.ai`.

## Text generation

- Endpoint: `/v1/chat/completions`
- Model: `MB5R2CF-azure/gpt-5.4-mini`
- Expected output: structured JSON object
- Temperature: `0.25`
- Timeout: 120 seconds

## Embeddings

- Endpoint: `/v1/embeddings`
- Model: `MB5R2CF-azure/text-embedding-3-small`
- Verified dimension: 1,536
- Planned Pinecone metric: cosine similarity

`adaptive_teacher/llm.py` uses the asynchronous `httpx` client. The API key is read only by server-side Python and is never returned to the browser. During local development, a missing key activates deterministic demo responses so the interface remains testable without spending the course budget. Production fails closed if the key is missing unless demo mode was explicitly enabled.

# API contract

## GET /api/team_info

Returns the group batch/order identifier, team name, and three team members. Final email and batch/order values still need to be filled in.

## GET /api/agent_info

Returns the description, purpose, prompt template, and a full example containing the response and every traced step.

## GET /api/model_architecture

Returns the PNG architecture diagram. Its tool names match the runtime trace exactly.

## POST /api/execute

Request body:

```json
{
  "prompt": "Student request or plain-text learning material"
}
```

A successful response contains exactly these four top-level fields:

```json
{
  "status": "ok",
  "error": null,
  "response": "Agent response",
  "steps": []
}
```

An error response keeps the same four fields, sets `status` to `error`, provides a human-readable `error`, sets `response` to `null`, and returns an empty `steps` list.

## DELETE /api/session

Deletes the current temporary session, clears the cookie, and returns `{"status":"ok"}`.

# Environment and secret management

The two environment files have different roles:

- `.env.local` contains real local configuration and secrets. It is ignored by Git and is automatically loaded by the Python server for direct local development.
- `.env.example` contains only safe placeholder values and variable names. It is not loaded by the application. It documents configuration for teammates, clean clones, and Vercel.

The existing `.env.local` is already configured on the development machine and should not be overwritten. In Vercel, the same values must be added through Project Settings rather than committed to the repository.

# Local development and validation

Create a Python 3.12 virtual environment and install the development dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Start the site:

```bash
uvicorn app:app --reload --port 3000
```

Run the automated suite:

```bash
python -m pytest
```

The tests set `ADAPTIVE_TEACHER_DEMO_MODE=true` before importing the application. Therefore, they validate the interface, endpoints, session continuity, traces, input errors, concurrent call limiting, mastery persistence, and failure handling without making billable LLMod requests. A mocked HTTP transport also verifies the real LLMod URL, authorization header, payload, and structured-response parser without accessing the network.

# Planned Supabase and Pinecone design

## Supabase responsibilities

Supabase will hold document metadata and any session or ingestion records that must survive a single process. The persisted scope should remain minimal because the product decision specifies temporary learning memory.

## Pinecone responsibilities

Pinecone will store vectorized chunks under a temporary session namespace. The planned index uses 1,536 dimensions and cosine similarity.

The retrieval flow is:

1. receive plain-text material;
2. split it into bounded, overlapping chunks;
3. generate embeddings through LLMod;
4. upsert vectors and source metadata under the session namespace;
5. embed the current retrieval query;
6. retrieve the most relevant chunks;
7. give only those chunks and the relevant state to the selected tool; and
8. delete the namespace when the session is reset or expires.

The implementation must account for retries and partial failures so Supabase metadata and Pinecone vectors do not become inconsistent.

# Vercel deployment design

The root `app.py` is a recognized FastAPI entrypoint. Vercel can deploy it as one Python Function and serve files in `public/` through its static asset layer. No Node.js build step is needed.

Before production deployment:

1. connect Supabase and Pinecone;
2. add all private values in Vercel Environment Variables;
3. import the GitHub repository and deploy `main`;
4. verify the root interface without authentication;
5. verify all four required endpoints;
6. confirm that traces contain every actual LLM call; and
7. keep the Vercel project available until grading is complete.

Vercel's Python runtime uses a read-only function filesystem except for ephemeral `/tmp`, and separate instances do not share module memory. The application must not rely on local files or global dictionaries for durable production data.

# Security and operational considerations

- Never commit `.env.local` or any API key.
- Keep LLMod, Supabase, and Pinecone credentials server-side.
- Never place a secret in `public/` or browser JavaScript.
- Limit context to relevant state and retrieved source chunks.
- Avoid unnecessary model calls to protect the course budget.
- Add a durable global quota or rate limiter before public deployment; the 16-call session cap alone does not prevent users from creating additional sessions.
- Treat traces as potentially sensitive because they contain full prompts and model responses.
- Validate external-service errors and return the required human-readable API error shape.
- Delete temporary vector namespaces when a session ends or expires.

# Remaining work checklist

- Fill in final team email values and the group batch/order number.
- Create and connect the Supabase project and schema.
- Create a Pinecone index with 1,536 dimensions and cosine similarity.
- Implement chunking, upsert, query, retry, and namespace deletion.
- Implement durable project-wide LLMod usage limiting or rate limiting.
- Add the new Supabase and Pinecone variable names to `.env.example`.
- Add all production secrets to Vercel Project Settings.
- Deploy to Vercel and verify the GUI and required endpoints in production.
