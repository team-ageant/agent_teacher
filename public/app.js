"use strict";

const MAX_LLM_CALLS = 16;

const state = {
  messages: [],
  busy: false,
  resetting: false,
  usage: { used: 0, remaining: MAX_LLM_CALLS },
  openTrace: null,
  requestController: null,
};
 
const elements = {
  composer: document.querySelector("#composer"),
  prompt: document.querySelector("#prompt"),
  submit: document.querySelector("#submit-prompt"),
  reset: document.querySelector("#reset-session"),
  messages: document.querySelector("#messages"),
  empty: document.querySelector("#empty-state"),
  thinking: document.querySelector("#thinking"),
  scrollAnchor: document.querySelector("#scroll-anchor"),
  messageCount: document.querySelector("#message-count"),
  usage: document.querySelector("#usage"),
  usageText: document.querySelector("#usage-text"),
  error: document.querySelector("#error-banner"),
};

function createId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function setError(message) {
  elements.error.textContent = message || "";
  elements.error.hidden = !message;
}

function setBusy(busy) {
  state.busy = busy;
  elements.thinking.hidden = !busy;
  elements.messages.setAttribute("aria-busy", String(busy));
  updateSubmitState();
}

function updateSubmitState() {
  elements.submit.disabled = state.busy || state.resetting || !elements.prompt.value.trim();
}

function updateUsage() {
  const { used, remaining } = state.usage;
  elements.usageText.textContent = `${remaining} / ${MAX_LLM_CALLS} calls left`;
  elements.usage.setAttribute("aria-label", `${remaining} LLM calls remaining; ${used} used`);
}

function updateMessageCount() {
  const count = state.messages.length;
  elements.messageCount.textContent = count ? `${count} messages` : "Ready to learn";
}

function safeJson(value) {
  try {
    const serialized = JSON.stringify(value, null, 2);
    return serialized === undefined ? String(value) : serialized;
  } catch {
    return String(value);
  }
}

function addTraceField(container, heading, value) {
  const title = document.createElement("h4");
  title.textContent = heading;

  const content = document.createElement("pre");
  content.textContent = value == null ? "" : String(value);

  container.append(title, content);
}

function createTrace(message) {
  const trace = document.createElement("div");
  trace.className = "trace";
  trace.id = `trace-${message.id}`;

  message.steps.forEach((step, index) => {
    const details = document.createElement("details");
    details.open = true;

    const summary = document.createElement("summary");
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    summary.append(number, document.createTextNode(String(step?.module ?? "Unknown module")));

    const body = document.createElement("div");
    body.className = "trace-body";
    addTraceField(body, "System prompt", step?.prompt?.System_prompt);
    addTraceField(body, "User prompt", step?.prompt?.User_prompt);
    addTraceField(body, "Response", safeJson(step?.response));

    details.append(summary, body);
    trace.append(details);
  });

  return trace;
}

function createMessage(message) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;
  article.dataset.messageId = message.id;

  const label = document.createElement("div");
  label.className = "message-label";
  const role = document.createElement("span");
  role.textContent = message.role === "student" ? "YOU" : "ADAPTIVE TEACHER";
  label.append(role);

  if (Array.isArray(message.steps) && message.steps.length > 0) {
    const traceButton = document.createElement("button");
    const isOpen = state.openTrace === message.id;
    const noun = message.steps.length === 1 ? "step" : "steps";
    traceButton.type = "button";
    traceButton.textContent = `${message.steps.length} traced ${noun}`;
    traceButton.dataset.traceToggle = message.id;
    traceButton.setAttribute("aria-expanded", String(isOpen));
    traceButton.setAttribute("aria-controls", `trace-${message.id}`);
    label.append(traceButton);
  }

  const text = document.createElement("p");
  text.textContent = message.text;
  text.dir = "auto";

  article.append(label, text);
  if (state.openTrace === message.id && Array.isArray(message.steps)) {
    article.append(createTrace(message));
  }
  return article;
}

function renderMessages({ scroll = false } = {}) {
  elements.messages.querySelectorAll(".message").forEach((message) => message.remove());
  elements.empty.hidden = state.messages.length > 0;

  const fragment = document.createDocumentFragment();
  state.messages.forEach((message) => fragment.append(createMessage(message)));
  elements.messages.insertBefore(fragment, elements.thinking);
  updateMessageCount();

  if (scroll) {
    requestAnimationFrame(() => {
      elements.scrollAnchor.scrollIntoView({ behavior: "smooth", block: "end" });
    });
  }
}

function readUsageHeaders(response) {
  const used = Number(response.headers.get("X-LLM-Calls-Used"));
  const remaining = Number(response.headers.get("X-LLM-Calls-Remaining"));
  if (Number.isFinite(used) && Number.isFinite(remaining)) {
    state.usage = { used, remaining };
    updateUsage();
  }
}

async function readApiResult(response) {
  try {
    return await response.json();
  } catch {
    throw new Error(`The server returned an invalid response (HTTP ${response.status}).`);
  }
}

async function runAgent(event) {
  event.preventDefault();
  const value = elements.prompt.value.trim();
  if (!value || state.busy) return;

  state.messages.push({ id: createId(), role: "student", text: value });
  elements.prompt.value = "";
  setError("");
  setBusy(true);
  renderMessages({ scroll: true });

  const controller = new AbortController();
  state.requestController = controller;

  try {
    const response = await fetch("/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: value }),
      signal: controller.signal,
    });
    const result = await readApiResult(response);
    readUsageHeaders(response);

    if (!response.ok || result?.status === "error" || !result?.response) {
      throw new Error(result?.error || "The agent could not complete the request.");
    }

    state.messages.push({
      id: createId(),
      role: "teacher",
      text: String(result.response),
      steps: Array.isArray(result.steps) ? result.steps : [],
    });
    renderMessages({ scroll: true });
  } catch (error) {
    if (error?.name !== "AbortError") {
      setError(error instanceof Error ? error.message : "Unexpected error");
    }
  } finally {
    if (state.requestController === controller) {
      state.requestController = null;
      setBusy(false);
      renderMessages({ scroll: true });
    }
  }
}

async function resetSession() {
  if (state.resetting) return;
  state.requestController?.abort();
  state.requestController = null;
  setBusy(false);
  state.resetting = true;
  elements.reset.disabled = true;

  state.messages = [];
  state.usage = { used: 0, remaining: MAX_LLM_CALLS };
  state.openTrace = null;
  elements.prompt.value = "";
  setError("");
  updateUsage();
  renderMessages();
  updateSubmitState();

  try {
    await fetch("/api/session", { method: "DELETE" });
  } catch {
    // The local reset is still useful if the server cannot be reached.
  } finally {
    state.resetting = false;
    elements.reset.disabled = false;
    updateSubmitState();
    elements.prompt.focus();
  }
}

elements.composer.addEventListener("submit", runAgent);
elements.prompt.addEventListener("input", updateSubmitState);
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});
elements.reset.addEventListener("click", resetSession);

elements.messages.addEventListener("click", (event) => {
  const starter = event.target.closest("[data-starter]");
  if (starter) {
    elements.prompt.value = starter.dataset.starter;
    updateSubmitState();
    elements.prompt.focus();
    elements.prompt.setSelectionRange(elements.prompt.value.length, elements.prompt.value.length);
    return;
  }

  const traceButton = event.target.closest("[data-trace-toggle]");
  if (!traceButton) return;
  const messageId = traceButton.dataset.traceToggle;
  state.openTrace = state.openTrace === messageId ? null : messageId;
  renderMessages();
  elements.messages.querySelector(`[data-trace-toggle="${CSS.escape(messageId)}"]`)?.focus();
});

updateUsage();
updateMessageCount();
updateSubmitState();
