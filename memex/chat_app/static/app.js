const STORAGE_KEY = "memex-chat-demo";

const state = {
  sessionId: crypto.randomUUID(),
  messages: [],
  memories: [],
  usedMemoryIds: new Set(),
};

const messagesEl = document.querySelector("#messages");
const messageTemplate = document.querySelector("#message-template");
const memoryTemplate = document.querySelector("#memory-template");
const memoryListEl = document.querySelector("#memory-list");
const usedMemoriesEl = document.querySelector("#used-memories");
const memoryCountEl = document.querySelector("#memory-count");
const providerEl = document.querySelector("#provider");
const composer = document.querySelector("#composer");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const refreshButton = document.querySelector("#refresh-memories");
const newSessionButton = document.querySelector("#new-session");

function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed.sessionId === "string") state.sessionId = parsed.sessionId;
    if (Array.isArray(parsed.messages)) state.messages = parsed.messages;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function saveState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      sessionId: state.sessionId,
      messages: state.messages,
    }),
  );
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderMessages() {
  messagesEl.replaceChildren();
  if (state.messages.length === 0) {
    appendMessage({
      role: "assistant",
      content:
        "Tell me something like 'My name is Riley and I prefer concise answers.' Then ask what I remember.",
      memoryNote: "No memories used yet.",
    });
    return;
  }
  for (const message of state.messages) appendMessage(message);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendMessage(message) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".message-role").textContent = message.role === "user" ? "You" : "Memex";
  node.querySelector(".message-body").textContent = message.content;
  node.querySelector(".message-memory").textContent = message.memoryNote ?? "";
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

function renderMemories({ createdId = "" } = {}) {
  memoryListEl.replaceChildren();
  memoryCountEl.textContent = `${state.memories.length} ${state.memories.length === 1 ? "memory" : "memories"}`;
  for (const memory of state.memories) {
    const node = memoryTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.id = memory.id;
    if (state.usedMemoryIds.has(memory.id)) node.classList.add("is-used");
    if (memory.id === createdId) node.classList.add("is-created");
    node.querySelector(".memory-type").textContent = memory.memory_type;
    node.querySelector(".memory-time").textContent = formatTime(memory.created_at);
    node.querySelector(".memory-text").textContent = memory.text;
    memoryListEl.appendChild(node);
  }
}

function renderUsedMemories(memories) {
  usedMemoriesEl.replaceChildren();
  if (memories.length === 0) {
    usedMemoriesEl.textContent = "No matching memory used.";
    return;
  }
  for (const memory of memories) {
    const item = document.createElement("div");
    item.className = "used-pill";
    item.textContent = `[${memory.memory_type}] ${memory.text}`;
    usedMemoriesEl.appendChild(item);
  }
}

async function loadHealth() {
  const response = await fetch("/api/health");
  const body = await response.json();
  providerEl.textContent = body.llm === "openai" ? "OpenAI streaming" : "Local demo model";
}

async function loadMemories() {
  const response = await fetch("/api/memories");
  if (!response.ok) throw new Error("Could not load memories.");
  const body = await response.json();
  state.memories = body.memories ?? [];
  renderMemories();
}

function upsertMemory(memory) {
  const existingIndex = state.memories.findIndex((item) => item.id === memory.id);
  if (existingIndex >= 0) state.memories.splice(existingIndex, 1);
  state.memories.unshift(memory);
  renderMemories({ createdId: memory.id });
}

async function sendMessage(message) {
  sendButton.disabled = true;
  input.disabled = true;

  const userMessage = { role: "user", content: message };
  state.messages.push(userMessage);
  appendMessage(userMessage);

  const assistantMessage = {
    role: "assistant",
    content: "",
    memoryNote: "Retrieving memory...",
  };
  state.messages.push(assistantMessage);
  const assistantNode = appendMessage(assistantMessage);
  const bodyEl = assistantNode.querySelector(".message-body");
  const memoryEl = assistantNode.querySelector(".message-memory");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message, session_id: state.sessionId }),
    });
    if (!response.ok || !response.body) throw new Error(await response.text());

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        handleStreamEvent(JSON.parse(line), assistantMessage, bodyEl, memoryEl);
      }
    }
  } catch (error) {
    assistantMessage.content = `Chat failed: ${error.message}`;
    bodyEl.textContent = assistantMessage.content;
    memoryEl.textContent = "";
  } finally {
    sendButton.disabled = false;
    input.disabled = false;
    input.focus();
    saveState();
  }
}

function handleStreamEvent(payload, assistantMessage, bodyEl, memoryEl) {
  const { event, data } = payload;
  if (event === "memory_context") {
    state.sessionId = data.session_id;
    const memories = data.memories ?? [];
    state.usedMemoryIds = new Set(memories.map((memory) => memory.id));
    renderUsedMemories(memories);
    renderMemories();
    assistantMessage.memoryNote =
      memories.length === 0 ? "No relevant memory used." : `Used ${memories.length} relevant memories.`;
    memoryEl.textContent = assistantMessage.memoryNote;
    return;
  }
  if (event === "memory_created") {
    upsertMemory(data.memory);
    return;
  }
  if (event === "token") {
    assistantMessage.content += data.text;
    bodyEl.textContent = assistantMessage.content;
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return;
  }
  if (event === "warning") {
    assistantMessage.memoryNote = `${assistantMessage.memoryNote} Provider warning: ${data.message}`;
    memoryEl.textContent = assistantMessage.memoryNote;
    return;
  }
  if (event === "done") {
    state.sessionId = data.session_id;
    assistantMessage.content = data.message;
    bodyEl.textContent = assistantMessage.content;
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || sendButton.disabled) return;
  input.value = "";
  sendMessage(message);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

refreshButton.addEventListener("click", () => {
  loadMemories().catch(() => undefined);
});

newSessionButton.addEventListener("click", () => {
  state.sessionId = crypto.randomUUID();
  state.messages = [];
  state.usedMemoryIds = new Set();
  saveState();
  renderUsedMemories([]);
  renderMessages();
});

loadState();
renderMessages();
renderUsedMemories([]);
loadHealth().catch(() => {
  providerEl.textContent = "Offline";
});
loadMemories().catch(() => {
  memoryCountEl.textContent = "Memory unavailable";
});
