const messages = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  if (!text) {
    return "";
  }
  if (typeof marked === "undefined") {
    return escapeHtml(text).replace(/\n/g, "<br>");
  }
  const raw = marked.parse(text, { breaks: true, gfm: true });
  if (typeof DOMPurify !== "undefined") {
    return DOMPurify.sanitize(raw, { USE_PROFILES: { html: true } });
  }
  return raw;
}
const statusDot = document.getElementById("ws-status");
const statusLabel = document.getElementById("ws-label");
const sessionEl = document.getElementById("session-id");
const lotResultEl = document.getElementById("lot-result");
const matchSummaryCard = document.getElementById("match-summary-card");
const matchSummaryEl = document.getElementById("match-summary");
const latestLotCard = document.getElementById("latest-lot-card");
const latestLotEl = document.getElementById("latest-lot");
const simResultEl = document.getElementById("sim-result");
const predictionResultEl = document.getElementById("prediction-result");
const eventLogEl = document.getElementById("event-log");
const workflowNameEl = document.getElementById("workflow-name");
const workflowUpdatedEl = document.getElementById("workflow-updated");
const loadWorkflowJsonButton = document.getElementById("load-workflow-json");
const workflowJsonEl = document.getElementById("workflow-json");
const workflowListEl = document.getElementById("workflow-list");
const frontendTriggerEl = document.getElementById("frontend-trigger");
const currentLotEl = document.getElementById("current-lot");
const toggleLogButton = document.getElementById("toggle-log");
const lotIdInput = document.getElementById("lot-id");
const lotSearchButton = document.getElementById("lot-search");
const lotSimButton = document.getElementById("lot-simulate");
const lotSearchSimButton = document.getElementById("lot-search-sim");
const lotClearButton = document.getElementById("lot-clear");
const lotStatusEl = document.getElementById("lot-status");
const simFormCard = document.getElementById("sim-form-card");
const lotCard = document.getElementById("lot-card");
const simResultCard = document.getElementById("sim-result-card");
const defectChartCard = document.getElementById("defect-chart-card");
const defectChartTitleEl = defectChartCard
  ? defectChartCard.querySelector("h3")
  : null;
const designCandidatesCard = document.getElementById("design-candidates-card");
const finalBriefingCard = document.getElementById("final-briefing-card");
const predictionCard = document.getElementById("prediction-card");
const frontendCard = document.getElementById("frontend-card");
const eventLogCard = document.getElementById("event-log-card");
const eventEmptyEl = document.getElementById("event-empty");
const mainPanelEl = document.getElementById("main-panels");
const eventPanelEl = document.getElementById("event-panel");
const eventPanelShowButton = document.getElementById("event-panel-show");
const eventPanelHideButton = document.getElementById("event-panel-hide");
const simModelInput = document.getElementById("sim-model");
const simTemperatureInput = document.getElementById("sim-temperature");
const simVoltageInput = document.getElementById("sim-voltage");
const simSizeInput = document.getElementById("sim-size");
const simCapacityInput = document.getElementById("sim-capacity");
const simProductionSelect = document.getElementById("sim-production");
const simApplyButton = document.getElementById("sim-apply");
const simRunButton = document.getElementById("sim-run");
const simFormStatus = document.getElementById("sim-form-status");
const recommendationApplyButton = document.getElementById("recommend-apply");
const recommendationStatusEl = document.getElementById("recommend-status");
const defectChartEl = document.getElementById("defect-chart");
const finalDefectChartEl = document.getElementById("final-defect-chart");
const designCandidatesEl = document.getElementById("design-candidates");
const finalBriefingEl = document.getElementById("final-briefing");
const defaultLotResultText = lotResultEl ? lotResultEl.textContent : "";
const defaultMatchSummaryText = matchSummaryEl
  ? matchSummaryEl.textContent
  : "";
const defaultLatestLotText = latestLotEl ? latestLotEl.textContent : "";
const defaultSimResultText = simResultEl ? simResultEl.textContent : "";
const defaultPredictionResultText = predictionResultEl ? predictionResultEl.textContent : "";
const defaultFrontendTriggerText = frontendTriggerEl ? frontendTriggerEl.textContent : "";
const defaultCurrentLotText = currentLotEl ? currentLotEl.textContent : "";
const defaultSimFormStatusText = simFormStatus ? simFormStatus.textContent : "";
const defaultRecommendationStatusText = recommendationStatusEl
  ? recommendationStatusEl.textContent
  : "";
const defaultDefectChartText = defectChartEl ? defectChartEl.textContent : "";
const defaultFinalDefectChartText = finalDefectChartEl
  ? finalDefectChartEl.textContent
  : "";
const defaultDesignCandidatesText = designCandidatesEl
  ? designCandidatesEl.textContent
  : "";
const defaultFinalBriefingText = finalBriefingEl ? finalBriefingEl.textContent : "";
const sessionListEl = document.getElementById("session-list");
const sessionEmptyEl = document.getElementById("session-empty");
const sessionCountEl = document.getElementById("session-count");
const newSessionButton = document.getElementById("new-session");

const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
let socket = null;
let socketPingInterval = null;
let suppressSocketCloseNotice = false;
let activeSocketSessionId = "";

function generateSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    return Array.from(bytes, (value, index) => {
      const hex = value.toString(16).padStart(2, "0");
      if (index === 4 || index === 6 || index === 8 || index === 10) {
        return `-${hex}`;
      }
      return hex;
    }).join("");
  }
  return `fallback-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const CURRENT_SESSION_KEY = "demo_current_session";
const LEGACY_SESSION_KEY = "demo_session_id";
const SESSIONS_STORAGE_KEY = "demo_sessions";
const HISTORY_STORAGE_PREFIX = "demo_chat_history";
const HISTORY_LIMIT = 200;
const SESSION_TITLE_LIMIT = 32;
const DEFAULT_SESSION_TITLE = "새 대화";
const LEGACY_SESSION_TITLE = "새 채팅";

let sessions = [];
let currentSessionId = "";
sessionEl.textContent = "--";
let lastLotId = "";
let lastSimulationParams = {};
let lastMatchSummary = null;
let lastLatestLot = null;
let isComposing = false;
let isEventPanelOpen = false;
let recommendationDirty = false;
let hasFinalBriefing = false;
let activeChatStatusSource = "";
let statusMessageEl = null;
let statusMessageTextEl = null;
let statusMessageDotsEl = null;
let statusAnimationInterval = null;
let pipelineLogMessageEl = null;
let pipelineLogListEl = null;
let pipelineLogSpinnerEl = null;
let pipelineLogLabelEl = null;
const pipelineStageTables = new Map();
const pipelineStageLabels = {
  recommendation: "추천",
  reference: "레퍼런스",
  grid: "그리드",
  final: "브리핑",
  planner: "PLANNER",
};

let historyEntries = [];
let lastDefectChartPayload = null;
let lastFinalDefectChartPayload = null;
const streamingMessages = new Map();

function loadSessions() {
  try {
    const raw = localStorage.getItem(SESSIONS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    return [];
  }
}

function saveSessions(list) {
  try {
    localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(list));
  } catch (error) {
    // Ignore storage failures.
  }
}

function loadCurrentSessionId() {
  try {
    const stored = localStorage.getItem(CURRENT_SESSION_KEY);
    if (stored) {
      return stored;
    }
    const legacy = localStorage.getItem(LEGACY_SESSION_KEY);
    if (legacy) {
      localStorage.setItem(CURRENT_SESSION_KEY, legacy);
      return legacy;
    }
  } catch (error) {
    return "";
  }
  return "";
}

function saveCurrentSessionId(sessionId) {
  try {
    localStorage.setItem(CURRENT_SESSION_KEY, sessionId);
  } catch (error) {
    // Ignore storage failures.
  }
}

function loadHistory(sessionId) {
  try {
    const raw = localStorage.getItem(`${HISTORY_STORAGE_PREFIX}:${sessionId}`);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    return [];
  }
}

function saveHistory(sessionId, entries) {
  try {
    const trimmed = entries.slice(-HISTORY_LIMIT);
    localStorage.setItem(
      `${HISTORY_STORAGE_PREFIX}:${sessionId}`,
      JSON.stringify(trimmed)
    );
  } catch (error) {
    // Ignore storage failures.
  }
}

function clearChatMessages() {
  messages.innerHTML = "";
  clearChatStatus();
  clearPipelineLog();
}

function formatSessionTime(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const datePart = date.toLocaleDateString("ko-KR", {
    month: "short",
    day: "numeric",
  });
  const timePart = date.toLocaleTimeString("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${datePart} ${timePart}`;
}

function createSession(title = DEFAULT_SESSION_TITLE, id = generateSessionId()) {
  return {
    id,
    title,
    updatedAt: new Date().toISOString(),
    preview: "",
  };
}

function ensureSession(sessionId) {
  let session = sessions.find((item) => item.id === sessionId);
  if (!session) {
    session = createSession(DEFAULT_SESSION_TITLE, sessionId);
    sessions.unshift(session);
  }
  return session;
}

function renderSessionList(list) {
  if (!sessionListEl) {
    return;
  }
  sessionListEl.querySelectorAll(".session-item").forEach((item) => item.remove());
  if (sessionCountEl) {
    sessionCountEl.textContent = String(list.length);
  }
  if (!list.length) {
    if (sessionEmptyEl) {
      sessionEmptyEl.classList.remove("hidden");
    }
    return;
  }
  if (sessionEmptyEl) {
    sessionEmptyEl.classList.add("hidden");
  }
  const sorted = list
    .slice()
    .sort(
      (a, b) =>
        new Date(b.updatedAt || 0).getTime() - new Date(a.updatedAt || 0).getTime()
    );
  sorted.forEach((session) => {
    const item = document.createElement("div");
    item.className = "session-item";
    if (session.id === currentSessionId) {
      item.classList.add("active");
    }
    item.dataset.sessionId = session.id;
    const title = document.createElement("button");
    title.type = "button";
    title.className = "session-select";
    const displayTitle =
      session.title === LEGACY_SESSION_TITLE
        ? DEFAULT_SESSION_TITLE
        : session.title || DEFAULT_SESSION_TITLE;
    title.textContent = displayTitle;
    title.title = displayTitle;
    title.addEventListener("click", () => {
      setActiveSession(session.id, { focusInput: true });
    });
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "session-delete";
    deleteButton.setAttribute("aria-label", "대화 삭제");
    deleteButton.innerHTML =
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 6h12" stroke-width="1.5" stroke-linecap="round" /><path d="M8 6V4h4v2" stroke-width="1.5" stroke-linecap="round" /><path d="M7 6l1 10h4l1-10" stroke-width="1.5" stroke-linecap="round" /><path d="M9.5 9.5v4" stroke-width="1.5" stroke-linecap="round" /><path d="M11.5 9.5v4" stroke-width="1.5" stroke-linecap="round" /></svg>';
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(session.id);
    });
    item.appendChild(title);
    item.appendChild(deleteButton);
    sessionListEl.appendChild(item);
  });
}

function pickLatestSessionId() {
  if (!sessions.length) {
    return "";
  }
  const sorted = sessions
    .slice()
    .sort(
      (a, b) =>
        new Date(b.updatedAt || 0).getTime() - new Date(a.updatedAt || 0).getTime()
    );
  return sorted[0]?.id || "";
}

function deleteSession(sessionId) {
  if (!sessionId) {
    return;
  }
  sessions = sessions.filter((item) => item.id !== sessionId);
  saveSessions(sessions);
  try {
    localStorage.removeItem(`${HISTORY_STORAGE_PREFIX}:${sessionId}`);
  } catch (error) {
    // Ignore storage failures.
  }

  if (currentSessionId === sessionId) {
    const nextSessionId = pickLatestSessionId();
    if (nextSessionId) {
      setActiveSession(nextSessionId, { focusInput: false });
      return;
    }
    const session = createSession();
    sessions = [session];
    saveSessions(sessions);
    setActiveSession(session.id, { focusInput: false });
    return;
  }

  renderSessionList(sessions);
}

function setActiveSession(sessionId, options = {}) {
  if (!sessionId) {
    return;
  }
  const shouldSwitch = sessionId !== currentSessionId;
  currentSessionId = sessionId;
  saveCurrentSessionId(sessionId);
  sessionEl.textContent = sessionId.slice(0, 8);
  historyEntries = loadHistory(sessionId);
  clearChatMessages();
  if (shouldSwitch) {
    resetEventPanel();
  }
  historyEntries.forEach((entry) => appendMessageToChat(entry));
  const session = ensureSession(sessionId);
  if (!session.preview && historyEntries.length) {
    updateSessionMeta(sessionId, historyEntries[historyEntries.length - 1]);
  }
  renderSessionList(sessions);
  if (typeof connectWebSocket === "function" && shouldSwitch) {
    connectWebSocket(sessionId);
  } else if (typeof connectWebSocket === "function" && !socket) {
    connectWebSocket(sessionId);
  }
  if (options.focusInput && input) {
    input.focus();
  }
}

function updateSessionMeta(sessionId, entry) {
  const session = ensureSession(sessionId);
  session.updatedAt = entry.timestamp;
  if (entry.role !== "system") {
    session.preview = entry.text;
  }
  const hasDefaultTitle =
    !session.title ||
    session.title === DEFAULT_SESSION_TITLE ||
    session.title === LEGACY_SESSION_TITLE;
  if (entry.role === "user" && hasDefaultTitle) {
    session.title =
      entry.text.length > SESSION_TITLE_LIMIT
        ? `${entry.text.slice(0, SESSION_TITLE_LIMIT)}...`
        : entry.text;
  }
  saveSessions(sessions);
}

function buildHistoryEntry(role, text) {
  const id =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    id,
    role,
    text,
    timestamp: new Date().toISOString(),
  };
}

function insertMessageElement(message) {
  if (statusMessageEl && statusMessageEl.parentNode === messages) {
    messages.insertBefore(message, statusMessageEl);
  } else {
    messages.appendChild(message);
  }
  messages.scrollTop = messages.scrollHeight;
}

function appendMessageToChat(entry) {
  const message = document.createElement("div");
  message.className = `message ${entry.role}`;
  message.dataset.messageId = entry.id;
  message.innerHTML = renderMarkdown(entry.text);
  insertMessageElement(message);
}

function addMessage(role, text) {
  if (role === "assistant") {
    clearChatStatus();
  }
  const entry = buildHistoryEntry(role, text);
  historyEntries.push(entry);
  saveHistory(currentSessionId, historyEntries);
  appendMessageToChat(entry);
  updateSessionMeta(currentSessionId, entry);
  renderSessionList(sessions);
}

function normalizeStreamMessageId(messageId) {
  if (messageId) {
    return messageId;
  }
  return `stream-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getOrCreateStreamingMessage(messageId) {
  const normalizedId = normalizeStreamMessageId(messageId);
  if (streamingMessages.has(normalizedId)) {
    return streamingMessages.get(normalizedId);
  }
  clearChatStatus();
  const entry = {
    id: normalizedId,
    role: "assistant",
    text: "",
    timestamp: new Date().toISOString(),
  };
  historyEntries.push(entry);
  const messageEl = document.createElement("div");
  messageEl.className = "message assistant streaming";
  messageEl.dataset.messageId = normalizedId;
  insertMessageElement(messageEl);
  const streamState = {
    entry,
    messageEl,
    blocks: [],
    blockElements: new Map(),
  };
  streamingMessages.set(normalizedId, streamState);
  return streamState;
}

function syncStreamingEntry(streamState) {
  streamState.entry.text = streamState.blocks.join("\n\n");
}

function startStreamingBlock(messageId, blockId) {
  const streamState = getOrCreateStreamingMessage(messageId);
  const normalizedBlockId =
    blockId || `block-${streamState.blocks.length + 1}`;
  const blockEl = document.createElement("div");
  blockEl.className = "stream-block stream-text";
  streamState.messageEl.appendChild(blockEl);
  const index = streamState.blocks.length;
  streamState.blocks.push("");
  streamState.blockElements.set(normalizedBlockId, { index, element: blockEl });
  messages.scrollTop = messages.scrollHeight;
  return { streamState, blockId: normalizedBlockId };
}

function appendStreamingDelta(messageId, blockId, delta) {
  if (!delta) {
    return;
  }
  const streamState = getOrCreateStreamingMessage(messageId);
  let blockInfo = streamState.blockElements.get(blockId || "");
  if (!blockInfo) {
    const created = startStreamingBlock(messageId, blockId);
    blockInfo = streamState.blockElements.get(created.blockId);
  }
  if (!blockInfo) {
    return;
  }
  streamState.blocks[blockInfo.index] =
    (streamState.blocks[blockInfo.index] || "") + delta;
  blockInfo.element.innerHTML = renderMarkdown(
    streamState.blocks[blockInfo.index]
  );
  syncStreamingEntry(streamState);
  messages.scrollTop = messages.scrollHeight;
}

function appendStreamingTable(messageId, markdown, animate = true) {
  if (!markdown) {
    return;
  }
  const streamState = getOrCreateStreamingMessage(messageId);
  const tableEl = document.createElement("div");
  tableEl.className = "stream-block stream-table";
  if (animate) {
    tableEl.classList.add("animate-in");
  }
  tableEl.innerHTML = renderMarkdown(markdown);
  streamState.messageEl.appendChild(tableEl);
  streamState.blocks.push(markdown);
  syncStreamingEntry(streamState);
  messages.scrollTop = messages.scrollHeight;
}

function finalizeStreamingMessage(messageId) {
  if (!messageId) {
    return;
  }
  const streamState = streamingMessages.get(messageId);
  if (!streamState) {
    return;
  }
  syncStreamingEntry(streamState);
  saveHistory(currentSessionId, historyEntries);
  streamState.messageEl.classList.remove("streaming");
  streamingMessages.delete(normalizedId);
  renderSessionList(sessions);
}

function initializeSessions() {
  sessions = loadSessions();
  currentSessionId = loadCurrentSessionId();
  if (!currentSessionId && sessions.length) {
    currentSessionId = sessions[0].id;
  }
  if (!currentSessionId) {
    const session = createSession();
    sessions = [session];
    currentSessionId = session.id;
  }
  ensureSession(currentSessionId);
  saveSessions(sessions);
  saveCurrentSessionId(currentSessionId);
  setActiveSession(currentSessionId, { focusInput: false });
}

initializeSessions();
if (newSessionButton) {
  newSessionButton.addEventListener("click", () => {
    const session = createSession();
    sessions.unshift(session);
    saveSessions(sessions);
    setActiveSession(session.id, { focusInput: true });
  });
}

function addEventLog(label, detail) {
  if (!eventLogEl) {
    return;
  }
  const row = document.createElement("div");
  row.textContent = `${new Date().toLocaleTimeString("ko-KR")} - ${label}: ${detail}`;
  eventLogEl.prepend(row);
}

function ensurePipelineLogMessage() {
  if (!messages) {
    return null;
  }
  if (!pipelineLogMessageEl) {
    pipelineLogMessageEl = document.createElement("div");
    pipelineLogMessageEl.className = "message assistant status pipeline-log";
    const header = document.createElement("div");
    header.className = "pipeline-log-header";
    pipelineLogLabelEl = document.createElement("span");
    pipelineLogLabelEl.className = "pipeline-log-title";
    pipelineLogLabelEl.textContent = "응답 생성 중";
    const hint = document.createElement("span");
    hint.className = "pipeline-log-hint";
    hint.textContent = "단계별 로그";
    header.appendChild(pipelineLogLabelEl);
    header.appendChild(hint);
    pipelineLogListEl = document.createElement("div");
    pipelineLogListEl.className = "pipeline-log-list";
    pipelineLogMessageEl.appendChild(header);
    pipelineLogMessageEl.appendChild(pipelineLogListEl);
  }
  const statusAnchor =
    statusMessageEl && statusMessageEl.parentNode === messages
      ? statusMessageEl
      : null;
  const streamingAnchor = messages.querySelector(
    ".message.assistant.streaming"
  );
  const anchor = streamingAnchor || statusAnchor;
  if (anchor) {
    messages.insertBefore(pipelineLogMessageEl, anchor);
  } else {
    messages.appendChild(pipelineLogMessageEl);
  }
  return pipelineLogMessageEl;
}

function clearPipelineLog() {
  if (pipelineLogMessageEl && pipelineLogMessageEl.parentNode) {
    pipelineLogMessageEl.parentNode.removeChild(pipelineLogMessageEl);
  }
  pipelineLogMessageEl = null;
  pipelineLogListEl = null;
  pipelineLogSpinnerEl = null;
  pipelineLogLabelEl = null;
  pipelineStageTables.clear();
}

function setPipelineLogComplete() {
  if (pipelineLogLabelEl) {
    pipelineLogLabelEl.textContent = "응답 완료";
  }
  if (pipelineLogSpinnerEl) {
    pipelineLogSpinnerEl.classList.add("is-hidden");
  }
  if (pipelineLogMessageEl) {
    pipelineLogMessageEl.classList.add("is-complete");
  }
}

function isPipelineErrorMessage(message) {
  if (!message) {
    return false;
  }
  return /실패|오류|에러|없음|없습니다|부족|중단|불가|찾지 못|못했습니다/i.test(
    message
  );
}

function renderPipelineStageTables(stage, detailEl) {
  if (!detailEl) {
    return;
  }
  detailEl.innerHTML = "";
  const payload = pipelineStageTables.get(stage);
  if (!payload) {
    if (stage === "planner") {
      return;
    }
    const empty = document.createElement("div");
    empty.className = "pipeline-log-empty";
    empty.textContent = stage === "recommendation" ? "표 없음" : "표 준비 중";
    detailEl.appendChild(empty);
    return;
  }
  const tables = Array.isArray(payload?.tables) ? payload.tables : [];
  const notes = Array.isArray(payload?.notes) ? payload.notes : [];
  if (!tables.length && !notes.length) {
    if (stage === "planner") {
      return;
    }
    const empty = document.createElement("div");
    empty.className = "pipeline-log-empty";
    empty.textContent = "표 준비 중";
    detailEl.appendChild(empty);
    return;
  }
  notes.forEach((note) => {
    if (!note) {
      return;
    }
    const noteEl = document.createElement("div");
    noteEl.className = "pipeline-log-note";
    noteEl.textContent = note;
    detailEl.appendChild(noteEl);
  });
  tables.forEach((table) => {
    const wrapper = document.createElement("div");
    wrapper.className = "pipeline-log-table";
    const title = table?.title ? String(table.title) : "";
    if (title) {
      const titleEl = document.createElement("div");
      titleEl.className = "pipeline-log-table-title";
      titleEl.textContent = title;
      wrapper.appendChild(titleEl);
    }
    const body = document.createElement("div");
    body.className = "pipeline-log-table-body";
    body.innerHTML = renderMarkdown(String(table?.markdown || ""));
    wrapper.appendChild(body);
    detailEl.appendChild(wrapper);
  });
}

function updatePipelineStageTables(stage, payload) {
  if (!stage) {
    return;
  }
  pipelineStageTables.set(stage, payload);
  if (!pipelineLogListEl) {
    return;
  }
  pipelineLogListEl
    .querySelectorAll(`.pipeline-log-entry[data-stage="${stage}"]`)
    .forEach((entry) => {
      const detail = entry.querySelector(".pipeline-log-detail");
      if (detail) {
        renderPipelineStageTables(stage, detail);
      }
    });
}

function appendPipelineStatus(stage, message, done = false) {
  if (!message) {
    return;
  }
  ensurePipelineLogMessage();
  if (!pipelineLogListEl) {
    return;
  }
  clearChatStatus("chat");
  const isError = isPipelineErrorMessage(message);
  const isRunning = !done && !isError;
  const cleanedMessage = isRunning
    ? message.replace(/\s*\.{1,3}\s*$/, "")
    : message;
  const stageKey = stage || "";
  const existingEntries = stageKey
    ? pipelineLogListEl.querySelectorAll(
        `.pipeline-log-entry[data-stage="${stageKey}"]`
      )
    : [];
  const lastEntry =
    existingEntries.length > 0
      ? existingEntries[existingEntries.length - 1]
      : null;
  if (lastEntry) {
    lastEntry.classList.toggle("is-done", done && !isError);
    lastEntry.classList.toggle("is-running", isRunning);
    lastEntry.classList.toggle("is-error", isError);
    const textEl = lastEntry.querySelector(".pipeline-log-text");
    if (textEl) {
      textEl.textContent = cleanedMessage;
    }
    const stateEl = lastEntry.querySelector(".pipeline-log-state");
    if (stateEl) {
      stateEl.textContent = isError ? "오류" : done ? "완료" : "진행";
    }
  } else {
    const entry = document.createElement("div");
    entry.className = "pipeline-log-entry";
    entry.dataset.stage = stageKey;
    if (done && !isError) {
      entry.classList.add("is-done");
    }
    if (isRunning) {
      entry.classList.add("is-running");
    }
    if (isError) {
      entry.classList.add("is-error");
    }

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "pipeline-log-toggle";
    const stageLabel = pipelineStageLabels[stage] || stage || "단계";
    const stageEl = document.createElement("span");
    stageEl.className = "pipeline-log-stage";
    stageEl.textContent = stageLabel;
    const textEl = document.createElement("span");
    textEl.className = "pipeline-log-text";
    textEl.textContent = cleanedMessage;
    const stateEl = document.createElement("span");
    stateEl.className = "pipeline-log-state";
    stateEl.textContent = isError ? "오류" : done ? "완료" : "진행";

    toggle.appendChild(stageEl);
    toggle.appendChild(textEl);
    toggle.appendChild(stateEl);

    const detail = document.createElement("div");
    detail.className = "pipeline-log-detail hidden";

    toggle.addEventListener("click", () => {
      const isOpen = entry.classList.toggle("is-open");
      detail.classList.toggle("hidden", !isOpen);
      if (isOpen) {
        renderPipelineStageTables(stage, detail);
      }
    });

    entry.appendChild(toggle);
    entry.appendChild(detail);
    pipelineLogListEl.appendChild(entry);
    messages.scrollTop = messages.scrollHeight;
  }

  if (done && stage === "grid") {
    setPipelineLogComplete();
  }
}

function setChatStatus(message, source = "system") {
  if (!message) {
    clearChatStatus(source);
    return;
  }
  if (
    source === "chat" &&
    pipelineLogMessageEl &&
    !pipelineLogMessageEl.classList.contains("is-complete")
  ) {
    return;
  }
  if (activeChatStatusSource === "pipeline" && source === "chat") {
    return;
  }
  activeChatStatusSource = source;
  if (!statusMessageEl) {
    statusMessageEl = document.createElement("div");
    statusMessageEl.className = "message assistant status";
    statusMessageTextEl = document.createElement("span");
    statusMessageTextEl.className = "status-text";
    statusMessageEl.appendChild(statusMessageTextEl);
    messages.appendChild(statusMessageEl);
  }
  const baseMessage = message.replace(/\s*\.{1,3}\s*$/, "") || message;
  if (statusMessageTextEl) {
    statusMessageTextEl.textContent = baseMessage;
  }
  messages.scrollTop = messages.scrollHeight;
}

function clearChatStatus(source = "") {
  if (source && activeChatStatusSource && source !== activeChatStatusSource) {
    return;
  }
  if (statusMessageEl && statusMessageEl.parentNode) {
    statusMessageEl.parentNode.removeChild(statusMessageEl);
  }
  statusMessageEl = null;
  statusMessageTextEl = null;
  statusMessageDotsEl = null;
  if (statusAnimationInterval) {
    clearInterval(statusAnimationInterval);
    statusAnimationInterval = null;
  }
  activeChatStatusSource = "";
}

function setLotStatus(message, isError = false) {
  if (!lotStatusEl) {
    return;
  }
  lotStatusEl.textContent = message;
  lotStatusEl.classList.toggle("error", isError);
  if (lotIdInput) {
    lotIdInput.classList.toggle("error", isError);
  }
}

function setSimStatus(message, isError = false) {
  if (!simFormStatus) {
    return;
  }
  simFormStatus.textContent = message;
  simFormStatus.classList.toggle("error", isError);
}

function setRecommendationStatus(message, isError = false) {
  if (!recommendationStatusEl) {
    return;
  }
  recommendationStatusEl.textContent = message;
  recommendationStatusEl.classList.toggle("error", isError);
}

function setCurrentLot(lotId) {
  if (!lotId) {
    return;
  }
  lastLotId = lotId;
  if (currentLotEl) {
    currentLotEl.textContent = `현재 LOT: ${lotId}`;
  }
  if (lotIdInput && document.activeElement !== lotIdInput) {
    lotIdInput.value = lotId;
  }
}

function resetEventPanel() {
  hideAllStreamCards();
  clearPipelineLog();
  lastDefectChartPayload = null;
  lastFinalDefectChartPayload = null;
  if (eventLogEl) {
    eventLogEl.innerHTML = "";
  }
  if (matchSummaryEl) {
    matchSummaryEl.textContent = defaultMatchSummaryText;
  }
  if (latestLotEl) {
    latestLotEl.textContent = defaultLatestLotText;
  }
  if (lotResultEl) {
    lotResultEl.textContent = defaultLotResultText;
  }
  if (simResultEl) {
    simResultEl.textContent = defaultSimResultText;
  }
  if (predictionResultEl) {
    predictionResultEl.textContent = defaultPredictionResultText;
  }
  if (defectChartEl) {
    defectChartEl.textContent = defaultDefectChartText;
  }
  if (finalDefectChartEl) {
    finalDefectChartEl.textContent = defaultFinalDefectChartText;
    finalDefectChartEl.style.display = "";
  }
  if (designCandidatesEl) {
    designCandidatesEl.textContent = defaultDesignCandidatesText;
  }
  if (finalBriefingEl) {
    finalBriefingEl.textContent = defaultFinalBriefingText;
  }
  if (frontendTriggerEl) {
    frontendTriggerEl.textContent = defaultFrontendTriggerText;
  }
  if (currentLotEl) {
    currentLotEl.textContent = defaultCurrentLotText;
  }
  if (lotStatusEl) {
    lotStatusEl.textContent = "";
    lotStatusEl.classList.remove("error");
  }
  if (simFormStatus) {
    simFormStatus.textContent = defaultSimFormStatusText;
    simFormStatus.classList.remove("error");
  }
  if (recommendationStatusEl) {
    recommendationStatusEl.textContent = defaultRecommendationStatusText;
    recommendationStatusEl.classList.remove("error");
  }
  if (simModelInput) {
    simModelInput.value = "";
  }
  if (simTemperatureInput) {
    simTemperatureInput.value = "";
  }
  if (simVoltageInput) {
    simVoltageInput.value = "";
  }
  if (simSizeInput) {
    simSizeInput.value = "";
  }
  if (simCapacityInput) {
    simCapacityInput.value = "";
  }
  if (simProductionSelect) {
    simProductionSelect.value = "";
  }
  updateMissingInputs([]);
  updateFilledInputs();
  lastLotId = "";
  lastSimulationParams = {};
  lastMatchSummary = null;
  lastLatestLot = null;
  recommendationDirty = false;
  hasFinalBriefing = false;
  if (isLogOpen) {
    toggleEventLog();
  }
  closeEventPanel();
  updateEventEmpty();
}
const streamCards = [
  matchSummaryCard,
  latestLotCard,
  lotCard,
  simFormCard,
  simResultCard,
  defectChartCard,
  designCandidatesCard,
  finalBriefingCard,
  predictionCard,
  frontendCard,
].filter(Boolean);
let isLogOpen = false;
if (eventLogCard) {
  isLogOpen = false;
  eventLogCard.classList.add("hidden");
  if (toggleLogButton && isLogOpen) {
    toggleLogButton.textContent = "로그 닫기";
  }
}

function updateEventEmpty() {
  if (!eventEmptyEl) {
    return;
  }
  const hasVisible = streamCards.some(
    (card) => card && !card.classList.contains("hidden")
  );
  const hasLogs = eventLogEl && eventLogEl.children.length > 0;
  eventEmptyEl.classList.toggle("hidden", hasVisible || isLogOpen || hasLogs);
}

function setEventPanelOpen(shouldOpen) {
  if (!mainPanelEl || !eventPanelEl) {
    return;
  }
  isEventPanelOpen = Boolean(shouldOpen);
  mainPanelEl.classList.toggle("event-open", isEventPanelOpen);
  mainPanelEl.classList.toggle("event-collapsed", !isEventPanelOpen);
  eventPanelEl.setAttribute("aria-hidden", String(!isEventPanelOpen));
  if (eventPanelShowButton) {
    eventPanelShowButton.setAttribute("aria-hidden", String(isEventPanelOpen));
  }
  if (eventPanelHideButton) {
    eventPanelHideButton.setAttribute("aria-hidden", String(!isEventPanelOpen));
  }
  if (isEventPanelOpen) {
    maybeOpenEventLog();
  }
}

function maybeOpenEventLog() {
  if (!eventLogCard || !eventLogEl) {
    return;
  }
  const hasVisibleCards = streamCards.some(
    (card) => card && !card.classList.contains("hidden")
  );
  const hasLogs = eventLogEl.children.length > 0;
  if (!hasVisibleCards && hasLogs && !isLogOpen) {
    toggleEventLog();
  }
}

function openEventPanel() {
  setEventPanelOpen(true);
}

function closeEventPanel() {
  setEventPanelOpen(false);
}

function hideAllStreamCards() {
  streamCards.forEach((card) => {
    if (card) {
      card.classList.add("hidden");
    }
  });
}

function showStreamCard(card) {
  if (!card) {
    return;
  }
  card.classList.remove("hidden");
  openEventPanel();
  updateEventEmpty();
}

function showOnlyStreamCards(cards = []) {
  hideAllStreamCards();
  const visibleCards = cards.filter(Boolean);
  visibleCards.forEach((card) => {
    card.classList.remove("hidden");
  });
  if (visibleCards.length) {
    openEventPanel();
  }
  updateEventEmpty();
}

function focusStage(stage) {
  const stageCards = {
    recommendation: [simResultCard],
    reference: [matchSummaryCard, latestLotCard, lotCard, defectChartCard],
    grid: [designCandidatesCard],
    final: [finalBriefingCard],
  };
  const cards = stageCards[stage];
  if (!cards) {
    return;
  }
  showOnlyStreamCards(cards.filter(Boolean));
}

function toggleEventLog() {
  if (!eventLogCard || !toggleLogButton) {
    return;
  }
  isLogOpen = !isLogOpen;
  eventLogCard.classList.toggle("hidden", !isLogOpen);
  if (isLogOpen) {
    openEventPanel();
  }
  toggleLogButton.textContent = isLogOpen ? "로그 닫기" : "이벤트 로그";
  updateEventEmpty();
}

function renderValue(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : "-";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function getRowValue(row, keys) {
  if (!row) {
    return null;
  }
  for (const key of keys) {
    if (row[key] !== undefined && row[key] !== null) {
      return row[key];
    }
  }
  return null;
}

function formatTimestamp(value) {
  if (!value) {
    return null;
  }
  const text = String(value);
  if (text.includes("T")) {
    return text.split("T")[1]?.replace("Z", "") || text;
  }
  return text;
}

function formatProductionMode(value) {
  if (!value) {
    return "-";
  }
  const normalized = String(value).toLowerCase();
  if (["양산", "개발"].includes(String(value))) {
    return String(value);
  }
  if (["mass", "production", "prod"].includes(normalized)) {
    return "양산";
  }
  if (["dev", "development"].includes(normalized)) {
    return "개발";
  }
  return String(value);
}

function formatMissingFields(missing = []) {
  if (!Array.isArray(missing) || !missing.length) {
    return "";
  }
  return missing
    .map((item) => {
      if (item === "production_mode") {
        return "양산/개발";
      }
      if (item === "temperature") return "온도";
      if (item === "voltage") return "전압";
      if (item === "size") return "크기";
      if (item === "capacity") return "용량";
      if (item === "chip_prod_id") return "기종";
      return item;
    })
    .join(", ");
}

function createKpiCard(label, value, className = "") {
  const card = document.createElement("div");
  card.className = `kpi-card ${className}`.trim();
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = renderValue(value);
  card.appendChild(span);
  card.appendChild(strong);
  return card;
}

function applyInputValue(input, value) {
  if (!input) {
    return;
  }
  if (document.activeElement === input) {
    return;
  }
  if (value === null || value === undefined || value === "") {
    if (!input.value) {
      input.value = "";
    }
    return;
  }
  input.value = String(value);
}

function setLastSimulationParams(params = {}) {
  if (!params || typeof params !== "object") {
    return;
  }
  const keys = [
    "chip_prod_id",
    "temperature",
    "voltage",
    "size",
    "capacity",
    "production_mode",
  ];
  const hasChange = keys.some(
    (key) => params[key] !== lastSimulationParams?.[key]
  );
  if (hasChange) {
    lastMatchSummary = null;
  }
  lastSimulationParams = { ...params };
}

function renderMatchSummary(payload = {}) {
  if (!matchSummaryEl) {
    return;
  }

  const summary = {
    chip_prod_id: payload.chip_prod_id || lastMatchSummary?.chip_prod_id || "",
    chip_prod_id_count:
      payload.chip_prod_id_count ?? lastMatchSummary?.chip_prod_id_count,
    chip_prod_id_samples: Array.isArray(payload.chip_prod_id_samples)
      ? payload.chip_prod_id_samples
      : lastMatchSummary?.chip_prod_id_samples || [],
    missing: Array.isArray(payload.missing)
      ? payload.missing
      : lastMatchSummary?.missing || [],
  };
  lastMatchSummary = summary;

  matchSummaryEl.innerHTML = "";
  const kpiRow = document.createElement("div");
  kpiRow.className = "kpi-row";
  if (summary.chip_prod_id) {
    kpiRow.appendChild(createKpiCard("선택 Chip", summary.chip_prod_id, "accent"));
  }
  if (Number.isFinite(summary.chip_prod_id_count)) {
    kpiRow.appendChild(createKpiCard("Chip 후보", summary.chip_prod_id_count));
  }
  if (kpiRow.childElementCount) {
    matchSummaryEl.appendChild(kpiRow);
  }

  const params = lastSimulationParams || {};
  const paramItems = [
    { label: "기종", value: params.chip_prod_id },
    { label: "온도", value: params.temperature },
    { label: "전압", value: params.voltage },
    { label: "크기", value: params.size },
    { label: "용량", value: params.capacity },
    { label: "구분", value: formatProductionMode(params.production_mode) },
  ];

  const inputSection = document.createElement("div");
  inputSection.className = "summary-section";
  const inputLabel = document.createElement("div");
  inputLabel.className = "summary-label";
  inputLabel.textContent = "입력 조건";
  inputSection.appendChild(inputLabel);
  const inputPills = document.createElement("div");
  inputPills.className = "summary-pills";

  let inputCount = 0;
  paramItems.forEach(({ label, value }) => {
    const display = value ?? "-";
    if (label === "기종" && (display === "-" || display === "")) {
      return;
    }
    const pill = document.createElement("span");
    pill.className = "pill summary-pill";
    pill.textContent = `${label} ${renderValue(display)}`;
    inputPills.appendChild(pill);
    inputCount += 1;
  });
  if (inputCount) {
    inputSection.appendChild(inputPills);
  } else {
    const empty = document.createElement("div");
    empty.className = "summary-note";
    empty.textContent = "입력 조건이 아직 없습니다.";
    inputSection.appendChild(empty);
  }
  const missingText = formatMissingFields(summary.missing);
  const note = document.createElement("div");
  note.className = "summary-note";
  note.textContent = missingText
    ? `입력 부족: ${missingText}`
    : "매칭 방식: 정확 일치";
  inputSection.appendChild(note);
  matchSummaryEl.appendChild(inputSection);

  const chipSection = document.createElement("div");
  chipSection.className = "summary-section";
  const chipLabel = document.createElement("div");
  chipLabel.className = "summary-label";
  chipLabel.textContent = "chip_prod_id 샘플";
  chipSection.appendChild(chipLabel);

  if (summary.chip_prod_id_samples.length) {
    const chipPills = document.createElement("div");
    chipPills.className = "summary-pills";
    summary.chip_prod_id_samples.forEach((chip) => {
      const pill = document.createElement("span");
      pill.className = "pill summary-pill summary-chip";
      pill.textContent = chip;
      chipPills.appendChild(pill);
    });
    chipSection.appendChild(chipPills);
  } else {
    const empty = document.createElement("div");
    empty.className = "summary-note";
    empty.textContent =
      summary.chip_prod_id_count === 0
        ? "일치하는 chip_prod_id가 없습니다."
        : "샘플을 준비 중입니다.";
    chipSection.appendChild(empty);
  }
  matchSummaryEl.appendChild(chipSection);
}

const LATEST_LOT_FIELDS = [
  {
    label: "양산/개발",
    keys: ["lot_type", "production_mode"],
    formatter: formatProductionMode,
  },
  { label: "온도", keys: ["temperature", "temp", "temp_grade"] },
  { label: "전압", keys: ["voltage", "volt"] },
  { label: "크기", keys: ["size", "chip_size"] },
  { label: "용량", keys: ["capacity", "capacitance"] },
  { label: "활성층", keys: ["active_layer"] },
  { label: "Sheet T", keys: ["sheet_t"] },
  { label: "Laydown", keys: ["laydown", "ldn_avr_value"] },
];

function renderLatestLot(payload = {}) {
  if (!latestLotEl) {
    return;
  }

  const rows = Array.isArray(payload.rows) ? payload.rows : [];
  const row = rows[0] || {};
  const lotId =
    payload.lot_id || getRowValue(row, ["lot_id", "lotid", "lot"]);
  const chipId = getRowValue(row, [
    "chip_prod_id",
    "chip_prod",
  ]);
  const designInputDate = getRowValue(row, ["design_input_date"]);

  lastLatestLot = { lot_id: lotId, chip_prod_id: chipId, row };

  latestLotEl.innerHTML = "";
  if (!row || !Object.keys(row).length) {
    latestLotEl.textContent = "최신 LOT 정보가 없습니다.";
    return;
  }

  const header = document.createElement("div");
  header.className = "lot-header";
  const fetchedAt = new Date().toLocaleTimeString("ko-KR");
  const sourceText = payload.source ? `source: ${payload.source}` : "";
  header.innerHTML = `
    <strong>${lotId || "LOT 정보"}</strong>
    <span>${sourceText ? `${sourceText} · ` : ""}조회 ${fetchedAt}</span>
  `;
  latestLotEl.appendChild(header);

  const kpiRow = document.createElement("div");
  kpiRow.className = "kpi-row";
  if (lotId) {
    kpiRow.appendChild(createKpiCard("LOT", lotId, "accent"));
  }
  if (chipId) {
    kpiRow.appendChild(createKpiCard("Chip", chipId));
  }
  if (designInputDate) {
    kpiRow.appendChild(createKpiCard("입력일", designInputDate));
  }
  if (kpiRow.childElementCount) {
    latestLotEl.appendChild(kpiRow);
  }

  const details = document.createElement("dl");
  details.className = "lot-grid latest-lot-grid";
  let detailCount = 0;
  LATEST_LOT_FIELDS.forEach((field) => {
    const raw = getRowValue(row, field.keys);
    if (raw === null || raw === undefined || raw === "") {
      return;
    }
    const dt = document.createElement("dt");
    dt.textContent = field.label;
    const dd = document.createElement("dd");
    const display = field.formatter ? field.formatter(raw) : renderValue(raw);
    dd.textContent = display;
    details.appendChild(dt);
    details.appendChild(dd);
    detailCount += 1;
  });
  if (detailCount) {
    latestLotEl.appendChild(details);
  } else {
    const empty = document.createElement("div");
    empty.className = "summary-note";
    empty.textContent = "표시할 상세 필드가 없습니다.";
    latestLotEl.appendChild(empty);
  }
}

function updateMissingInputs(missing = []) {
  const missingSet = new Set(missing || []);
  const mapping = [
    { key: "chip_prod_id", el: simModelInput },
    { key: "temperature", el: simTemperatureInput },
    { key: "voltage", el: simVoltageInput },
    { key: "size", el: simSizeInput },
    { key: "capacity", el: simCapacityInput },
    { key: "production_mode", el: simProductionSelect },
  ];
  mapping.forEach(({ key, el }) => {
    if (!el) {
      return;
    }
    el.classList.toggle("missing", missingSet.has(key));
  });
}

function updateFilledInputs() {
  const mapping = [
    simModelInput,
    simTemperatureInput,
    simVoltageInput,
    simSizeInput,
    simCapacityInput,
    simProductionSelect,
  ];
  mapping.forEach((el) => {
    if (!el) {
      return;
    }
    const value = String(el.value || "").trim();
    el.classList.toggle("filled", Boolean(value));
  });
}

function renderSimulationForm(payload = {}) {
  if (!payload || !simFormCard) {
    return;
  }
  hasFinalBriefing = false;
  showOnlyStreamCards([simFormCard]);
  const params = payload.params || {};
  setLastSimulationParams(params);
  applyInputValue(simModelInput, params.chip_prod_id);
  applyInputValue(simTemperatureInput, params.temperature);
  applyInputValue(simVoltageInput, params.voltage);
  applyInputValue(simSizeInput, params.size);
  applyInputValue(simCapacityInput, params.capacity);
  applyInputValue(simProductionSelect, params.production_mode);
  renderMatchSummary({ missing: payload.missing || [] });
  updateMissingInputs(payload.missing || []);
  updateFilledInputs();

  if (payload.missing && payload.missing.length) {
    const missingLabel = formatMissingFields(payload.missing);
    setSimStatus(`미입력: ${missingLabel}`, true);
  } else {
    setSimStatus("입력 완료: 추천 실행이 가능합니다.");
  }
}

function collectSimFormParams() {
  const chipProdId = simModelInput?.value.trim();
  const temperature = simTemperatureInput?.value.trim();
  const voltage = simVoltageInput?.value.trim();
  const size = simSizeInput?.value.trim();
  const capacity = simCapacityInput?.value.trim();
  const productionMode = simProductionSelect?.value || "";

  const parseNumber = (value) => {
    if (!value) {
      return null;
    }
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  };

  return {
    chip_prod_id: chipProdId || null,
    temperature: temperature || null,
    voltage: parseNumber(voltage),
    size: size || null,
    capacity: parseNumber(capacity),
    production_mode: productionMode || null,
  };
}

async function sendSimulationParams({ run } = { run: false }) {
  if (!simFormStatus) {
    return;
  }
  const params = collectSimFormParams();
  if (run) {
    setSimStatus("추천 실행 요청 전송 중...");
    await sendChatMessage("", {
      silentUser: true,
      payload: { intent: "simulation_run", params },
    });
    return;
  }
  setSimStatus("추천 입력 전송 중...");
  const payload = run
    ? { session_id: currentSessionId }
    : { session_id: currentSessionId, ...params };
  try {
    const response = await fetch(
      run ? "/api/simulation/run" : "/api/simulation/params",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );
    if (!response.ok) {
      setSimStatus("추천 요청 실패", true);
      return;
    }
    const data = await response.json();
    if (data.missing && data.missing.length) {
      renderSimulationForm({ params: data.params || params, missing: data.missing });
      return;
    }
    if (data.result) {
      if (!run) {
        renderSimResult({ params: data.params || params, result: data.result });
      }
      setSimStatus("추천 결과 수신 완료");
      updateMissingInputs([]);
      return;
    }
    if (!run) {
      setSimStatus("입력 반영 완료: 추천 실행을 눌러주세요.");
      updateMissingInputs([]);
      return;
    }
    setSimStatus("추천 실행 완료");
    updateMissingInputs([]);
  } catch (error) {
    setSimStatus("추천 요청 실패: 네트워크 오류", true);
  }
}

function collectRecommendationParams() {
  if (!simResultEl) {
    return {};
  }
  const params = {};
  simResultEl.querySelectorAll("input[data-param]").forEach((inputEl) => {
    const key = inputEl.dataset.param;
    if (!key) {
      return;
    }
    const raw = String(inputEl.value || "").trim();
    if (!raw) {
      return;
    }
    const numeric = Number(raw);
    params[key] = Number.isFinite(numeric) ? numeric : raw;
  });
  return params;
}

async function sendRecommendationParams() {
  if (!recommendationStatusEl) {
    return;
  }
  const params = collectRecommendationParams();
  if (!Object.keys(params).length) {
    setRecommendationStatus("파라미터를 먼저 입력해 주세요.", true);
    return;
  }
  setRecommendationStatus("파라미터 반영 중...");
  if (recommendationApplyButton) {
    recommendationApplyButton.disabled = true;
  }
  try {
    const response = await fetch("/api/recommendation/params", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: currentSessionId, params }),
    });
    if (!response.ok) {
      setRecommendationStatus("파라미터 반영 실패", true);
      return;
    }
    const data = await response.json();
    if (data.status === "missing") {
      setRecommendationStatus("추천 결과가 없습니다.", true);
      return;
    }
    if (data.result) {
      renderSimResult({ params: data.params || {}, result: data.result });
    }
    recommendationDirty = false;
    setRecommendationStatus("파라미터 반영 완료");
  } catch (error) {
    setRecommendationStatus("파라미터 반영 실패: 네트워크 오류", true);
  } finally {
    if (recommendationApplyButton) {
      recommendationApplyButton.disabled = false;
    }
  }
}

function renderLotResult(payload) {
  if (!payload || !payload.rows || payload.rows.length === 0) {
    if (lotResultEl) {
      lotResultEl.textContent = "조회 결과가 없습니다.";
    }
    return;
  }

  renderLatestLot(payload);
  renderMatchSummary();
  showOnlyStreamCards([matchSummaryCard, latestLotCard, lotCard]);

  const lotId = payload.lot_id || payload.query || "";
  setCurrentLot(lotId);
  const row = payload.rows[0] || {};
  const columnLabels = payload.column_labels || {};
  const kpiRow = document.createElement("div");
  kpiRow.className = "kpi-row";
  if (lotId) {
    kpiRow.appendChild(createKpiCard("LOT", lotId, "accent"));
  }
  const statusValue = getRowValue(row, ["status", "state", "lot_status"]);
  if (statusValue) {
    kpiRow.appendChild(createKpiCard("Status", statusValue));
  }
  const lineValue = getRowValue(row, ["line", "line_id", "line_no"]);
  if (lineValue) {
    kpiRow.appendChild(createKpiCard("Line", lineValue));
  }
  const timeValue = formatTimestamp(
    getRowValue(row, ["timestamp", "updated_at", "updatedAt"])
  );
  if (timeValue) {
    kpiRow.appendChild(createKpiCard("Updated", timeValue));
  }
  kpiRow.appendChild(createKpiCard("Rows", payload.rows.length));
  const list = document.createElement("dl");
  list.className = "lot-grid";

  Object.entries(row).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = columnLabels[key] || key;
    const dd = document.createElement("dd");
    dd.textContent = renderValue(value);
    list.appendChild(dt);
    list.appendChild(dd);
  });

  const header = document.createElement("div");
  header.className = "lot-header";
  const fetchedAt = new Date().toLocaleTimeString("ko-KR");
  header.innerHTML = `
    <strong>${lotId || "LOT 정보"}</strong>
    <span>${payload.source ? `source: ${payload.source}` : ""} · 조회 ${fetchedAt}</span>
  `;

  lotResultEl.innerHTML = "";
  lotResultEl.appendChild(header);
  lotResultEl.appendChild(kpiRow);
  lotResultEl.appendChild(list);

  if (payload.rows.length > 1) {
    const count = document.createElement("div");
    count.className = "lot-count";
    count.textContent = `추가 ${payload.rows.length - 1}건 있음`;
    lotResultEl.appendChild(count);
  }
}

function renderSimResult(payload) {
  if (!payload || !payload.result) {
    if (simResultEl) {
      simResultEl.textContent = "추천 결과가 없습니다.";
    }
    return;
  }

  if (!hasFinalBriefing) {
    showOnlyStreamCards([simResultCard]);
  }
  recommendationDirty = false;

  const params = payload.params || {};
  setLastSimulationParams(params);
  renderMatchSummary();
  const result = payload.result || {};
  const targetKeys = [
    "dc_time",
    "dc_freq",
    "bias_volt",
    "long_term_halt_volt",
    "long_term_halt_tmpt",
  ];
  const resultParams =
    result.params && typeof result.params === "object" ? result.params : null;
  const targetEntries = resultParams
    ? targetKeys
        .filter((key) => Object.prototype.hasOwnProperty.call(resultParams, key))
        .map((key) => [key, resultParams[key]])
    : [];
  const paramEntries = resultParams
    ? Object.entries(resultParams).filter(
        ([key]) => !targetKeys.includes(key)
      )
    : [];
  const kpiRow = document.createElement("div");
  kpiRow.className = "kpi-row";
  if (result.recommended_chip_prod_id) {
    kpiRow.appendChild(
      createKpiCard("추천 기종", result.recommended_chip_prod_id, "accent")
    );
  }
  if (result.representative_lot) {
    kpiRow.appendChild(createKpiCard("대표 LOT", result.representative_lot));
  }
  const paramCount = paramEntries.length;
  if (paramCount) {
    kpiRow.appendChild(createKpiCard("파라미터", `${paramCount}개`, "ghost"));
  }
  if (recommendationApplyButton) {
    recommendationApplyButton.disabled = !paramCount;
  }
  if (paramCount) {
    setRecommendationStatus("파라미터를 수정한 뒤 반영을 눌러주세요.");
  } else {
    setRecommendationStatus("추천 파라미터를 불러오지 못했습니다.", true);
  }

  simResultEl.innerHTML = "";
  if (kpiRow.childElementCount) {
    simResultEl.appendChild(kpiRow);
  }

  const productionLabel = formatProductionMode(params.production_mode);
  const inputSummary = document.createElement("div");
  inputSummary.className = "metric";
  inputSummary.innerHTML = `
    <span>입력값</span>
    <strong>
      T ${params.temperature ?? "-"}, V ${params.voltage ?? "-"}, S ${params.size ?? "-"}, C ${params.capacity ?? "-"}, M ${productionLabel}
    </strong>
  `;
  simResultEl.appendChild(inputSummary);

  if (targetEntries.length) {
    const sectionLabel = document.createElement("div");
    sectionLabel.className = "section-label";
    sectionLabel.textContent = "타겟 값";
    simResultEl.appendChild(sectionLabel);

    const targetGrid = document.createElement("div");
    targetGrid.className = "param-grid";
    targetEntries.forEach(([key, value]) => {
      const item = document.createElement("div");
      item.className = "param-item";
      const label = document.createElement("span");
      label.textContent = key;
      const input = document.createElement("input");
      input.type = "text";
      input.inputMode = "numeric";
      input.value = value ?? "";
      input.dataset.param = key;
      if (input.value === "") {
        input.classList.add("is-empty");
      }
      input.addEventListener("input", () => {
        recommendationDirty = true;
        setRecommendationStatus("파라미터 수정됨. 반영 버튼을 눌러주세요.");
        input.classList.toggle("is-empty", input.value.trim() === "");
      });
      item.appendChild(label);
      item.appendChild(input);
      targetGrid.appendChild(item);
    });
    simResultEl.appendChild(targetGrid);
  }

  if (paramEntries.length) {
    const sectionLabel = document.createElement("div");
    sectionLabel.className = "section-label";
    sectionLabel.textContent = "추천 파라미터";
    simResultEl.appendChild(sectionLabel);

    const paramGrid = document.createElement("div");
    paramGrid.className = "param-grid";
    paramEntries.forEach(([key, value]) => {
      const item = document.createElement("div");
      item.className = "param-item";
      const label = document.createElement("span");
      label.textContent = key;
      const input = document.createElement("input");
      input.type = "text";
      input.inputMode = "numeric";
      input.value = value ?? "";
      input.dataset.param = key;
      if (input.value === "") {
        input.classList.add("is-empty");
      }
      input.addEventListener("input", () => {
        recommendationDirty = true;
        setRecommendationStatus("파라미터 수정됨. 반영 버튼을 눌러주세요.");
        input.classList.toggle("is-empty", input.value.trim() === "");
      });
      item.appendChild(label);
      item.appendChild(input);
      paramGrid.appendChild(item);
    });
    simResultEl.appendChild(paramGrid);
  }

  const table = buildResultTable(result);
  if (table) {
    const sectionLabel = document.createElement("div");
    sectionLabel.className = "section-label";
    sectionLabel.textContent = "추천 결과 상세";
    simResultEl.appendChild(sectionLabel);
    simResultEl.appendChild(table);
  }
}

function renderDefectRateChartInto(targetEl, payload, options = {}) {
  if (!targetEl) {
    return;
  }
  const histogram = payload.histogram;
  const lots = Array.isArray(payload.lots) ? payload.lots : [];
  const config = payload.config || {};
  const metricLabel = payload.metric_label || "";
  const chartType = String(
    payload.chart_type || config.chart_type || (histogram ? "histogram" : "bar")
  ).toLowerCase();
  if (!histogram && !lots.length) {
    targetEl.textContent = metricLabel
      ? `${metricLabel} 그래프가 없습니다.`
      : "그래프가 없습니다.";
    return;
  }

  const stats = payload.stats || {};
  const valueUnit =
    payload.value_unit ||
    stats.value_unit ||
    (histogram && histogram.value_unit) ||
    config.value_unit ||
    "";
  const tableRows = Array.isArray(payload.table_rows) ? payload.table_rows : [];
  const tableColumns = Array.isArray(payload.table_columns)
    ? payload.table_columns
    : [];
  targetEl.innerHTML = "";
  if (stats && stats.count && options.includeMeta !== false) {
    const meta = document.createElement("div");
    meta.className = "candidate-meta";
    const formatStat = (value) => {
      if (value === null || value === undefined) {
        return "--";
      }
      const num = Number(value);
      if (!Number.isFinite(num)) {
        return "--";
      }
      if (valueUnit === "ratio") {
        return `${(num * 100).toFixed(2)}%`;
      }
      if (valueUnit === "percent") {
        return `${num.toFixed(2)}%`;
      }
      if (valueUnit) {
        return `${num.toFixed(3)} ${valueUnit}`;
      }
      return num.toFixed(3);
    };
    const avg = formatStat(stats.avg);
    const min = formatStat(stats.min);
    const max = formatStat(stats.max);
    const prefix =
      options.metaPrefix ||
      (metricLabel ? `${metricLabel}` : "필터");
    meta.textContent = `${prefix} ${stats.count}건 · 평균 ${avg} · 최소 ${min} · 최대 ${max}`;
    targetEl.appendChild(meta);
  }

  if (chartType === "table" || tableRows.length) {
    const table = buildResultTable({
      rows: tableRows,
      columns: tableColumns,
      column_labels: payload.column_labels || payload.table_column_labels,
    });
    if (table) {
      const wrapper = document.createElement("div");
      wrapper.className = "table-scroll";
      wrapper.appendChild(table);
      targetEl.appendChild(wrapper);
    } else {
      targetEl.textContent = metricLabel
        ? `${metricLabel} 표가 없습니다.`
        : "표가 없습니다.";
    }
    return;
  }

  if (
    chartType === "histogram" &&
    histogram &&
    Array.isArray(histogram.bins) &&
    histogram.bins.length
  ) {
    const bins = histogram.bins;
    const maxCount = Math.max(
      ...bins.map((bin) => Number(bin.count) || 0),
      1
    );
    const histogramUnit = histogram.value_unit || valueUnit;
    const unit =
      histogramUnit === "percent" || histogramUnit === "ratio"
        ? "%"
        : histogramUnit
        ? ` ${histogramUnit}`
        : "";
    const normalize = histogram.normalize || "count";
    const formatRange = (value) => {
      if (value === null || value === undefined) {
        return "-";
      }
      const num = Number(value);
      if (!Number.isFinite(num)) {
        return "-";
      }
      if (histogramUnit === "ratio") {
        return (num * 100).toFixed(1);
      }
      if (histogramUnit === "percent") {
        return num.toFixed(1);
      }
      return unit ? num.toFixed(2) : num.toFixed(3);
    };

    const chart = document.createElement("div");
    chart.className = "defect-histogram";
    bins.forEach((bin) => {
      const count = Number(bin.count) || 0;
      const value = Number(bin.value);
      const displayValue =
        normalize === "count" || Number.isNaN(value)
          ? String(count)
          : normalize === "percent"
          ? `${value.toFixed(1)}%`
          : value.toFixed(3);

      const item = document.createElement("div");
      item.className = "defect-bin";
      const bar = document.createElement("div");
      bar.className = "defect-bin-bar";
      bar.style.height = `${(count / maxCount) * 100}%`;
      bar.title = `${formatRange(bin.start)}-${formatRange(bin.end)}${unit} · ${displayValue}`;
      const label = document.createElement("div");
      label.className = "defect-bin-label";
      label.textContent = `${formatRange(bin.start)}-${formatRange(bin.end)}${unit}`;
      const valueEl = document.createElement("div");
      valueEl.className = "defect-bin-value";
      valueEl.textContent = displayValue;
      item.appendChild(bar);
      item.appendChild(label);
      item.appendChild(valueEl);
      chart.appendChild(item);
    });
    targetEl.appendChild(chart);
  return;
  }

  if (["line", "scatter", "area"].includes(chartType)) {
    const series = Array.isArray(payload.series)
      ? payload.series
      : lots
          .map((item, index) => ({
            x: index + 1,
            y: Number(item.defect_rate),
            lot_id: item.lot_id,
          }))
          .filter((point) => Number.isFinite(point.y));

    if (!series.length) {
      defectChartEl.textContent = "No chart points available.";
      return;
    }

    const values = series.map((point) => point.y);
    const minY = Math.min(...values);
    const maxY = Math.max(...values);
    const spread = maxY - minY || 1;
    const width = 520;
    const height = 170;
    const padding = 18;
    const step = series.length > 1 ? (width - padding * 2) / (series.length - 1) : 0;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("class", "defect-line-chart");

    const points = series.map((point, index) => {
      const x = padding + step * index;
      const y =
        height - padding - ((point.y - minY) / spread) * (height - padding * 2);
      return { x, y, raw: point };
    });

    if (chartType === "area") {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const d =
        points
          .map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`)
          .join(" ") +
        ` L${points[points.length - 1].x},${height - padding}` +
        ` L${points[0].x},${height - padding} Z`;
      path.setAttribute("d", d);
      path.setAttribute("fill", "rgba(66, 99, 235, 0.18)");
      path.setAttribute("stroke", "rgba(66, 99, 235, 0.6)");
      path.setAttribute("stroke-width", "2");
      svg.appendChild(path);
    } else if (chartType === "line") {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const d = points
        .map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`)
        .join(" ");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "rgba(66, 99, 235, 0.7)");
      path.setAttribute("stroke-width", "2");
      svg.appendChild(path);
    }

    points.forEach((point) => {
      const circle = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle"
      );
      circle.setAttribute("cx", point.x);
      circle.setAttribute("cy", point.y);
      circle.setAttribute("r", chartType === "scatter" ? "3.5" : "2.5");
      circle.setAttribute("fill", "rgba(66, 99, 235, 0.85)");
      svg.appendChild(circle);
    });

    targetEl.appendChild(svg);
    return;
  }

  const barOrientation = String(
    payload.bar_orientation || config.bar_orientation || "horizontal"
  ).toLowerCase();
  if (chartType === "bar" && (barOrientation === "vertical" || barOrientation === "column")) {
    const maxRate = Math.max(
      ...lots.map((item) => Number(item.defect_rate) || 0),
      0.0001
    );
    const formatBarValue = (value) => {
      const num = Number(value);
      if (!Number.isFinite(num)) {
        return "-";
      }
      if (valueUnit === "ratio") {
        return `${(num * 100).toFixed(2)}%`;
      }
      if (valueUnit === "percent") {
        return `${num.toFixed(2)}%`;
      }
      if (valueUnit) {
        return `${num.toFixed(3)} ${valueUnit}`;
      }
      return num.toFixed(3);
    };
    const chart = document.createElement("div");
    chart.className = "defect-column-chart";
    lots.forEach((item) => {
      const rate = Number(item.defect_rate) || 0;
      const column = document.createElement("div");
      column.className = "defect-column";
      const value = document.createElement("div");
      value.className = "defect-column-value";
      value.textContent = formatBarValue(rate);
      const bar = document.createElement("div");
      bar.className = "defect-column-bar";
      bar.style.height = `${Math.min(100, (rate / maxRate) * 100)}%`;
      const label = document.createElement("div");
      label.className = "defect-column-label";
      label.textContent = item.label || item.lot_id || "-";
      column.appendChild(value);
      column.appendChild(bar);
      column.appendChild(label);
      chart.appendChild(column);
    });
    targetEl.appendChild(chart);
    return;
  }

  const maxRate = Math.max(
    ...lots.map((item) => Number(item.defect_rate) || 0),
    0.0001
  );
  const topIndices = new Set(
    lots
      .map((item, index) => ({
        index,
        rate: Number(item.defect_rate),
      }))
      .filter((item) => Number.isFinite(item.rate))
      .sort((a, b) => b.rate - a.rate)
      .slice(0, 3)
      .map((item) => item.index)
  );
  const chart = document.createElement("div");
  chart.className = "defect-chart";
  lots.forEach((item, index) => {
    const rate = Number(item.defect_rate) || 0;
    const percent = (rate * 100).toFixed(2);
    const row = document.createElement("div");
    row.className = "defect-row";
    if (topIndices.has(index)) {
      row.classList.add("highlight");
    }
    const label = document.createElement("div");
    label.className = "defect-label";
    label.textContent = item.label || item.lot_id || "-";
    const bar = document.createElement("div");
    bar.className = "defect-bar";
    const fill = document.createElement("div");
    fill.className = "defect-bar-fill";
    fill.style.width = `${Math.min(100, (rate / maxRate) * 100)}%`;
    bar.appendChild(fill);
    const value = document.createElement("div");
    value.className = "defect-value";
    value.textContent = `${percent}%`;
    row.appendChild(label);
    row.appendChild(bar);
    row.appendChild(value);
    chart.appendChild(row);
  });
  targetEl.appendChild(chart);
}

function renderDefectRateChart(payload = {}) {
  if (!defectChartEl) {
    return;
  }
  lastDefectChartPayload = payload;
  showStreamCard(defectChartCard);
  if (defectChartTitleEl) {
    const label = payload.metric_label || "불량률 분포";
    defectChartTitleEl.textContent = payload.metric_label
      ? `${label} 그래프`
      : label;
  }
  renderDefectRateChartInto(defectChartEl, payload);
}

function renderFinalDefectChart(payload = {}) {
  if (!finalDefectChartEl) {
    return;
  }
  finalDefectChartEl.style.display = "";
  lastFinalDefectChartPayload = payload;
  const charts = Array.isArray(payload.charts) ? payload.charts : null;
  finalDefectChartEl.innerHTML = "";
  if (charts && charts.length) {
    const stack = document.createElement("div");
    stack.className = "defect-chart-stack";
    charts.forEach((chart, index) => {
      const panel = document.createElement("div");
      panel.className = "defect-chart-block";

      const title = document.createElement("div");
      title.className = "candidate-meta";
      const label =
        chart.title || `TOP${chart.rank || index + 1} 설계안`;
      const lotCount = Number(chart.lot_count);
      title.textContent = Number.isFinite(lotCount) && lotCount >= 0
        ? `${label} · LOT ${lotCount}개`
        : label;
      panel.appendChild(title);

      const chartEl = document.createElement("div");
      chartEl.className = "defect-chart";
      renderDefectRateChartInto(chartEl, chart, {
        includeMeta: true,
      });
      panel.appendChild(chartEl);
      stack.appendChild(panel);
    });
    finalDefectChartEl.appendChild(stack);
    return;
  }
  renderDefectRateChartInto(finalDefectChartEl, payload, {
    includeMeta: false,
  });
}

function renderDesignCandidates(payload = {}) {
  if (!designCandidatesEl) {
    return;
  }
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  const designLabels = payload.design_labels || payload.column_labels || {};
  if (!candidates.length) {
    designCandidatesEl.textContent = "설계 후보가 없습니다.";
    return;
  }
  showStreamCard(designCandidatesCard);
  designCandidatesEl.innerHTML = "";

  const offset = payload.offset || 0;
  const total = payload.total || candidates.length;
  const meta = document.createElement("div");
  meta.className = "candidate-meta";
  meta.textContent = `전체 ${total}건 중 ${offset + 1}~${
    offset + candidates.length
  }건 표시`;
  designCandidatesEl.appendChild(meta);

  const table = document.createElement("table");
  table.className = "result-table";
  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th>순번</th>
      <th>LOT</th>
      <th>예측 타겟</th>
      <th>불량률</th>
      <th>설계값</th>
    </tr>
  `;
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  candidates.forEach((item, index) => {
    const row = document.createElement("tr");
    const rank = item.rank || offset + index + 1;
    const defect = item.defect_rate
      ? `${(item.defect_rate * 100).toFixed(2)}%`
      : "-";
    const target = item.predicted_target ?? "-";
    const design =
      item.design && typeof item.design === "object"
        ? Object.entries(item.design)
            .map(([key, value]) => {
              const label = designLabels[key] || key;
              return `${label}=${renderValue(value)}`;
            })
            .join(", ")
        : "-";
    row.innerHTML = `
      <td>${rank}</td>
      <td>${item.lot_id || "-"}</td>
      <td>${target}</td>
      <td>${defect}</td>
      <td>${design}</td>
    `;
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  designCandidatesEl.appendChild(table);
}

function renderFinalBriefing(payload = {}) {
  if (!finalBriefingEl) {
    return;
  }
  hasFinalBriefing = true;
  showOnlyStreamCards([finalBriefingCard]);
  finalBriefingEl.innerHTML = "";
  if (finalDefectChartEl) {
    finalDefectChartEl.textContent = "";
    finalDefectChartEl.style.display = "none";
  }

  const chipProdId = payload.chip_prod_id || "-";
  const referenceLot = payload.reference_lot || "-";
  const kpiRow = document.createElement("div");
  kpiRow.className = "kpi-row";
  kpiRow.appendChild(createKpiCard("추천 기종", chipProdId, "accent"));
  kpiRow.appendChild(createKpiCard("레퍼런스 LOT", referenceLot));
  finalBriefingEl.appendChild(kpiRow);

  const blocks = Array.isArray(payload.design_blocks)
    ? payload.design_blocks
    : [];
  if (!blocks.length) {
    const empty = document.createElement("div");
    empty.className = "candidate-meta";
    empty.textContent = "설계 후보가 아직 없습니다.";
    finalBriefingEl.appendChild(empty);
    return;
  }

  const formatMetric = (value, unit) => {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return "-";
    }
    if (unit === "ratio") {
      return `${(num * 100).toFixed(2)}%`;
    }
    if (unit === "percent") {
      return `${num.toFixed(2)}%`;
    }
    if (unit) {
      return `${num.toFixed(3)} ${unit}`;
    }
    return num.toFixed(3);
  };

  const container = document.createElement("div");
  container.className = "design-blocks";
  const wrapTable = (tableEl) => {
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    wrapper.appendChild(tableEl);
    return wrapper;
  };
  const buildDesignTable = (items) => {
    const table = document.createElement("table");
    table.className = "result-table design-table";
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    items.forEach((item) => {
      const th = document.createElement("th");
      th.textContent = item.label || item.key || "-";
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    const valueRow = document.createElement("tr");
    items.forEach((item) => {
      const td = document.createElement("td");
      td.textContent = renderValue(item.value);
      valueRow.appendChild(td);
    });
    tbody.appendChild(valueRow);
    table.appendChild(tbody);
    return table;
  };
  blocks.slice(0, 3).forEach((block, index) => {
    const blockEl = document.createElement("div");
    blockEl.className = "design-block";

    const rank = block.rank || index + 1;
    const predicted = block.predicted_target;
    const lotCount = Number(block.lot_count);
    const header = document.createElement("div");
    header.className = "candidate-meta";
    let headerText = `TOP${rank} 설계안`;
    if (predicted !== undefined && predicted !== null && predicted !== "") {
      headerText += ` · 예측지표 ${renderValue(predicted)}`;
    }
    if (Number.isFinite(lotCount) && lotCount >= 0) {
      headerText += ` · LOT ${lotCount}개`;
    }
    header.textContent = headerText;
    blockEl.appendChild(header);

    const designDisplay = Array.isArray(block.design_display)
      ? block.design_display
      : [];
    if (designDisplay.length) {
      const table = buildDesignTable(designDisplay);
      blockEl.appendChild(wrapTable(table));
    } else {
      const emptyDesign = document.createElement("div");
      emptyDesign.className = "candidate-meta";
      emptyDesign.textContent = "설계 값이 없습니다.";
      blockEl.appendChild(emptyDesign);
    }

    const metrics = Array.isArray(block.metrics) ? block.metrics : [];
    if (metrics.length) {
      const pills = document.createElement("div");
      pills.className = "summary-pills";
      metrics.forEach((metric) => {
        const pill = document.createElement("span");
        pill.className = "pill summary-pill";
        pill.textContent = `${metric.label || metric.key} ${formatMetric(metric.value, metric.unit)}`;
        pills.appendChild(pill);
      });
      blockEl.appendChild(pills);
    }

    if (block.chart && typeof block.chart === "object") {
      const chartEl = document.createElement("div");
      chartEl.className = "defect-chart";
      renderDefectRateChartInto(chartEl, block.chart, {
        includeMeta: true,
      });
      blockEl.appendChild(chartEl);
    } else {
      const emptyChart = document.createElement("div");
      emptyChart.className = "candidate-meta";
      emptyChart.textContent = "불량률 데이터가 없습니다.";
      blockEl.appendChild(emptyChart);
    }

    const matchRows = Array.isArray(block.match_rows) ? block.match_rows : [];
    const matchColumns = Array.isArray(block.match_columns)
      ? block.match_columns
      : [];
    const matchCount = Number(block.match_row_count);
    if (matchRows.length) {
      const matchMeta = document.createElement("div");
      matchMeta.className = "candidate-meta";
      let matchText = Number.isFinite(matchCount)
        ? `매칭 LOT ${matchCount}개`
        : `매칭 LOT ${matchRows.length}개`;
      if (
        Number.isFinite(matchCount) &&
        matchCount > matchRows.length
      ) {
        matchText += ` (상위 ${matchRows.length}개 표시)`;
      }
      if (block.match_recent_months) {
        matchText += ` · 최근 ${block.match_recent_months}개월`;
      }
      matchMeta.textContent = matchText;
      blockEl.appendChild(matchMeta);
      const matchTable = buildResultTable({
        rows: matchRows,
        columns: matchColumns,
        column_labels: block.match_column_labels || payload.column_labels,
      });
      if (matchTable) {
        blockEl.appendChild(wrapTable(matchTable));
      }
    } else {
      const emptyMatch = document.createElement("div");
      emptyMatch.className = "candidate-meta";
      emptyMatch.textContent = "매칭 LOT가 없습니다.";
      blockEl.appendChild(emptyMatch);
    }

    container.appendChild(blockEl);
  });
  finalBriefingEl.appendChild(container);
}

function buildResultTable(result) {
  if (!result || typeof result !== "object") {
    return null;
  }
  const rows = Array.isArray(result.rows) ? result.rows : null;
  if (!rows || rows.length === 0) {
    return null;
  }
  let columns = Array.isArray(result.columns) ? result.columns : [];
  const firstRow = rows[0];
  if (!columns.length) {
    if (firstRow && typeof firstRow === "object" && !Array.isArray(firstRow)) {
      columns = Object.keys(firstRow);
    } else if (Array.isArray(firstRow)) {
      columns = firstRow.map((_, index) => `col${index + 1}`);
    }
  }
  if (!columns.length) {
    return null;
  }

  const columnLabels = result.column_labels || result.columnLabels || {};
  const table = document.createElement("table");
  table.className = "result-table";
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = columnLabels[column] || column;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column, index) => {
      const td = document.createElement("td");
      let value = "-";
      if (Array.isArray(row)) {
        value = row[index];
      } else if (row && typeof row === "object") {
        value = row[column];
      }
      td.textContent = renderValue(value);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

function renderPredictionResult(payload) {
  if (!payload || !payload.result) {
    if (predictionResultEl) {
      predictionResultEl.textContent = "예측 결과가 없습니다.";
    }
    return;
  }

  showOnlyStreamCards([predictionCard]);

  const result = payload.result || {};
  const recommendation = payload.recommendation || {};
  const chipProdId = recommendation.recommended_chip_prod_id || "-";
  const repLot = recommendation.representative_lot || "-";

  predictionResultEl.innerHTML = "";

  const kpiRow = document.createElement("div");
  kpiRow.className = "kpi-row";
  kpiRow.appendChild(createKpiCard("추천 기종", chipProdId, "accent"));
  kpiRow.appendChild(createKpiCard("대표 LOT", repLot));
  predictionResultEl.appendChild(kpiRow);

  const grid = document.createElement("div");
  grid.className = "metric-grid";
  grid.innerHTML = `
    <div class="metric">
      <span>예측 용량</span>
      <strong>${renderValue(result.predicted_capacity)}</strong>
    </div>
    <div class="metric">
      <span>예측 DC용량</span>
      <strong>${renderValue(result.predicted_dc_capacity)}</strong>
    </div>
    <div class="metric">
      <span>신뢰성 통과확률</span>
      <strong>${result.reliability_pass_prob ?? "-"}</strong>
    </div>
  `;
  predictionResultEl.appendChild(grid);
}

function handleEvent(event) {
  if (!event || !event.type) {
    return;
  }

  if (event.type === "chat_stream_start") {
    const messageId = event.payload?.message_id || "";
    getOrCreateStreamingMessage(messageId);
    return;
  }

  if (event.type === "chat_stream_block_start") {
    const messageId = event.payload?.message_id || "";
    const blockId = event.payload?.block_id || "";
    startStreamingBlock(messageId, blockId);
    return;
  }

  if (event.type === "chat_stream_delta") {
    const messageId = event.payload?.message_id || "";
    const blockId = event.payload?.block_id || "";
    const delta = event.payload?.delta || "";
    appendStreamingDelta(messageId, blockId, delta);
    return;
  }

  if (event.type === "chat_stream_block_end") {
    return;
  }

  if (event.type === "chat_stream_end") {
    const messageId = event.payload?.message_id || "";
    finalizeStreamingMessage(messageId);
    return;
  }

  if (event.type === "briefing_table") {
    const messageId = event.payload?.message_id || "";
    const markdown = event.payload?.markdown || "";
    const animate = event.payload?.animate !== false;
    appendStreamingTable(messageId, markdown, animate);
    return;
  }

  if (event.type === "chat_message") {
    const role = event.payload?.role || "assistant";
    const content = event.payload?.content || "";
    if (content) {
      addMessage(role, content);
    }
  }

  if (event.type === "lot_result" || event.type === "db_result") {
    renderLotResult(event.payload);
    addEventLog("LOT", `조회 ${event.payload.rows.length}`);
    if (event.payload.lot_id) {
      setLotStatus(`${event.payload.lot_id} 조회 완료`);
    }
  }

  if (event.type === "simulation_form") {
    renderSimulationForm(event.payload);
    addEventLog("추천", "입력 패널 업데이트");
  }

  if (event.type === "simulation_result") {
    const chipProdId = event.payload?.result?.recommended_chip_prod_id || "결과 수신";
    addEventLog("추천", `기종: ${chipProdId}`);
    return;
  }

  if (event.type === "defect_rate_chart") {
    const label = event.payload?.metric_label || "불량률";
    addEventLog(label, "그래프 업데이트");
    return;
  }
  if (event.type === "final_defect_chart") {
    if (hasFinalBriefing) {
      return;
    }
    renderFinalDefectChart(event.payload);
    const label = event.payload?.metric_label || "불량률";
    addEventLog(label, "최종 그래프 업데이트");
  }

  if (event.type === "design_candidates") {
    addEventLog("설계", "후보 업데이트");
    return;
  }

  if (event.type === "final_briefing") {
    renderFinalBriefing(event.payload);
    renderMatchSummary(event.payload);
    addEventLog("브리핑", "최종 브리핑 업데이트");
  }

  if (event.type === "stage_focus") {
    const stage = event.payload?.stage || "";
    focusStage(stage);
  }

  if (event.type === "prediction_result") {
    const prob = event.payload?.result?.reliability_pass_prob;
    const detail = prob ? `통과확률: ${prob}` : "결과 수신";
    addEventLog("예측", detail);
    return;
  }

  if (event.type === "pipeline_status") {
    const stage = event.payload?.stage || "";
    const message = event.payload?.message || "";
    const done = Boolean(event.payload?.done);
    if (message) {
      appendPipelineStatus(stage, message, done);
    }
  }

  if (event.type === "pipeline_stage_tables") {
    const stage = event.payload?.stage || "";
    const tables = Array.isArray(event.payload?.tables) ? event.payload.tables : [];
    const notes = Array.isArray(event.payload?.notes) ? event.payload.notes : [];
    if (stage) {
      updatePipelineStageTables(stage, { tables, notes });
    }
  }

  if (event.type === "workflow_log") {
    const label = event.payload?.label || "LOG";
    const detail = event.payload?.detail || "";
    const meta = event.payload?.meta;
    let metaText = "";
    if (meta && typeof meta === "object") {
      metaText = Object.entries(meta)
        .map(([key, value]) => `${key}=${value}`)
        .join(", ");
    }
    const combined = metaText ? `${detail} | ${metaText}` : detail;
    addEventLog(label, combined);
    updateEventEmpty();
  }

  if (event.type === "frontend_trigger") {
    const message = event.payload?.message || "프론트 트리거가 도착했습니다.";
    addEventLog("UI", message);
    const isWarning = /필요|부족|없습니다|실패/.test(message);
    if (message.toLowerCase().includes("lot")) {
      setLotStatus(message, isWarning);
    }
    if (message.includes("추천") || message.includes("시뮬레이션")) {
      setSimStatus(message, isWarning);
    }
  }
}

function shouldApplyUiEvent() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return true;
  }
  return activeSocketSessionId !== currentSessionId;
}

function connectWebSocket(sessionId) {
  if (!sessionId) {
    return;
  }
  if (socket) {
    suppressSocketCloseNotice = true;
    socket.close();
  }
  activeSocketSessionId = sessionId;
  const wsUrl = `${wsProtocol}://${window.location.host}/ws?session_id=${encodeURIComponent(sessionId)}`;
  socket = new WebSocket(wsUrl);
  statusLabel.textContent = "연결 중";

  socket.addEventListener("open", () => {
    suppressSocketCloseNotice = false;
    statusDot.classList.add("live");
    statusLabel.textContent = "연결 중";
    addMessage("system", "WebSocket 연결 완료. 이벤트 스트림이 활성화되었습니다.");
  });

  socket.addEventListener("close", () => {
    statusDot.classList.remove("live");
    statusLabel.textContent = "연결 끊김";
    activeSocketSessionId = "";
    if (!suppressSocketCloseNotice) {
      addMessage("system", "WebSocket 연결이 종료되었습니다.");
    }
    suppressSocketCloseNotice = false;
  });

  socket.addEventListener("message", (event) => {
    try {
      const data = JSON.parse(event.data);
      handleEvent(data);
    } catch (error) {
      addEventLog("WS", "JSON이 아닌 메시지가 수신되었습니다.");
    }
  });

  if (socketPingInterval) {
    clearInterval(socketPingInterval);
  }
  socketPingInterval = setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send("ping");
    }
  }, 20000);
}

async function sendChatMessage(message, options = {}) {
  const { silentUser = false, payload = null } = options || {};
  if (!silentUser) {
    addMessage("user", message);
  }
  input.value = "";
  input.style.height = "auto";
  const startedAt = performance.now();
  setChatStatus("응답 생성 중...", "chat");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: currentSessionId,
        message,
        ...(payload || {}),
      }),
    });

    if (!response.ok) {
      addMessage("assistant", "서버 오류입니다. 잠시 후 다시 시도해 주세요.");
      return;
    }

    const data = await response.json();
    if (data.ui_event && shouldApplyUiEvent()) {
      handleEvent(data.ui_event);
    }
    const hasLiveSocket =
      socket &&
      socket.readyState === WebSocket.OPEN &&
      activeSocketSessionId === currentSessionId;
    if (!data.streamed || !hasLiveSocket) {
      addMessage("assistant", data.assistant_message || "(응답 없음)");
    }
  } catch (error) {
    addMessage("assistant", "네트워크 오류입니다. 백엔드 상태를 확인해 주세요.");
  } finally {
    const elapsed = Math.round(performance.now() - startedAt);
    addEventLog("CLIENT", `chat ${elapsed}ms`);
    clearChatStatus("chat");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }
  await sendChatMessage(message);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});

input.addEventListener("compositionstart", () => {
  isComposing = true;
});

input.addEventListener("compositionend", () => {
  isComposing = false;
});

input.addEventListener("keydown", (event) => {
  if (isComposing || event.isComposing || event.keyCode === 229) {
    return;
  }
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (form.requestSubmit) {
      form.requestSubmit();
    } else {
      form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    }
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    const prompt = button.getAttribute("data-prompt");
    input.value = prompt;
    input.dispatchEvent(new Event("input"));
    form.dispatchEvent(new Event("submit"));
  });
});

function resolveLotId({ allowLast = false } = {}) {
  const raw = lotIdInput?.value.trim() || "";
  if (raw) {
    setCurrentLot(raw);
    return raw;
  }
  if (allowLast && lastLotId) {
    return lastLotId;
  }
  return "";
}

function handleLotAction(action) {
  const allowLast = action === "simulate";
  const lotId = resolveLotId({ allowLast });
  if (!lotId) {
    setLotStatus("LOT ID를 입력해 주세요.", true);
    return;
  }
  setLotStatus(`${lotId} 요청 전송 중...`);

  if (action === "search" || action === "search_sim") {
    if (lotResultEl) {
      showOnlyStreamCards([lotCard]);
      lotResultEl.textContent = `${lotId} 조회 중...`;
    }
  }

  if (action === "search") {
    sendChatMessage(`${lotId} 정보 보여줘.`);
  } else if (action === "simulate") {
    sendChatMessage(`${lotId} 추천 시뮬레이션 해줘.`);
  } else if (action === "search_sim") {
    sendChatMessage(`${lotId} 정보 조회하고 추천까지 진행해줘.`);
  }
}

if (lotSearchButton) {
  lotSearchButton.addEventListener("click", () => handleLotAction("search"));
}
if (lotSimButton) {
  lotSimButton.addEventListener("click", () => handleLotAction("simulate"));
}
if (lotSearchSimButton) {
  lotSearchSimButton.addEventListener("click", () => handleLotAction("search_sim"));
}
if (lotClearButton) {
  lotClearButton.addEventListener("click", () => {
    if (lotIdInput) {
      lotIdInput.value = "";
    }
    setLotStatus("LOT ID를 입력해 주세요.");
  });
}
if (lotIdInput) {
  lotIdInput.addEventListener("input", () => {
    setLotStatus("LOT ID를 입력해 주세요.");
  });
}

if (simApplyButton) {
  simApplyButton.addEventListener("click", () => {
    sendSimulationParams({ run: false });
  });
}
if (simRunButton) {
  simRunButton.addEventListener("click", () => {
    sendSimulationParams({ run: true });
  });
}
if (recommendationApplyButton) {
  recommendationApplyButton.addEventListener("click", () => {
    sendRecommendationParams();
  });
}
[
  simTemperatureInput,
  simVoltageInput,
  simSizeInput,
  simCapacityInput,
  simProductionSelect,
].forEach((inputEl) => {
  if (!inputEl) {
    return;
  }
  inputEl.addEventListener("input", () => {
    inputEl.classList.remove("missing");
    updateFilledInputs();
    setSimStatus("입력값을 수정 중입니다.");
  });
});

const testButton = document.getElementById("test-trigger");
if (testButton) {
  testButton.addEventListener("click", async () => {
    addMessage("system", "간이 테스트를 실행합니다. (API 키 불필요)");
    try {
      const response = await fetch("/api/test/trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: currentSessionId }),
      });

      if (!response.ok) {
        addMessage("assistant", "테스트 호출에 실패했습니다.");
        return;
      }

      addMessage("assistant", "테스트 호출 완료. 이벤트 패널을 확인해 주세요.");
    } catch (error) {
      addMessage("assistant", "테스트 요청 중 오류가 발생했습니다.");
    }
  });
}

setLotStatus("LOT ID를 입력해 주세요.");
setSimStatus("추천 입력 대기");
if (recommendationApplyButton) {
  recommendationApplyButton.disabled = true;
}
updateEventEmpty();

addMessage(
  "assistant",
  "LOT 조회 또는 인접기종 추천을 요청해 주세요. 추천 완료 후 예측 시뮬레이션도 이어서 진행할 수 있습니다."
);

async function loadWorkflowMeta() {
  if (!workflowNameEl || !workflowUpdatedEl) {
    return;
  }
  try {
    const response = await fetch("/api/workflow");
    if (!response.ok) {
      workflowNameEl.textContent = "워크플로우 없음";
      return;
    }
    const data = await response.json();
    workflowNameEl.textContent = data.meta?.name || "워크플로우";
    workflowUpdatedEl.textContent = data.meta?.updated_at
      ? `업데이트: ${data.meta.updated_at}`
      : "--";
  } catch (error) {
    workflowNameEl.textContent = "로드 실패";
  }
}

function renderWorkflowList({ workflows = [], activeId = "" } = {}) {
  if (!workflowListEl) {
    return;
  }
  workflowListEl.innerHTML = "";
  if (!workflows.length) {
    const emptyItem = document.createElement("li");
    emptyItem.className = "workflow-empty";
    emptyItem.textContent = "저장된 워크플로우가 없습니다.";
    workflowListEl.appendChild(emptyItem);
    return;
  }

  workflows.forEach((workflow) => {
    const item = document.createElement("li");
    item.className = "workflow-item";
    const info = document.createElement("div");
    info.className = "workflow-info";

    const selectButton = document.createElement("button");
    selectButton.type = "button";
    selectButton.className = "workflow-select";
    selectButton.textContent = workflow.name || "워크플로우";
    if (workflow.id && workflow.id === activeId) {
      selectButton.classList.add("active");
    }
    selectButton.addEventListener("click", () => {
      if (!workflow.id) {
        return;
      }
      applySavedWorkflow(workflow.id);
    });

    const date = document.createElement("div");
    date.className = "workflow-date";
    date.textContent = workflow.updated_at
      ? `업데이트: ${workflow.updated_at}`
      : "업데이트: -";

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "workflow-delete";
    deleteButton.title = "삭제";
    deleteButton.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16" /><path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" /><path d="M7 7l1 12a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2l1-12" /><path d="M10 11v6" /><path d="M14 11v6" /></svg>';
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      if (!workflow.id) {
        return;
      }
      const confirmed = window.confirm(
        `"${workflow.name || "워크플로우"}"를 삭제할까요?`
      );
      if (!confirmed) {
        return;
      }
      deleteSavedWorkflow(workflow.id);
    });

    info.appendChild(selectButton);
    info.appendChild(date);
    item.appendChild(info);
    item.appendChild(deleteButton);
    workflowListEl.appendChild(item);
  });
}

async function loadWorkflowCatalog() {
  if (!workflowListEl) {
    return;
  }
  try {
    const response = await fetch("/api/workflows");
    if (!response.ok) {
      renderWorkflowList({ workflows: [] });
      return;
    }
    const data = await response.json();
    renderWorkflowList({
      workflows: data.workflows || [],
      activeId: data.active_id || "",
    });
  } catch (error) {
    renderWorkflowList({ workflows: [] });
  }
}

async function applySavedWorkflow(workflowId) {
  if (!workflowId) {
    return;
  }
  try {
    const response = await fetch(`/api/workflows/${workflowId}/apply`, {
      method: "POST",
    });
    if (!response.ok) {
      addMessage("system", "워크플로우 적용에 실패했습니다.");
      return;
    }
    await loadWorkflowMeta();
    await loadWorkflowCatalog();
    addMessage("system", "워크플로우가 적용되었습니다.");
  } catch (error) {
    addMessage("system", "워크플로우 적용 중 오류가 발생했습니다.");
  }
}

async function deleteSavedWorkflow(workflowId) {
  if (!workflowId) {
    return;
  }
  try {
    const response = await fetch(`/api/workflows/${workflowId}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      addMessage("system", "워크플로우 삭제에 실패했습니다.");
      return;
    }
    await loadWorkflowCatalog();
    addMessage("system", "워크플로우가 삭제되었습니다.");
  } catch (error) {
    addMessage("system", "워크플로우 삭제 중 오류가 발생했습니다.");
  }
}

async function toggleWorkflowJson() {
  if (!workflowJsonEl || !loadWorkflowJsonButton) {
    return;
  }
  const isOpen = !workflowJsonEl.classList.contains("hidden");
  if (isOpen) {
    workflowJsonEl.classList.add("hidden");
    loadWorkflowJsonButton.textContent = "워크플로우 JSON 불러오기";
    return;
  }
  workflowJsonEl.classList.remove("hidden");
  workflowJsonEl.textContent = "로드 중...";
  loadWorkflowJsonButton.textContent = "JSON 닫기";
  try {
    const response = await fetch("/api/workflow");
    if (!response.ok) {
      workflowJsonEl.textContent = "로드 실패: 워크플로우 없음";
      return;
    }
    const data = await response.json();
    workflowJsonEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    workflowJsonEl.textContent = "로드 실패";
  }
}

loadWorkflowMeta();
loadWorkflowCatalog();

if (loadWorkflowJsonButton) {
  loadWorkflowJsonButton.addEventListener("click", toggleWorkflowJson);
}
if (toggleLogButton) {
  toggleLogButton.addEventListener("click", toggleEventLog);
}
if (eventPanelShowButton) {
  eventPanelShowButton.addEventListener("click", () => {
    setEventPanelOpen(true);
  });
}
if (eventPanelHideButton) {
  eventPanelHideButton.addEventListener("click", () => {
    setEventPanelOpen(false);
  });
}
setEventPanelOpen(false);
