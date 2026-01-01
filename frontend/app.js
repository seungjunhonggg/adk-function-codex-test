const messages = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const statusDot = document.getElementById("ws-status");
const statusLabel = document.getElementById("ws-label");
const sessionEl = document.getElementById("session-id");
const lotResultEl = document.getElementById("lot-result");
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
const designCandidatesCard = document.getElementById("design-candidates-card");
const predictionCard = document.getElementById("prediction-card");
const frontendCard = document.getElementById("frontend-card");
const eventLogCard = document.getElementById("event-log-card");
const eventEmptyEl = document.getElementById("event-empty");
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
const designCandidatesEl = document.getElementById("design-candidates");
const sessionListEl = document.getElementById("session-list");
const sessionEmptyEl = document.getElementById("session-empty");
const sessionCountEl = document.getElementById("session-count");
const newSessionButton = document.getElementById("new-session");

const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
let socket = null;
let socketPingInterval = null;
let suppressSocketCloseNotice = false;

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
let isComposing = false;
let recommendationDirty = false;

let historyEntries = [];

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

function appendMessageToChat(entry) {
  const message = document.createElement("div");
  message.className = `message ${entry.role}`;
  message.dataset.messageId = entry.id;
  message.textContent = entry.text;
  messages.appendChild(message);
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, text) {
  const entry = buildHistoryEntry(role, text);
  historyEntries.push(entry);
  saveHistory(currentSessionId, historyEntries);
  appendMessageToChat(entry);
  updateSessionMeta(currentSessionId, entry);
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

const streamCards = [
  lotCard,
  simFormCard,
  simResultCard,
  defectChartCard,
  designCandidatesCard,
  predictionCard,
  frontendCard,
].filter(Boolean);
let isLogOpen = false;
if (eventLogCard) {
  isLogOpen = true;
  eventLogCard.classList.remove("hidden");
  if (toggleLogButton) {
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
  updateEventEmpty();
}

function showOnlyStreamCards(cards = []) {
  hideAllStreamCards();
  cards.forEach((card) => {
    if (card) {
      card.classList.remove("hidden");
    }
  });
  updateEventEmpty();
}

function toggleEventLog() {
  if (!eventLogCard || !toggleLogButton) {
    return;
  }
  isLogOpen = !isLogOpen;
  eventLogCard.classList.toggle("hidden", !isLogOpen);
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

function updateMissingInputs(missing = []) {
  const missingSet = new Set(missing || []);
  const mapping = [
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
  showOnlyStreamCards([simFormCard]);
  const params = payload.params || {};
  applyInputValue(simModelInput, params.model_name);
  applyInputValue(simTemperatureInput, params.temperature);
  applyInputValue(simVoltageInput, params.voltage);
  applyInputValue(simSizeInput, params.size);
  applyInputValue(simCapacityInput, params.capacity);
  applyInputValue(simProductionSelect, params.production_mode);
  updateMissingInputs(payload.missing || []);
  updateFilledInputs();

  if (payload.missing && payload.missing.length) {
    const missingLabel = payload.missing
      .map((item) => {
        if (item === "production_mode") {
          return "양산/개발";
        }
        if (item === "temperature") return "온도";
        if (item === "voltage") return "전압";
        if (item === "size") return "크기";
        if (item === "capacity") return "용량";
        return item;
      })
      .join(", ");
    setSimStatus(`미입력: ${missingLabel}`, true);
  } else {
    setSimStatus("입력 완료: 추천 실행이 가능합니다.");
  }
}

function collectSimFormParams() {
  const modelName = simModelInput?.value.trim();
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
    model_name: modelName || null,
    temperature: temperature || null,
    voltage: parseNumber(voltage),
    size: parseNumber(size),
    capacity: parseNumber(capacity),
    production_mode: productionMode || null,
  };
}

async function sendSimulationParams({ run } = { run: false }) {
  if (!simFormStatus) {
    return;
  }
  setSimStatus("추천 입력 전송 중...");
  const params = collectSimFormParams();
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
      renderSimResult({ params: data.params || params, result: data.result });
      setSimStatus("추천 실행 완료");
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

  showOnlyStreamCards([lotCard]);

  const lotId = payload.lot_id || payload.query || "";
  setCurrentLot(lotId);
  const row = payload.rows[0] || {};
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
    dt.textContent = key;
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

  showOnlyStreamCards([simResultCard]);
  recommendationDirty = false;

  const params = payload.params || {};
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
  if (result.recommended_model) {
    kpiRow.appendChild(
      createKpiCard("추천 기종", result.recommended_model, "accent")
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

  const productionLabel =
    params.production_mode === "mass"
      ? "양산"
      : params.production_mode === "dev"
        ? "개발"
        : params.production_mode || "-";
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

function renderDefectRateChart(payload = {}) {
  if (!defectChartEl) {
    return;
  }
  const lots = Array.isArray(payload.lots) ? payload.lots : [];
  if (!lots.length) {
    defectChartEl.textContent = "불량률 그래프가 없습니다.";
    return;
  }
  showStreamCard(defectChartCard);

  const stats = payload.stats || {};
  const maxRate = Math.max(
    ...lots.map((item) => Number(item.defect_rate) || 0),
    0.0001
  );

  defectChartEl.innerHTML = "";
  if (stats && stats.count) {
    const meta = document.createElement("div");
    meta.className = "candidate-meta";
    const avg = stats.avg ? `${(stats.avg * 100).toFixed(2)}%` : "--";
    const min = stats.min ? `${(stats.min * 100).toFixed(2)}%` : "--";
    const max = stats.max ? `${(stats.max * 100).toFixed(2)}%` : "--";
    meta.textContent = `필터 ${stats.count}건 · 평균 ${avg} · 최소 ${min} · 최대 ${max}`;
    defectChartEl.appendChild(meta);
  }

  const chart = document.createElement("div");
  chart.className = "defect-chart";
  lots.forEach((item) => {
    const rate = Number(item.defect_rate) || 0;
    const percent = (rate * 100).toFixed(2);
    const row = document.createElement("div");
    row.className = "defect-row";
    const label = document.createElement("div");
    label.className = "defect-label";
    label.textContent = item.lot_id || "-";
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
  defectChartEl.appendChild(chart);
}

function renderDesignCandidates(payload = {}) {
  if (!designCandidatesEl) {
    return;
  }
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
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
            .map(([key, value]) => `${key}=${renderValue(value)}`)
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

  const table = document.createElement("table");
  table.className = "result-table";
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
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
  const modelName = recommendation.recommended_model || "-";
  const repLot = recommendation.representative_lot || "-";

  predictionResultEl.innerHTML = "";

  const kpiRow = document.createElement("div");
  kpiRow.className = "kpi-row";
  kpiRow.appendChild(createKpiCard("추천 기종", modelName, "accent"));
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
    renderSimResult(event.payload);
    const modelName = event.payload?.result?.recommended_model || "결과 수신";
    addEventLog("추천", `기종: ${modelName}`);
  }

  if (event.type === "defect_rate_chart") {
    renderDefectRateChart(event.payload);
    addEventLog("불량률", "그래프 업데이트");
  }

  if (event.type === "design_candidates") {
    renderDesignCandidates(event.payload);
    addEventLog("설계", "후보 업데이트");
  }

  if (event.type === "prediction_result") {
    renderPredictionResult(event.payload);
    const prob = event.payload?.result?.reliability_pass_prob;
    const detail = prob ? `통과확률: ${prob}` : "결과 수신";
    addEventLog("예측", detail);
  }

  if (event.type === "workflow_log") {
    const label = event.payload?.label || "LOG";
    const detail = event.payload?.detail || "";
    addEventLog(label, detail);
    updateEventEmpty();
  }

  if (event.type === "frontend_trigger") {
    const message = event.payload?.message || "프론트 트리거가 도착했습니다.";
    const payloadData = event.payload?.data;
    const hasResultPayload =
      payloadData &&
      typeof payloadData === "object" &&
      ("result" in payloadData || "rows" in payloadData);
    if (frontendTriggerEl) {
      frontendTriggerEl.textContent = message;
    }
    if (!hasResultPayload) {
      showOnlyStreamCards([frontendCard]);
    }
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

function connectWebSocket(sessionId) {
  if (!sessionId) {
    return;
  }
  if (socket) {
    suppressSocketCloseNotice = true;
    socket.close();
  }
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

async function sendChatMessage(message) {
  addMessage("user", message);
  input.value = "";
  input.style.height = "auto";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: currentSessionId, message }),
    });

    if (!response.ok) {
      addMessage("assistant", "서버 오류입니다. 잠시 후 다시 시도해 주세요.");
      return;
    }

    const data = await response.json();
    addMessage("assistant", data.assistant_message || "(응답 없음)");
  } catch (error) {
    addMessage("assistant", "네트워크 오류입니다. 백엔드 상태를 확인해 주세요.");
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
