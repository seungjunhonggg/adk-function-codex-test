// 기본 상수와 로컬 저장소 키를 정의한다.
const API_URL = "/api/chat";
const STORAGE_KEYS = {
  sessionId: "mlcc_demo_session_id",
  messages: "mlcc_demo_messages",
  demoMode: "mlcc_demo_mode",
  apiKey: "mlcc_demo_api_key",
  model: "mlcc_demo_model",
  baseUrl: "mlcc_demo_base_url",
};

// DOM 요소를 한 번만 가져와서 재사용한다.
const threadEl = document.getElementById("thread");
const emptyStateEl = document.getElementById("emptyState");
const composerEl = document.getElementById("composer");
const messageInputEl = document.getElementById("messageInput");
const demoToggleEl = document.getElementById("demoToggle");
const sessionIdEl = document.getElementById("sessionId");
const clearButtonEl = document.getElementById("clearButton");
const apiKeyInputEl = document.getElementById("apiKeyInput");
const modelInputEl = document.getElementById("modelInput");
const baseUrlInputEl = document.getElementById("baseUrlInput");

// 화면 상태를 단순 객체로 관리한다.
const state = {
  sessionId: "",
  demoMode: true,
  messages: [],
  typingEl: null,
  apiKey: "",
  model: "",
  baseUrl: "",
};

// 로컬 저장소에서 상태를 복원한다.
function loadState() {
  const storedSession = localStorage.getItem(STORAGE_KEYS.sessionId);
  const storedMessages = localStorage.getItem(STORAGE_KEYS.messages);
  const storedDemo = localStorage.getItem(STORAGE_KEYS.demoMode);
  const storedApiKey = localStorage.getItem(STORAGE_KEYS.apiKey);
  const storedModel = localStorage.getItem(STORAGE_KEYS.model);
  const storedBaseUrl = localStorage.getItem(STORAGE_KEYS.baseUrl);
  state.sessionId = storedSession || createSessionId();
  state.messages = storedMessages ? JSON.parse(storedMessages) : [];
  state.demoMode = storedDemo ? storedDemo === "true" : true;
  state.apiKey = storedApiKey || "";
  state.model = storedModel || "";
  state.baseUrl = storedBaseUrl || "";
}

// 로컬 저장소에 상태를 저장한다.
function saveState() {
  localStorage.setItem(STORAGE_KEYS.sessionId, state.sessionId);
  localStorage.setItem(STORAGE_KEYS.messages, JSON.stringify(state.messages));
  localStorage.setItem(STORAGE_KEYS.demoMode, String(state.demoMode));
  localStorage.setItem(STORAGE_KEYS.apiKey, state.apiKey);
  localStorage.setItem(STORAGE_KEYS.model, state.model);
  localStorage.setItem(STORAGE_KEYS.baseUrl, state.baseUrl);
}

// 세션 아이디를 생성한다.
function createSessionId() {
  if (window.crypto && window.crypto.randomUUID) {
    return `sess-${window.crypto.randomUUID()}`;
  }
  return `sess-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

// 입력창 높이를 자동으로 맞춘다.
function resizeInput() {
  messageInputEl.style.height = "auto";
  messageInputEl.style.height = `${messageInputEl.scrollHeight}px`;
}

// 스레드 하단으로 스크롤한다.
function scrollToBottom() {
  threadEl.scrollTop = threadEl.scrollHeight;
}

// 메시지를 상태에 추가하고 렌더링한다.
function addMessage(message) {
  state.messages.push(message);
  saveState();
  renderMessages();
}

// 타이핑 표시를 추가하거나 제거한다.
function setTyping(isTyping) {
  if (isTyping) {
    const typingEl = document.createElement("div");
    typingEl.className = "message message--assistant";
    typingEl.innerHTML = `
      <div class="assistant-card">
        <div class="assistant-meta">
          <span class="route-pill">thinking</span>
          <span>assistant</span>
        </div>
        <div class="block">
          <div class="block__text">...</div>
        </div>
      </div>
    `;
    state.typingEl = typingEl;
    threadEl.appendChild(typingEl);
    scrollToBottom();
  } else if (state.typingEl) {
    if (threadEl.contains(state.typingEl)) {
      threadEl.removeChild(state.typingEl);
    }
    state.typingEl = null;
  }
}

// 메시지 목록을 화면에 렌더링한다.
function renderMessages() {
  threadEl.innerHTML = "";
  if (state.messages.length === 0 && emptyStateEl) {
    threadEl.appendChild(emptyStateEl);
  } else if (emptyStateEl && emptyStateEl.parentElement) {
    emptyStateEl.remove();
  }
  state.messages.forEach((message) => {
    if (message.role === "user") {
      threadEl.appendChild(renderUserMessage(message));
    } else {
      threadEl.appendChild(renderAssistantMessage(message));
    }
  });
  scrollToBottom();
}

// 사용자 메시지 DOM을 만든다.
function renderUserMessage(message) {
  const wrapper = document.createElement("div");
  wrapper.className = "message message--user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message.text;
  wrapper.appendChild(bubble);
  return wrapper;
}

// 어시스턴트 메시지 DOM을 만든다.
function renderAssistantMessage(message) {
  const wrapper = document.createElement("div");
  wrapper.className = "message message--assistant";
  const card = document.createElement("div");
  card.className = "assistant-card";

  const meta = document.createElement("div");
  meta.className = "assistant-meta";
  const pill = document.createElement("span");
  pill.className = "route-pill";
  pill.textContent = message.route || "assistant";
  const metaText = document.createElement("span");
  metaText.textContent = "assistant";
  meta.appendChild(pill);
  meta.appendChild(metaText);
  card.appendChild(meta);

  const blocks = message.blocks || [];
  blocks.forEach((block) => {
    card.appendChild(renderBlock(block, message.tables, message.charts));
  });

  wrapper.appendChild(card);
  return wrapper;
}

// 블록 타입에 따라 카드 내용을 만든다.
function renderBlock(block, tables, charts) {
  if (block.type === "table_ref") {
    return renderTableCard(block.table_key, tables);
  }
  if (block.type === "chart_ref") {
    return renderChartCard(block.chart_id, charts);
  }
  return renderTextCard(block);
}

// 텍스트 블록을 만든다.
function renderTextCard(block) {
  const card = document.createElement("div");
  card.className = "block";
  if (block.section) {
    const label = document.createElement("div");
    label.className = "block__label";
    label.textContent = block.section;
    card.appendChild(label);
  }
  const text = document.createElement("div");
  text.className = "block__text";
  text.textContent = block.value || "";
  card.appendChild(text);
  return card;
}

// 테이블 블록을 만든다.
function renderTableCard(tableKey, tables) {
  const card = document.createElement("div");
  card.className = "block table-card";
  const label = document.createElement("div");
  label.className = "block__label";
  label.textContent = tableKey || "table";
  card.appendChild(label);

  const rows = tables && tableKey ? tables[tableKey] : null;
  if (!rows || !Array.isArray(rows) || rows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "block__text";
    empty.textContent = "No table data.";
    card.appendChild(empty);
    return card;
  }

  const columns = Object.keys(rows[0] || {});
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach((col) => {
    const th = document.createElement("th");
    th.textContent = col;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((col) => {
      const td = document.createElement("td");
      const value = row[col] === null || row[col] === undefined ? "" : row[col];
      td.textContent = String(value);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  card.appendChild(table);
  return card;
}

// 차트 블록을 만든다.
function renderChartCard(chartId, charts) {
  const card = document.createElement("div");
  card.className = "block chart-card";
  const label = document.createElement("div");
  label.className = "block__label";
  label.textContent = chartId || "chart";
  card.appendChild(label);

  const chart =
    charts && chartId ? charts.find((item) => item.chart_id === chartId) : null;
  if (!chart) {
    const empty = document.createElement("div");
    empty.className = "block__text";
    empty.textContent = "No chart data.";
    card.appendChild(empty);
    return card;
  }

  const title = document.createElement("div");
  title.className = "chart-title";
  title.textContent = chart.title || "Chart";
  card.appendChild(title);

  const svg = buildSvgChart(chart);
  svg.classList.add("chart-canvas");
  card.appendChild(svg);
  return card;
}

// SVG 차트를 생성한다.
function buildSvgChart(chart) {
  const width = 360;
  const height = 180;
  const padding = 28;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const series = chart.series && chart.series[0] ? chart.series[0] : null;
  const points = series ? series.points || [] : [];
  const values = points.map((point) => Number(point.y) || 0);
  const maxValue = Math.max(...values, 1);

  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axis.setAttribute("x1", padding);
  axis.setAttribute("y1", height - padding);
  axis.setAttribute("x2", width - padding);
  axis.setAttribute("y2", height - padding);
  axis.setAttribute("stroke", "rgba(0,0,0,0.2)");
  axis.setAttribute("stroke-width", "1");
  svg.appendChild(axis);

  if (chart.type === "line") {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    let d = "";
    points.forEach((point, index) => {
      const x =
        padding +
        (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
      const y =
        height -
        padding -
        ((Number(point.y) || 0) / maxValue) * (height - padding * 2);
      d += index === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
    });
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#0d6c63");
    path.setAttribute("stroke-width", "2");
    svg.appendChild(path);
  } else if (chart.type === "scatter") {
    points.forEach((point, index) => {
      const x =
        padding +
        (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
      const y =
        height -
        padding -
        ((Number(point.y) || 0) / maxValue) * (height - padding * 2);
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", y);
      dot.setAttribute("r", "4");
      dot.setAttribute("fill", "#b5842f");
      svg.appendChild(dot);
    });
  } else {
    const barWidth = (width - padding * 2) / Math.max(points.length, 1);
    points.forEach((point, index) => {
      const value = Number(point.y) || 0;
      const barHeight = (value / maxValue) * (height - padding * 2);
      const x = padding + index * barWidth + barWidth * 0.2;
      const y = height - padding - barHeight;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", barWidth * 0.6);
      rect.setAttribute("height", barHeight);
      rect.setAttribute("rx", "4");
      rect.setAttribute("fill", "#0d6c63");
      svg.appendChild(rect);
    });
  }

  return svg;
}

// 오버라이드 페이로드를 만든다.
function buildOverrides() {
  const overrides = {};
  if (state.apiKey) {
    overrides.api_key = state.apiKey;
  }
  if (state.model) {
    overrides.model = state.model;
  }
  if (state.baseUrl) {
    overrides.base_url = state.baseUrl;
  }
  if (Object.keys(overrides).length === 0) {
    return null;
  }
  return overrides;
}

// 사용자 입력을 서버로 전송한다.
async function sendMessage(text) {
  setTyping(true);
  try {
    const overrides = buildOverrides();
    const payload = {
      session_id: state.sessionId,
      message: text,
      demo: state.demoMode,
    };
    if (overrides) {
      payload.overrides = overrides;
    }
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    addMessage({
      role: "assistant",
      route: data.route,
      blocks: data.blocks || [],
      tables: data.tables || {},
      charts: data.charts || [],
    });
  } catch (error) {
    addMessage({
      role: "assistant",
      route: "error",
      blocks: [
        {
          type: "text",
          section: "error",
          value: "Request failed. Check the server and try again.",
        },
      ],
      tables: {},
      charts: [],
    });
  } finally {
    setTyping(false);
  }
}

// 세션 정보를 화면에 반영한다.
function updateSessionUi() {
  sessionIdEl.textContent = state.sessionId;
  demoToggleEl.checked = state.demoMode;
  apiKeyInputEl.value = state.apiKey;
  modelInputEl.value = state.model;
  baseUrlInputEl.value = state.baseUrl;
}

// 입력 폼 이벤트를 등록한다.
composerEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInputEl.value.trim();
  if (!text) {
    return;
  }
  addMessage({ role: "user", text });
  messageInputEl.value = "";
  resizeInput();
  sendMessage(text);
});

// 엔터 키 동작을 제어한다.
messageInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composerEl.requestSubmit();
  }
});

// 입력창 크기를 자동으로 갱신한다.
messageInputEl.addEventListener("input", resizeInput);

// 데모 모드 토글을 저장한다.
demoToggleEl.addEventListener("change", () => {
  state.demoMode = demoToggleEl.checked;
  saveState();
});

// API 키 입력을 저장한다.
apiKeyInputEl.addEventListener("input", () => {
  state.apiKey = apiKeyInputEl.value.trim();
  saveState();
});

// 모델 입력을 저장한다.
modelInputEl.addEventListener("input", () => {
  state.model = modelInputEl.value.trim();
  saveState();
});

// Base URL 입력을 저장한다.
baseUrlInputEl.addEventListener("input", () => {
  state.baseUrl = baseUrlInputEl.value.trim();
  saveState();
});

// 대화 내용을 초기화한다.
clearButtonEl.addEventListener("click", () => {
  state.sessionId = createSessionId();
  state.messages = [];
  saveState();
  updateSessionUi();
  renderMessages();
});

// 초기 로딩을 수행한다.
loadState();
updateSessionUi();
renderMessages();
resizeInput();
