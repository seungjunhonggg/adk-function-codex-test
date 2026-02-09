// 기본 상수와 로컬 저장소 키를 정의한다.
const API_URL = "/chat";
const API_STREAM_URL = "/chat/stream";
const STORAGE_KEYS = {
  sessionId: "mlcc_demo_session_id",
  messages: "mlcc_demo_messages",
  demoMode: "mlcc_demo_mode",
};

// DOM 요소를 한 번만 가져와서 재사용한다.
const threadEl = document.getElementById("thread");
const composerEl = document.getElementById("composer");
const messageInputEl = document.getElementById("messageInput");
const demoToggleEl = document.getElementById("demoToggle");
const sessionIdEl = document.getElementById("sessionId");
const clearButtonEl = document.getElementById("clearButton");
const newChatButtonEl = document.getElementById("newChatButton");
const insightsToggleEl = document.getElementById("insightsToggle");
const appEl = document.querySelector(".app");
const insightsEl = document.querySelector(".insights");
// 조합 입력 진행 여부를 저장한다.
let isComposing = false;
// 조합 종료 후 전송 예약 여부를 저장한다.
let pendingSubmit = false;

// 화면 상태를 단순 객체로 관리한다.
const state = {
  sessionId: "",
  demoMode: false,
  messages: [],
  typingEl: null,
  insightsHidden: true,
};

// 로컬 저장소에서 상태를 복원한다.
function loadState() {
  const storedSession = localStorage.getItem(STORAGE_KEYS.sessionId);
  const storedMessages = localStorage.getItem(STORAGE_KEYS.messages);
  const storedDemo = localStorage.getItem(STORAGE_KEYS.demoMode);
  state.sessionId = storedSession || createSessionId();
  state.messages = storedMessages ? JSON.parse(storedMessages) : [];
  state.demoMode = storedDemo ? storedDemo === "true" : false;
}

// 로컬 저장소에 상태를 저장한다.
function saveState() {
  localStorage.setItem(STORAGE_KEYS.sessionId, state.sessionId);
  localStorage.setItem(STORAGE_KEYS.messages, JSON.stringify(state.messages));
  localStorage.setItem(STORAGE_KEYS.demoMode, String(state.demoMode));
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
    const card = document.createElement("div");
    card.className = "assistant-card";

    const meta = document.createElement("div");
    meta.className = "assistant-meta";
    const pill = document.createElement("span");
    pill.className = "route-pill";
    pill.textContent = "thinking";
    const metaText = document.createElement("span");
    metaText.textContent = "assistant";
    meta.appendChild(pill);
    meta.appendChild(metaText);
    card.appendChild(meta);

    card.appendChild(
      renderProgressLog([{ text: "답변 생성하는 중", status: "in_progress" }])
    );
    typingEl.appendChild(card);
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

// 진행 로그 DOM을 만든다.
function renderProgressLog(logs) {
  const container = document.createElement("div");
  container.className = "progress-log";
  if (!Array.isArray(logs) || logs.length === 0) {
    return container;
  }
  logs.forEach((log) => {
    const row = document.createElement("div");
    row.className = "progress-log__row";
    const logStatus = log.status || "in_progress";
    if (logStatus !== "in_progress") {
      row.classList.add("is-static");
    }
    if (logStatus === "done") {
      row.classList.add("is-done");
    }

    const spinner = document.createElement("span");
    spinner.className = "progress-log__spinner";
    row.appendChild(spinner);

    const text = document.createElement("span");
    text.textContent = log.text || "";
    row.appendChild(text);

    const status = document.createElement("span");
    status.className = "progress-log__status";
    if (logStatus === "done") {
      status.textContent = "완료";
    } else if (logStatus === "pending") {
      status.textContent = "대기";
    } else if (logStatus === "error") {
      status.textContent = "오류";
    } else {
      status.textContent = "진행중";
      const dots = document.createElement("span");
      dots.className = "progress-log__dots";
      dots.innerHTML = "<span></span><span></span><span></span>";
      status.appendChild(dots);
    }
    row.appendChild(status);
    container.appendChild(row);
  });
  return container;
}

// 블록 타입에 따라 카드 내용을 만든다.
function renderBlock(block, tables, charts) {
  if (block.type === "table_ref") {
    return renderTableCard(block.table_key, tables);
  }
  if (block.type === "chart_ref") {
    return renderChartCard(block.chart_id, charts);
  }
  if (block.type === "input_form") {
    return renderInputForm(block);
  }
  if (block.type === "progress_log") {
    return renderProgressLog(block.logs || []);
  }
  return renderTextCard(block);
}

// 입력 폼을 만든다.
function renderInputForm(block) {
  const wrapper = document.createElement("div");
  wrapper.className = "input-form-wrapper";

  const header = document.createElement("div");
  header.className = "form-header";
  if (block.title) {
    const title = document.createElement("div");
    title.className = "form-title";
    title.textContent = block.title;
    header.appendChild(title);
  }
  const desc = document.createElement("div");
  desc.className = "form-description";
  desc.textContent = "아래 두 가지 방법 중 하나를 선택해주세요.";
  header.appendChild(desc);
  wrapper.appendChild(header);

  const boxContainer = document.createElement("div");
  boxContainer.className = "form-box-container";

  const fields = Array.isArray(block.fields) ? block.fields : [];
  const leftFields = fields.filter((f) => f.column !== "right");
  const rightFields = fields.filter((f) => f.column === "right");

  // 왼쪽 박스 (조건 직접 입력)
  const leftBox = document.createElement("div");
  leftBox.className = "input-form-card input-form-card--left";
  leftBox.dataset.formType = "core";

  const leftHeader = document.createElement("div");
  leftHeader.className = "form-box-header";
  const leftLabel = document.createElement("div");
  leftLabel.className = "form-box-label";
  leftLabel.textContent = "방법 1";
  const leftTitle = document.createElement("div");
  leftTitle.className = "form-box-title";
  leftTitle.textContent = "조건 직접 입력";
  leftHeader.appendChild(leftLabel);
  leftHeader.appendChild(leftTitle);
  leftBox.appendChild(leftHeader);

  const leftGrid = document.createElement("div");
  leftGrid.className = "form-grid-2x2";
  leftFields.forEach((field) => {
    leftGrid.appendChild(createFormGroup(field));
  });
  leftBox.appendChild(leftGrid);

  const leftActions = document.createElement("div");
  leftActions.className = "form-actions";
  const leftSubmitBtn = document.createElement("button");
  leftSubmitBtn.className = "form-submit-btn";
  leftSubmitBtn.textContent = block.submit_label || "시뮬레이션 시작";
  leftSubmitBtn.type = "button";
  if (block.submitted) {
    leftBox.classList.add("is-submitted");
    leftSubmitBtn.disabled = true;
    leftSubmitBtn.textContent = "Submitted";
  }
  leftSubmitBtn.addEventListener("click", () => {
    handleFormSubmit(leftBox, block.form_id, "core");
  });
  leftActions.appendChild(leftSubmitBtn);
  leftBox.appendChild(leftActions);
  boxContainer.appendChild(leftBox);

  // OR 구분자
  const orDivider = document.createElement("div");
  orDivider.className = "form-or-divider";
  const orText = document.createElement("span");
  orText.className = "form-or-text";
  orText.textContent = "또는";
  orDivider.appendChild(orText);
  boxContainer.appendChild(orDivider);

  // 오른쪽 박스 (CHIP 기종 검색)
  const rightBox = document.createElement("div");
  rightBox.className = "input-form-card input-form-card--right";
  rightBox.dataset.formType = "chip";

  const rightHeader = document.createElement("div");
  rightHeader.className = "form-box-header";
  const rightLabel = document.createElement("div");
  rightLabel.className = "form-box-label";
  rightLabel.textContent = "방법 2";
  const rightTitle = document.createElement("div");
  rightTitle.className = "form-box-title";
  rightTitle.textContent = "CHIP 기종으로 검색";
  rightHeader.appendChild(rightLabel);
  rightHeader.appendChild(rightTitle);
  rightBox.appendChild(rightHeader);

  const rightContent = document.createElement("div");
  rightContent.className = "form-chip-content";
  rightFields.forEach((field) => {
    rightContent.appendChild(createFormGroup(field));
  });
  rightBox.appendChild(rightContent);

  const rightActions = document.createElement("div");
  rightActions.className = "form-actions";
  const rightSubmitBtn = document.createElement("button");
  rightSubmitBtn.className = "form-submit-btn";
  rightSubmitBtn.textContent = block.submit_label || "시뮬레이션 시작";
  rightSubmitBtn.type = "button";
  if (block.submitted) {
    rightBox.classList.add("is-submitted");
    rightSubmitBtn.disabled = true;
    rightSubmitBtn.textContent = "Submitted";
  }
  rightSubmitBtn.addEventListener("click", () => {
    handleFormSubmit(rightBox, block.form_id, "chip");
  });
  rightActions.appendChild(rightSubmitBtn);
  rightBox.appendChild(rightActions);
  boxContainer.appendChild(rightBox);

  wrapper.appendChild(boxContainer);
  return wrapper;
}

// 폼 그룹 생성 헬퍼 함수
function createFormGroup(field) {
  const group = document.createElement("div");
  group.className = "form-group";

  const label = document.createElement("label");
  label.className = "form-label";
  label.textContent = field.label || field.key;
  group.appendChild(label);

  if (field.type === "select" && Array.isArray(field.options)) {
    const input = document.createElement("select");
    input.className = "form-input";
    input.name = field.key;
    if (!field.value) {
      const placeholder = document.createElement("option");
      placeholder.text = "선택해주세요";
      placeholder.value = "";
      placeholder.disabled = true;
      placeholder.selected = true;
      input.appendChild(placeholder);
    }
    field.options.forEach((opt) => {
      const option = document.createElement("option");
      option.value = opt;
      option.textContent = opt + (field.unit ? ` ${field.unit}` : "");
      if (opt === field.value) option.selected = true;
      input.appendChild(option);
    });
    group.appendChild(input);
  } else if (field.unit_options && Array.isArray(field.unit_options)) {
    const inputWrapper = document.createElement("div");
    inputWrapper.className = "form-input-group";
    const input = document.createElement("input");
    input.className = "form-input";
    input.type = "number";
    input.name = field.key;
    input.value = field.value || "";
    input.placeholder = field.label || "값 입력";
    const unitSelect = document.createElement("select");
    unitSelect.className = "form-input form-input-unit";
    unitSelect.name = `${field.key}_unit`;
    field.unit_options.forEach((opt, idx) => {
      const option = document.createElement("option");
      option.value = opt;
      option.textContent = opt;
      if (idx === 0) option.selected = true;
      unitSelect.appendChild(option);
    });
    inputWrapper.appendChild(input);
    inputWrapper.appendChild(unitSelect);
    group.appendChild(inputWrapper);
  } else {
    const input = document.createElement("input");
    input.className = "form-input";
    input.type = field.type || "text";
    input.name = field.key;
    input.value = field.value || "";
    if (field.placeholder) {
      input.placeholder = field.placeholder;
    } else if (field.label) {
      input.placeholder = field.label;
    }
    group.appendChild(input);
  }

  const errorText = document.createElement("div");
  errorText.className = "form-error-text";
  errorText.textContent = "입력이 필요합니다";
  group.appendChild(errorText);

  return group;
}

// 폼 제출 처리
function handleFormSubmit(cardEl, formId, formType) {
  const groups = cardEl.querySelectorAll(".form-group");
  const data = {};
  const entries = [];
  let isValid = true;
  const coreKeys = ["temperature", "size", "capacity", "voltage"];
  const chipKey = "chip_prod_id";

  groups.forEach((group) => {
    const input = Array.from(group.querySelectorAll(".form-input")).find(
      (el) => !el.name.endsWith("_unit")
    );
    if (!input) return;
    const errorText = group.querySelector(".form-error-text");
    const unitSelect = group.querySelector(
      `select[name="${input.name}_unit"]`
    );
    const val = input.value.trim();
    entries.push({ input, errorText, unitSelect, value: val, key: input.name });
  });

  entries.forEach((entry) => {
    entry.input.classList.remove("has-error");
    if (entry.errorText) entry.errorText.classList.remove("is-visible");
    void entry.input.offsetWidth;

    let isRequired = false;
    if (formType === "core") {
      isRequired = coreKeys.includes(entry.key);
    } else if (formType === "chip") {
      isRequired = entry.key === chipKey;
    }
    if (isRequired && !entry.value) {
      isValid = false;
      entry.input.classList.add("has-error");
      if (entry.errorText) entry.errorText.classList.add("is-visible");
      return;
    }
    if (!entry.value) return;

    let finalValue = entry.value;
    if (entry.unitSelect) {
      const unit = entry.unitSelect.value;
      const numVal = parseFloat(entry.value);
      if (!isNaN(numVal)) {
        if (unit === "nF") finalValue = String(numVal * 1000);
        else if (unit === "uF") finalValue = String(numVal * 1000000);
      }
    }
    data[entry.key] = finalValue;
  });

  if (!isValid) return;

  cardEl.classList.add("is-submitted");
  const btn = cardEl.querySelector(".form-submit-btn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "처리 중...";
  }
  const wrapper = cardEl.closest(".input-form-wrapper");
  if (wrapper) wrapper.classList.add("is-submitted");

  const messageText = JSON.stringify(data, null, 2);
  sendMessage(messageText);
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
  text.appendChild(renderMarkdownLite(block.value || ""));
  card.appendChild(text);
  return card;
}

// 마크다운 라이트 렌더러: 줄바꿈, 테이블, 볼드, 리스트를 처리한다.
function renderMarkdownLite(raw) {
  const container = document.createElement("div");
  container.className = "md-rendered";
  const lines = raw.split("\n");
  let i = 0;

  while (i < lines.length) {
    // 마크다운 테이블 감지: | 로 시작하는 연속 행
    if (lines[i].trim().startsWith("|")) {
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      if (tableLines.length >= 2) {
        container.appendChild(buildMdTable(tableLines));
      } else {
        container.appendChild(mdParagraph(tableLines.join("\n")));
      }
      continue;
    }

    // 빈 줄은 건너뛴다.
    if (lines[i].trim() === "") {
      i++;
      continue;
    }

    // 일반 텍스트: 다음 빈 줄이나 테이블 시작까지 모은다.
    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].trim().startsWith("|")
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    container.appendChild(mdParagraph(paraLines.join("\n")));
  }

  return container;
}

// 일반 텍스트를 <p>로 만든다. 인라인 마크다운(볼드)과 줄바꿈을 처리한다.
function mdParagraph(text) {
  const p = document.createElement("p");
  p.className = "md-para";
  // 줄바꿈을 <br>로, **bold**를 <strong>으로 변환한다.
  const html = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
  p.innerHTML = html;
  return p;
}

// 마크다운 테이블 행들을 <table>로 만든다.
function buildMdTable(tableLines) {
  const wrapper = document.createElement("div");
  wrapper.className = "block table-card md-table-wrapper";
  const table = document.createElement("table");
  table.className = "md-table";

  const rows = tableLines.map((line) =>
    line
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim())
  );

  // 구분선 행(---) 위치를 찾는다.
  let separatorIdx = -1;
  for (let r = 0; r < rows.length; r++) {
    if (rows[r].every((cell) => /^[-:\s]+$/.test(cell))) {
      separatorIdx = r;
      break;
    }
  }

  // 헤더가 있으면 thead를 만든다.
  let bodyStart = 0;
  if (separatorIdx >= 0) {
    const thead = document.createElement("thead");
    for (let r = 0; r < separatorIdx; r++) {
      const tr = document.createElement("tr");
      rows[r].forEach((cell) => {
        const th = document.createElement("th");
        th.innerHTML = inlineMd(cell);
        tr.appendChild(th);
      });
      thead.appendChild(tr);
    }
    table.appendChild(thead);
    bodyStart = separatorIdx + 1;
  }

  // tbody를 만든다.
  const tbody = document.createElement("tbody");
  for (let r = bodyStart; r < rows.length; r++) {
    const tr = document.createElement("tr");
    rows[r].forEach((cell) => {
      const td = document.createElement("td");
      td.innerHTML = inlineMd(cell);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrapper.appendChild(table);
  return wrapper;
}

// 인라인 마크다운(볼드)을 변환한다.
function inlineMd(text) {
  return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

// HTML 특수문자를 이스케이프한다.
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
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

  const columnSet = new Set();
  rows.forEach((row) => {
    if (!row || typeof row !== "object") return;
    Object.keys(row).forEach((key) => {
      if (!key.startsWith("__")) columnSet.add(key);
    });
  });
  const columns = Array.from(columnSet);
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
    if (row && row.__row_state) tr.classList.add("table-row--highlight");
    columns.forEach((col) => {
      const td = document.createElement("td");
      const value =
        row[col] === null || row[col] === undefined ? "" : row[col];
      td.textContent = String(value);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  card.appendChild(table);
  return card;
}

// 차트 블록 (placeholder)
function renderChartCard(chartId, charts) {
  const card = document.createElement("div");
  card.className = "block chart-card";
  const label = document.createElement("div");
  label.className = "block__label";
  label.textContent = chartId || "chart";
  card.appendChild(label);
  const empty = document.createElement("div");
  empty.className = "block__text";
  empty.textContent = "Chart rendering placeholder.";
  card.appendChild(empty);
  return card;
}

// 현재 활성 스트림 컨트롤러를 저장한다.
let activeStreamController = null;

// 타이핑 로그를 갱신한다.
function updateTypingLogs(logs) {
  if (!state.typingEl) return;
  const card = state.typingEl.querySelector(".assistant-card");
  if (!card) return;
  const nextLog = renderProgressLog(logs || []);
  const currentLog = card.querySelector(".progress-log");
  if (currentLog) {
    card.replaceChild(nextLog, currentLog);
  } else {
    card.appendChild(nextLog);
  }
  scrollToBottom();
}

// SSE 청크를 파싱한다.
function parseSseChunk(chunk) {
  let eventName = "message";
  const dataLines = [];
  chunk.split("\n").forEach((line) => {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
      return;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  });
  return { event: eventName, data: dataLines.join("\n") };
}

// 사용자 입력을 서버로 전송한다 (SSE 스트리밍).
async function sendMessage(text) {
  setTyping(true);

  // 이전 스트림이 있으면 중단한다.
  if (activeStreamController) {
    activeStreamController.abort();
  }
  activeStreamController = new AbortController();

  try {
    const payload = {
      session_id: state.sessionId,
      message: text,
    };

    const response = await fetch(API_STREAM_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: activeStreamController.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let hasFinal = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";

      chunks.forEach((chunk) => {
        const trimmed = chunk.trim();
        if (!trimmed) return;
        const parsed = parseSseChunk(trimmed);
        if (!parsed) return;

        if (parsed.event === "progress") {
          const progressData = JSON.parse(parsed.data || "{}");
          updateTypingLogs(progressData.logs || []);
          return;
        }

        if (parsed.event === "trigger") {
          const triggerData = JSON.parse(parsed.data || "{}");
          // 트리거를 input_form 블록으로 렌더링한다.
          setTyping(false);
          addMessage({
            role: "assistant",
            route: "mlcc_agent",
            blocks: [triggerData],
            tables: {},
            charts: [],
          });
          setTyping(true);
          return;
        }

        if (parsed.event === "final") {
          hasFinal = true;
          const data = JSON.parse(parsed.data || "{}");
          if (data.session_id) {
            state.sessionId = data.session_id;
            updateSessionUi();
          }
          const responseText = data.response || "";
          if (responseText) {
            addMessage({
              role: "assistant",
              route: "mlcc_agent",
              blocks: [{ type: "text", section: "응답", value: responseText }],
              tables: {},
              charts: [],
            });
          }
        }
      });
    }

    if (!hasFinal) {
      throw new Error("final event missing");
    }
  } catch (error) {
    if (error && error.name === "AbortError") return;
    addMessage({
      role: "assistant",
      route: "error",
      blocks: [
        {
          type: "text",
          section: "error",
          value: "처리 중 문제가 발생했어요. 잠시 후 다시 시도해주세요.",
        },
      ],
      tables: {},
      charts: [],
    });
  } finally {
    setTyping(false);
    activeStreamController = null;
  }
}

// 세션 정보를 UI에 반영한다.
function updateSessionUi() {
  if (sessionIdEl) sessionIdEl.textContent = state.sessionId;
  if (demoToggleEl) demoToggleEl.checked = state.demoMode;
}

// 입력 폼 이벤트를 등록한다.
composerEl.addEventListener("submit", (event) => {
  event.preventDefault();
  if (isComposing) {
    pendingSubmit = true;
    return;
  }
  const text = messageInputEl.value.trim();
  if (!text) return;
  addMessage({ role: "user", text });
  messageInputEl.value = "";
  resizeInput();
  sendMessage(text);
});

// 엔터 키 동작을 제어한다.
messageInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    if (event.isComposing || isComposing || event.keyCode === 229) {
      pendingSubmit = true;
      return;
    }
    event.preventDefault();
    composerEl.requestSubmit();
  }
});

// 조합 입력 시작/종료를 기록한다.
messageInputEl.addEventListener("compositionstart", () => {
  isComposing = true;
});
messageInputEl.addEventListener("compositionend", () => {
  isComposing = false;
  if (pendingSubmit) {
    pendingSubmit = false;
    composerEl.requestSubmit();
  }
  resizeInput();
});

messageInputEl.addEventListener("input", resizeInput);

if (demoToggleEl) {
  demoToggleEl.addEventListener("change", () => {
    state.demoMode = demoToggleEl.checked;
    saveState();
  });
}

// 오른쪽 패널 토글을 처리한다.
function setInsightsHidden(hidden) {
  state.insightsHidden = hidden;
  if (appEl) appEl.classList.toggle("is-insights-hidden", hidden);
  if (insightsEl) insightsEl.style.display = hidden ? "none" : "flex";
  if (insightsToggleEl) {
    insightsToggleEl.textContent = hidden ? "<" : ">";
    insightsToggleEl.setAttribute("aria-pressed", String(hidden));
  }
}

if (insightsToggleEl) {
  insightsToggleEl.addEventListener("click", () => {
    setInsightsHidden(!state.insightsHidden);
  });
}

// 대화를 초기화한다.
function resetConversation() {
  setTyping(false);
  state.sessionId = createSessionId();
  state.messages = [];
  saveState();
  updateSessionUi();
  renderMessages();
}

if (clearButtonEl) {
  clearButtonEl.addEventListener("click", () => resetConversation());
}
if (newChatButtonEl) {
  newChatButtonEl.addEventListener("click", () => resetConversation());
}

// 초기 로딩을 수행한다.
loadState();
updateSessionUi();
renderMessages();
resizeInput();
setInsightsHidden(state.insightsHidden);
