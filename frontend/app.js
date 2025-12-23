const messages = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const statusDot = document.getElementById("ws-status");
const statusLabel = document.getElementById("ws-label");
const sessionEl = document.getElementById("session-id");
const lotResultEl = document.getElementById("lot-result");
const simResultEl = document.getElementById("sim-result");
const eventLogEl = document.getElementById("event-log");
const workflowNameEl = document.getElementById("workflow-name");
const workflowUpdatedEl = document.getElementById("workflow-updated");
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
const frontendCard = document.getElementById("frontend-card");
const eventLogCard = document.getElementById("event-log-card");
const eventEmptyEl = document.getElementById("event-empty");
const simTemperatureInput = document.getElementById("sim-temperature");
const simVoltageInput = document.getElementById("sim-voltage");
const simSizeInput = document.getElementById("sim-size");
const simCapacityInput = document.getElementById("sim-capacity");
const simProductionSelect = document.getElementById("sim-production");
const simApplyButton = document.getElementById("sim-apply");
const simRunButton = document.getElementById("sim-run");
const simFormStatus = document.getElementById("sim-form-status");

const sessionId = crypto.randomUUID();
sessionEl.textContent = sessionId.slice(0, 8);
let lastLotId = "";
let isComposing = false;

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  messages.appendChild(message);
  messages.scrollTop = messages.scrollHeight;
}

function addEventLog(label, detail) {
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
  frontendCard,
].filter(Boolean);
let isLogOpen = false;

function updateEventEmpty() {
  if (!eventEmptyEl) {
    return;
  }
  const hasVisible = streamCards.some(
    (card) => card && !card.classList.contains("hidden")
  );
  eventEmptyEl.classList.toggle("hidden", hasVisible || isLogOpen);
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
  toggleLogButton.textContent = isLogOpen ? "로그 닫기" : "활동 로그";
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
    setSimStatus("입력 완료: 시뮬레이션 실행 가능");
  }
}

function collectSimFormParams() {
  const temperature = simTemperatureInput?.value.trim();
  const voltage = simVoltageInput?.value.trim();
  const size = simSizeInput?.value.trim();
  const capacity = simCapacityInput?.value.trim();
  const productionMode = simProductionSelect?.value || "";
  return {
    temperature: temperature ? Number(temperature) : null,
    voltage: voltage ? Number(voltage) : null,
    size: size ? Number(size) : null,
    capacity: capacity ? Number(capacity) : null,
    production_mode: productionMode || null,
  };
}

async function sendSimulationParams({ run } = { run: false }) {
  if (!simFormStatus) {
    return;
  }
  setSimStatus("입력값 전송 중...");
  const params = collectSimFormParams();
  const payload = run
    ? { session_id: sessionId }
    : { session_id: sessionId, ...params };
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
      setSimStatus("시뮬레이션 요청 실패", true);
      return;
    }
    const data = await response.json();
    if (data.missing && data.missing.length) {
      renderSimulationForm({ params: data.params || params, missing: data.missing });
      return;
    }
    setSimStatus("시뮬레이션 실행 완료");
    updateMissingInputs([]);
  } catch (error) {
    setSimStatus("시뮬레이션 요청 실패: 네트워크 오류", true);
  }
}

function renderLotResult(payload) {
  if (!payload || !payload.rows || payload.rows.length === 0) {
    lotResultEl.textContent = "조회 결과가 없습니다.";
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
    count.textContent = `추가 ${payload.rows.length - 1}건 더 있음`;
    lotResultEl.appendChild(count);
  }
}

function renderSimResult(payload) {
  if (!payload || !payload.result) {
    simResultEl.textContent = "시뮬레이션 데이터가 없습니다.";
    return;
  }

  showOnlyStreamCards([simResultCard]);

  const params = payload.params || {};
  const result = payload.result || {};
  const riskMap = {
    low: "낮음",
    medium: "중간",
    high: "높음",
    낮음: "낮음",
    중간: "중간",
    높음: "높음",
  };
  const riskLabel = riskMap[result.risk_band] || result.risk_band || "-";
  const riskClass =
    riskLabel === "높음"
      ? "risk-high"
      : riskLabel === "중간"
        ? "risk-medium"
        : "risk-low";

  const kpiRow = document.createElement("div");
  kpiRow.className = "kpi-row";
  kpiRow.appendChild(
    createKpiCard("Yield", `${result.predicted_yield ?? "-"}%`, "accent")
  );
  kpiRow.appendChild(createKpiCard("Risk", riskLabel, riskClass));
  kpiRow.appendChild(
    createKpiCard(
      "Throughput",
      `${result.estimated_throughput ?? "-"} u/h`
    )
  );
  if (lastLotId) {
    kpiRow.appendChild(createKpiCard("LOT", lastLotId, "ghost"));
  }

  simResultEl.innerHTML = "";
  simResultEl.appendChild(kpiRow);

  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  metrics.innerHTML = `
    <div class="metric">
      <span>예측 수율</span>
      <strong>${result.predicted_yield ?? "-"}%</strong>
    </div>
    <div class="metric">
      <span>리스크 구간</span>
      <strong>${riskLabel}</strong>
    </div>
    <div class="metric">
      <span>처리량</span>
      <strong>${result.estimated_throughput ?? "-"} u/h</strong>
    </div>
    <div class="metric">
      <span>메모</span>
      <strong>${result.notes || "-"}</strong>
    </div>
  `;
  const paramBlock = document.createElement("div");
  paramBlock.className = "metric";
  paramBlock.style.marginTop = "12px";
  const productionLabel =
    params.production_mode === "mass"
      ? "양산"
      : params.production_mode === "dev"
        ? "개발"
        : params.production_mode || "-";
  paramBlock.innerHTML = `
    <span>입력값</span>
    <strong>
      T ${params.temperature}, V ${params.voltage}, S ${params.size}, C ${params.capacity}, M ${productionLabel}
    </strong>
  `;

  simResultEl.appendChild(metrics);
  simResultEl.appendChild(paramBlock);
}

function handleEvent(event) {
  if (!event || !event.type) {
    return;
  }

  if (event.type === "lot_result" || event.type === "db_result") {
    renderLotResult(event.payload);
    addEventLog("LOT", `행 수: ${event.payload.rows.length}`);
    if (event.payload.lot_id) {
      setLotStatus(`${event.payload.lot_id} 조회 완료`);
    }
  }

  if (event.type === "simulation_form") {
    renderSimulationForm(event.payload);
  }

  if (event.type === "simulation_result") {
    renderSimResult(event.payload);
    const yieldValue = event.payload?.result?.predicted_yield;
    addEventLog("SIM", `수율: ${yieldValue ?? "-"}%`);
  }

  if (event.type === "frontend_trigger") {
    const message = event.payload?.message || "프론트 트리거가 수신되었습니다.";
    if (frontendTriggerEl) {
      frontendTriggerEl.textContent = message;
    }
    showOnlyStreamCards([frontendCard]);
    addEventLog("UI", message);
    if (message.toLowerCase().includes("lot")) {
      setLotStatus(message, true);
    }
    if (message.includes("시뮬레이션")) {
      setSimStatus(message, true);
    }
  }
}

const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
const wsUrl = `${wsProtocol}://${window.location.host}/ws`;
const socket = new WebSocket(wsUrl);

socket.addEventListener("open", () => {
  statusDot.classList.add("live");
  statusLabel.textContent = "실시간";
  addMessage("system", "WebSocket 연결 완료. 이벤트 스트림이 활성화되었습니다.");
});

socket.addEventListener("close", () => {
  statusDot.classList.remove("live");
  statusLabel.textContent = "연결 끊김";
  addMessage("system", "WebSocket 연결이 종료되었습니다.");
});

socket.addEventListener("message", (event) => {
  try {
    const data = JSON.parse(event.data);
    handleEvent(data);
  } catch (error) {
    addEventLog("WS", "JSON이 아닌 메시지가 수신되었습니다.");
  }
});

setInterval(() => {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send("ping");
  }
}, 20000);

async function sendChatMessage(message) {
  addMessage("user", message);
  input.value = "";
  input.style.height = "auto";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });

    if (!response.ok) {
      addMessage("assistant", "서버 오류입니다. 잠시 후 다시 시도해주세요.");
      return;
    }

    const data = await response.json();
    addMessage("assistant", data.assistant_message || "(응답 없음)");
  } catch (error) {
    addMessage("assistant", "네트워크 오류입니다. 백엔드 서버를 확인해주세요.");
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
    setLotStatus("LOT ID를 입력해주세요.", true);
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
    sendChatMessage(`${lotId} 예측 시뮬레이션 해줘.`);
  } else if (action === "search_sim") {
    sendChatMessage(`${lotId} 정보 조회하고 시뮬레이션도 해줘.`);
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
    setLotStatus("LOT ID를 입력하세요.");
  });
}
if (lotIdInput) {
  lotIdInput.addEventListener("input", () => {
    setLotStatus("LOT ID를 입력하세요.");
  });
}

if (toggleLogButton) {
  toggleLogButton.addEventListener("click", toggleEventLog);
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
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) {
        addMessage("assistant", "테스트 호출이 실패했습니다.");
        return;
      }

      addMessage("assistant", "테스트 트리거 전송 완료. 이벤트 패널을 확인하세요.");
    } catch (error) {
      addMessage("assistant", "테스트 요청 중 오류가 발생했습니다.");
    }
  });
}

setLotStatus("LOT ID를 입력하세요.");
setSimStatus("시뮬레이션 입력 대기");
updateEventEmpty();

addMessage(
  "assistant",
  "LOT 정보 조회 또는 LOT 기반 예측 시뮬레이션을 요청하세요. 요청을 올바른 에이전트로 라우팅합니다."
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

loadWorkflowMeta();
