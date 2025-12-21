const messages = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const statusDot = document.getElementById("ws-status");
const statusLabel = document.getElementById("ws-label");
const sessionEl = document.getElementById("session-id");
const dbResultEl = document.getElementById("db-result");
const simResultEl = document.getElementById("sim-result");
const eventLogEl = document.getElementById("event-log");
const workflowNameEl = document.getElementById("workflow-name");
const workflowUpdatedEl = document.getElementById("workflow-updated");
const frontendTriggerEl = document.getElementById("frontend-trigger");

const sessionId = crypto.randomUUID();
sessionEl.textContent = sessionId.slice(0, 8);

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

function renderDbResult(payload) {
  if (!payload || !payload.rows || payload.rows.length === 0) {
    dbResultEl.textContent = "조회 결과가 없습니다.";
    return;
  }

  const table = document.createElement("table");
  table.className = "table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>라인</th>
        <th>상태</th>
        <th>온도</th>
        <th>전압</th>
        <th>크기</th>
        <th>용량</th>
        <th>시간</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");
  payload.rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.line}</td>
      <td>${row.status}</td>
      <td>${row.temperature.toFixed(1)}</td>
      <td>${row.voltage.toFixed(2)}</td>
      <td>${row.size.toFixed(1)}</td>
      <td>${row.capacity.toFixed(2)}</td>
      <td>${row.timestamp.split("T")[1].replace("Z", "")}</td>
    `;
    tbody.appendChild(tr);
  });

  dbResultEl.innerHTML = "";
  dbResultEl.appendChild(table);
}

function renderSimResult(payload) {
  if (!payload || !payload.result) {
    simResultEl.textContent = "시뮬레이션 데이터가 없습니다.";
    return;
  }

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

  simResultEl.innerHTML = `
    <div class="metric-grid">
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
    </div>
    <div class="metric" style="margin-top: 12px;">
      <span>입력값</span>
      <strong>
        T ${params.temperature}, V ${params.voltage}, S ${params.size}, C ${params.capacity}
      </strong>
    </div>
  `;
}

function handleEvent(event) {
  if (!event || !event.type) {
    return;
  }

  if (event.type === "db_result") {
    renderDbResult(event.payload);
    addEventLog("DB", `행 수: ${event.payload.rows.length}`);
  }

  if (event.type === "simulation_result") {
    renderSimResult(event.payload);
    addEventLog("SIM", `수율: ${event.payload.result.predicted_yield}%`);
  }

  if (event.type === "frontend_trigger") {
    const message = event.payload?.message || "프론트 트리거가 수신되었습니다.";
    if (frontendTriggerEl) {
      frontendTriggerEl.textContent = message;
    }
    addEventLog("UI", message);
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }

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
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    const prompt = button.getAttribute("data-prompt");
    input.value = prompt;
    input.dispatchEvent(new Event("input"));
    form.dispatchEvent(new Event("submit"));
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

addMessage(
  "assistant",
  "공정 데이터 조회 또는 예측 시뮬레이션을 요청하세요. 요청을 올바른 에이전트로 라우팅합니다."
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
