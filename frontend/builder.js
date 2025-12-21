const paletteList = document.getElementById("palette-list");
const canvas = document.getElementById("canvas");
const connections = document.getElementById("connections");
const flowJson = document.getElementById("flow-json");
const exportButton = document.getElementById("export-json");
const copyButton = document.getElementById("copy-json");
const loadSampleButton = document.getElementById("load-sample");
const clearButton = document.getElementById("clear-canvas");
const saveWorkflowButton = document.getElementById("save-workflow");
const loadWorkflowButton = document.getElementById("load-workflow");
const workflowNameInput = document.getElementById("workflow-name");
const workflowDescInput = document.getElementById("workflow-desc");
const workflowStatus = document.getElementById("workflow-status");
const validateWorkflowButton = document.getElementById("validate-workflow");
const previewWorkflowButton = document.getElementById("preview-workflow");
const previewMessageInput = document.getElementById("preview-message");
const previewResult = document.getElementById("preview-result");

const nodeEmpty = document.getElementById("node-empty");
const nodeForm = document.getElementById("node-form");
const nodeIdEl = document.getElementById("node-id");
const nodeLabelInput = document.getElementById("node-label");
const nodeTypeInput = document.getElementById("node-type");
const nodeSubtypeSelect = document.getElementById("node-subtype");
const nodeExecutionField = document.getElementById("node-execution-field");
const nodeExecutionSelect = document.getElementById("node-execution");
const nodeKeywordsInput = document.getElementById("node-keywords");
const nodeInputFormat = document.getElementById("node-input-format");
const nodeOutputFormat = document.getElementById("node-output-format");
const nodeDeleteButton = document.getElementById("node-delete");

const NODE_LIBRARY = {
  user: {
    title: "사용자 요청",
    desc: "사용자 입력을 받는 시작 노드",
    color: "#2f7cff",
    subtypes: [{ key: "request", label: "요청" }],
  },
  agent: {
    title: "에이전트",
    desc: "오케스트레이터 또는 세부 에이전트",
    color: "#ff6b35",
    subtypes: [
      { key: "orchestrator", label: "오케스트레이터" },
      { key: "db_agent", label: "DB 에이전트" },
      { key: "simulation_agent", label: "시뮬레이션 에이전트" },
    ],
  },
  function: {
    title: "함수",
    desc: "DB/API 호출 및 프론트 트리거",
    color: "#2a9d8f",
    subtypes: [
      { key: "db_function", label: "DB 함수" },
      { key: "api_function", label: "API 함수" },
      { key: "frontend_trigger", label: "Frontend Trigger" },
    ],
  },
};

let nodeCounter = 1;
const nodes = new Map();
const edges = [];
let pendingConnection = null;
let selectedNodeId = null;
let isDirty = false;
let isLoading = false;
let previewedNodeIds = new Set();

function markDirty() {
  if (isLoading) {
    return;
  }
  clearPreviewHighlight();
  if (isDirty) {
    return;
  }
  isDirty = true;
  if (workflowStatus) {
    workflowStatus.textContent = "저장 상태: 변경됨";
  }
}

function buildPalette() {
  Object.keys(NODE_LIBRARY).forEach((typeKey) => {
    const type = NODE_LIBRARY[typeKey];
    const item = document.createElement("div");
    item.className = "palette-item";
    item.innerHTML = `<strong>${type.title}</strong><span>${type.desc}</span>`;
    item.addEventListener("click", () => {
      const rect = canvas.getBoundingClientRect();
      const x = rect.width / 2 - 80 + Math.random() * 80;
      const y = rect.height / 2 - 60 + Math.random() * 80;
      createNode(typeKey, x, y);
    });
    paletteList.appendChild(item);
  });
}

function createNode(typeKey, x, y, options = {}) {
  const type = NODE_LIBRARY[typeKey];
  if (!type) {
    return null;
  }

  const id = options.id || `node-${nodeCounter++}`;
  const subtype = options.subtype || type.subtypes[0].key;
  const label = options.label || type.title;
  const executionMode =
    options.execution_mode || (typeKey === "agent" ? "handoff" : "");

  const nodeEl = document.createElement("div");
  nodeEl.className = "node";
  nodeEl.style.left = `${x}px`;
  nodeEl.style.top = `${y}px`;
  nodeEl.style.setProperty("--node-color", type.color);
  nodeEl.dataset.nodeId = id;

  const header = document.createElement("div");
  header.className = "node-header";

  const body = document.createElement("div");
  body.className = "node-body";

  const ports = document.createElement("div");
  ports.className = "node-ports";

  const inPort = document.createElement("div");
  inPort.className = "port";
  inPort.dataset.port = "in";

  const outPort = document.createElement("div");
  outPort.className = "port";
  outPort.dataset.port = "out";

  ports.appendChild(inPort);
  ports.appendChild(outPort);

  nodeEl.appendChild(header);
  nodeEl.appendChild(body);
  nodeEl.appendChild(ports);
  canvas.appendChild(nodeEl);

  const node = {
    id,
    type: typeKey,
    subtype,
    execution_mode: executionMode,
    label,
    keywords: options.keywords || [],
    input_format: options.input_format || "",
    output_format: options.output_format || "",
    x,
    y,
    color: type.color,
    element: nodeEl,
    header,
    body,
    inPort,
    outPort,
  };

  nodes.set(id, node);

  updateNodeVisual(node);
  enableDrag(node, header);
  enablePorts(node);

  nodeEl.addEventListener("click", (event) => {
    event.stopPropagation();
    selectNode(node);
  });

  updateFlowJson();
  updateConnections();
  selectNode(node);
  markDirty();

  return node;
}

function updateNodeVisual(node) {
  const subtypeLabel = getSubtypeLabel(node.type, node.subtype);
  const keywordText = node.keywords.length
    ? `키워드: ${node.keywords.join(", ")}`
    : "키워드 없음";
  const modeText =
    node.type === "agent" ? `모드: ${node.execution_mode || "handoff"}` : "";
  const metaText = [keywordText, modeText].filter(Boolean).join(" · ");
  node.header.textContent = node.label;
  node.body.innerHTML = `<div>${subtypeLabel}</div><div class="node-meta">${metaText}</div>`;
}

function enableDrag(node, handle) {
  handle.addEventListener("mousedown", (event) => {
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = node.element.offsetLeft;
    const startTop = node.element.offsetTop;

    function onMove(moveEvent) {
      const nextLeft = startLeft + (moveEvent.clientX - startX);
      const nextTop = startTop + (moveEvent.clientY - startY);
      node.element.style.left = `${nextLeft}px`;
      node.element.style.top = `${nextTop}px`;
      node.x = nextLeft;
      node.y = nextTop;
      updateConnections();
      updateFlowJson();
    }

    function onUp() {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

function enablePorts(node) {
  node.outPort.addEventListener("click", (event) => {
    event.stopPropagation();
    clearPortHighlights();
    node.outPort.classList.add("active");
    pendingConnection = { from: node.id };
  });

  node.inPort.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!pendingConnection) {
      return;
    }
    if (pendingConnection.from === node.id) {
      clearPortHighlights();
      pendingConnection = null;
      return;
    }
    createEdge(pendingConnection.from, node.id);
    clearPortHighlights();
    pendingConnection = null;
  });
}

function clearPortHighlights() {
  nodes.forEach((node) => {
    node.outPort.classList.remove("active");
  });
}

function clearPreviewHighlight() {
  previewedNodeIds.forEach((nodeId) => {
    const node = nodes.get(nodeId);
    if (node) {
      node.element.classList.remove("preview");
    }
  });
  previewedNodeIds = new Set();
}

function applyPreviewHighlight(nodeIds = []) {
  clearPreviewHighlight();
  nodeIds.forEach((nodeId) => {
    const node = nodes.get(nodeId);
    if (!node) {
      return;
    }
    node.element.classList.add("preview");
    previewedNodeIds.add(nodeId);
  });
}

function createEdge(fromId, toId) {
  const fromNode = nodes.get(fromId);
  const toNode = nodes.get(toId);
  if (!fromNode || !toNode) {
    return;
  }

  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("stroke", fromNode.color);
  path.setAttribute("stroke-width", "2");
  path.setAttribute("fill", "none");
  path.setAttribute("marker-end", "url(#arrow)");
  connections.appendChild(path);

  edges.push({
    id: `edge-${edges.length + 1}`,
    from: fromId,
    to: toId,
    element: path,
  });

  updateConnections();
  updateFlowJson();
  markDirty();
}

function updateConnections() {
  edges.forEach((edge) => {
    const fromNode = nodes.get(edge.from);
    const toNode = nodes.get(edge.to);
    if (!fromNode || !toNode) {
      return;
    }

    const start = getPortCenter(fromNode.outPort);
    const end = getPortCenter(toNode.inPort);
    const offset = Math.max(60, Math.abs(end.x - start.x) * 0.4);
    const path = `M ${start.x} ${start.y} C ${start.x + offset} ${start.y}, ${
      end.x - offset
    } ${end.y}, ${end.x} ${end.y}`;
    edge.element.setAttribute("d", path);
  });
}

function getPortCenter(portEl) {
  const portRect = portEl.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  return {
    x: portRect.left - canvasRect.left + portRect.width / 2,
    y: portRect.top - canvasRect.top + portRect.height / 2,
  };
}

function updateFlowJson() {
  const payload = buildWorkflowPayload();
  flowJson.value = JSON.stringify(payload, null, 2);
}

function buildWorkflowPayload() {
  return {
    meta: {
      name: workflowNameInput.value.trim() || "워크플로우",
      description: workflowDescInput.value.trim(),
      updated_at: new Date().toISOString(),
    },
    nodes: Array.from(nodes.values()).map((node) => ({
      id: node.id,
      type: node.type,
      subtype: node.subtype,
      execution_mode: node.type === "agent" ? node.execution_mode : "",
      label: node.label,
      keywords: node.keywords,
      input_format: node.input_format,
      output_format: node.output_format,
      position: { x: Math.round(node.x), y: Math.round(node.y) },
    })),
    edges: edges.map((edge) => ({ from: edge.from, to: edge.to })),
  };
}

function resetCanvas() {
  pendingConnection = null;
  clearPortHighlights();
  clearSelection();
  clearPreviewHighlight();
  nodes.forEach((node) => node.element.remove());
  nodes.clear();
  edges.splice(0, edges.length);
  connections.innerHTML = "";
  initMarkers();
  updateFlowJson();
  markDirty();
}

function loadSample() {
  renderWorkflow(defaultWorkflow());
}

function renderWorkflow(workflow) {
  isLoading = true;
  resetCanvas();
  workflowNameInput.value = workflow.meta?.name || "워크플로우";
  workflowDescInput.value = workflow.meta?.description || "";

  const idNumbers = [];
  (workflow.nodes || []).forEach((node) => {
    const position = node.position || { x: 80, y: 80 };
    const created = createNode(node.type, position.x, position.y, {
      id: node.id,
      subtype: node.subtype,
      execution_mode: node.execution_mode,
      label: node.label,
      keywords: node.keywords || [],
      input_format: node.input_format || "",
      output_format: node.output_format || "",
    });

    const match = String(node.id || "").match(/node-(\d+)/);
    if (match) {
      idNumbers.push(Number(match[1]));
    }

    if (created) {
      updateNodeVisual(created);
    }
  });

  (workflow.edges || []).forEach((edge) => {
    createEdge(edge.from, edge.to);
  });

  if (idNumbers.length) {
    nodeCounter = Math.max(...idNumbers) + 1;
  } else {
    nodeCounter = nodes.size + 1;
  }

  updateFlowJson();
  updateConnections();
  isLoading = false;
  isDirty = false;
  if (workflowStatus) {
    workflowStatus.textContent = "저장 상태: 불러옴";
  }
  if (previewResult) {
    previewResult.textContent = "검증/미리보기 결과가 여기에 표시됩니다.";
  }
}

function initMarkers() {
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", "arrow");
  marker.setAttribute("markerWidth", "10");
  marker.setAttribute("markerHeight", "10");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "3");
  marker.setAttribute("orient", "auto");

  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", "M0,0 L9,3 L0,6 Z");
  path.setAttribute("fill", "#6b6b6b");

  marker.appendChild(path);
  defs.appendChild(marker);
  connections.appendChild(defs);
}

function selectNode(node) {
  if (!node) {
    return;
  }
  clearSelection();
  selectedNodeId = node.id;
  node.element.classList.add("selected");
  nodeEmpty.classList.add("hidden");
  nodeForm.classList.remove("hidden");

  nodeIdEl.textContent = node.id;
  nodeLabelInput.value = node.label;
  nodeTypeInput.value = NODE_LIBRARY[node.type].title;
  populateSubtypeOptions(node.type, node.subtype);
  if (node.type === "agent") {
    nodeExecutionField.classList.remove("hidden");
    nodeExecutionSelect.value = node.execution_mode || "handoff";
  } else {
    nodeExecutionField.classList.add("hidden");
  }
  nodeKeywordsInput.value = node.keywords.join(", ");
  nodeInputFormat.value = node.input_format;
  nodeOutputFormat.value = node.output_format;
}

function clearSelection() {
  if (selectedNodeId && nodes.has(selectedNodeId)) {
    nodes.get(selectedNodeId).element.classList.remove("selected");
  }
  selectedNodeId = null;
  nodeEmpty.classList.remove("hidden");
  nodeForm.classList.add("hidden");
}

function deleteSelectedNode() {
  if (!selectedNodeId || !nodes.has(selectedNodeId)) {
    return;
  }
  pendingConnection = null;
  clearPortHighlights();
  const nodeId = selectedNodeId;
  const node = nodes.get(nodeId);
  if (!node) {
    return;
  }

  for (let i = edges.length - 1; i >= 0; i -= 1) {
    const edge = edges[i];
    if (edge.from === nodeId || edge.to === nodeId) {
      edge.element.remove();
      edges.splice(i, 1);
    }
  }

  node.element.remove();
  nodes.delete(nodeId);
  clearSelection();
  updateFlowJson();
  updateConnections();
  markDirty();
}

function populateSubtypeOptions(typeKey, selected) {
  nodeSubtypeSelect.innerHTML = "";
  NODE_LIBRARY[typeKey].subtypes.forEach((subtype) => {
    const option = document.createElement("option");
    option.value = subtype.key;
    option.textContent = subtype.label;
    if (subtype.key === selected) {
      option.selected = true;
    }
    nodeSubtypeSelect.appendChild(option);
  });
}

function updateSelectedNode() {
  if (!selectedNodeId || !nodes.has(selectedNodeId)) {
    return;
  }
  const node = nodes.get(selectedNodeId);
  node.label = nodeLabelInput.value.trim() || node.label;
  node.subtype = nodeSubtypeSelect.value;
  if (node.type === "agent") {
    node.execution_mode = nodeExecutionSelect.value;
  }
  node.keywords = parseKeywords(nodeKeywordsInput.value);
  node.input_format = nodeInputFormat.value;
  node.output_format = nodeOutputFormat.value;
  updateNodeVisual(node);
  updateFlowJson();
  markDirty();
}

function parseKeywords(value) {
  return value
    .split(/[,\n]/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);
}

function getSubtypeLabel(typeKey, subtypeKey) {
  const subtype = NODE_LIBRARY[typeKey].subtypes.find(
    (item) => item.key === subtypeKey
  );
  return subtype ? subtype.label : subtypeKey;
}

function formatNodeTitle(node) {
  if (!node) {
    return "알 수 없음";
  }
  const label = node.label || node.id || "노드";
  const typeText = node.type ? `${node.type}/${node.subtype || "-"}` : "노드";
  const modeText =
    node.type === "agent" && node.execution_mode
      ? `, ${node.execution_mode}`
      : "";
  return `${label} (${typeText}${modeText})`;
}

async function validateWorkflow() {
  if (!previewResult) {
    return;
  }
  previewResult.textContent = "검증 중...";
  const payload = buildWorkflowPayload();
  try {
    const response = await fetch("/api/workflow/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow: payload }),
    });
    const data = await response.json();
    if (!data.valid) {
      const errorText = (data.errors || []).join("\n- ");
      previewResult.textContent = `검증 실패:\n- ${errorText}`;
      return;
    }
    previewResult.textContent = "검증 통과: 저장 가능한 상태입니다.";
  } catch (error) {
    previewResult.textContent = "검증 실패: 네트워크 오류";
  }
}

async function previewWorkflow() {
  if (!previewResult || !previewMessageInput) {
    return;
  }
  const message = previewMessageInput.value.trim();
  if (!message) {
    previewResult.textContent = "테스트 메시지를 입력해주세요.";
    return;
  }
  previewResult.textContent = "라우팅 미리보기 중...";
  const payload = buildWorkflowPayload();
  try {
    const response = await fetch("/api/workflow/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow: payload, message }),
    });
    if (!response.ok) {
      const detail = await response.json();
      previewResult.textContent = `미리보기 실패: ${
        detail?.detail?.errors?.join(", ") || "오류"
      }`;
      clearPreviewHighlight();
      return;
    }
    const data = await response.json();
    const pathNodes = data.path_nodes || [];
    const pathLabel = pathNodes.length
      ? pathNodes.map(formatNodeTitle).join(" → ")
      : "경로 없음";
    previewResult.textContent = `선택 노드: ${formatNodeTitle(
      data.selected_node
    )}\n경로: ${pathLabel}`;
    applyPreviewHighlight(data.path || []);
  } catch (error) {
    previewResult.textContent = "미리보기 실패: 네트워크 오류";
    clearPreviewHighlight();
  }
}

async function saveWorkflow() {
  if (!saveWorkflowButton) {
    return;
  }
  const payload = buildWorkflowPayload();
  workflowStatus.textContent = "저장 중...";
  try {
    const response = await fetch("/api/workflow", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow: payload }),
    });
    if (!response.ok) {
      const detail = await response.json();
      workflowStatus.textContent = `저장 실패: ${detail?.detail?.errors?.join(", ") || "오류"}`;
      return;
    }
    const data = await response.json();
    workflowStatus.textContent = `저장 완료: ${data.meta?.updated_at || ""}`;
    isDirty = false;
  } catch (error) {
    workflowStatus.textContent = "저장 실패: 네트워크 오류";
  }
}

async function loadWorkflow() {
  workflowStatus.textContent = "로드 중...";
  try {
    const response = await fetch("/api/workflow");
    if (!response.ok) {
      workflowStatus.textContent = "로드 실패";
      return;
    }
    const data = await response.json();
    renderWorkflow(data);
    workflowStatus.textContent = `로드 완료: ${data.meta?.updated_at || ""}`;
  } catch (error) {
    workflowStatus.textContent = "로드 실패: 네트워크 오류";
  }
}

function defaultWorkflow() {
  return {
    meta: {
      name: "공정 모니터링 기본 워크플로우",
      description: "오케스트레이터 → DB/시뮬레이션 분기 → 함수 실행 → 프론트 트리거",
      updated_at: new Date().toISOString(),
    },
    nodes: [
      {
        id: "node-user",
        type: "user",
        subtype: "request",
        label: "사용자 요청",
        keywords: [],
        input_format: "자연어 요청",
        output_format: "텍스트",
        position: { x: 80, y: 160 },
      },
      {
        id: "node-orchestrator",
        type: "agent",
        subtype: "orchestrator",
        execution_mode: "handoff",
        label: "오케스트레이터",
        keywords: ["공정", "데이터", "예측", "시뮬레이션"],
        input_format: "사용자 메시지",
        output_format: "경로 선택",
        position: { x: 300, y: 160 },
      },
      {
        id: "node-db-agent",
        type: "agent",
        subtype: "db_agent",
        execution_mode: "handoff",
        label: "DB 에이전트",
        keywords: ["공정", "라인", "상태", "데이터", "조회"],
        input_format: "라인/상태 필터",
        output_format: "공정 레코드",
        position: { x: 520, y: 60 },
      },
      {
        id: "node-sim-agent",
        type: "agent",
        subtype: "simulation_agent",
        execution_mode: "handoff",
        label: "시뮬레이션 에이전트",
        keywords: ["예측", "시뮬레이션", "what-if", "forecast"],
        input_format: "온도/전압/크기/용량",
        output_format: "예측 결과",
        position: { x: 520, y: 260 },
      },
      {
        id: "node-db-function",
        type: "function",
        subtype: "db_function",
        label: "DB 함수",
        keywords: [],
        input_format: "검색 키워드",
        output_format: "DB 결과",
        position: { x: 740, y: 60 },
      },
      {
        id: "node-api-function",
        type: "function",
        subtype: "api_function",
        label: "API 함수",
        keywords: [],
        input_format: "파라미터 JSON",
        output_format: "시뮬레이션 결과",
        position: { x: 740, y: 260 },
      },
      {
        id: "node-frontend-trigger",
        type: "function",
        subtype: "frontend_trigger",
        label: "프론트 트리거",
        keywords: [],
        input_format: "이벤트 메시지",
        output_format: "UI 표시",
        position: { x: 960, y: 160 },
      },
    ],
    edges: [
      { from: "node-user", to: "node-orchestrator" },
      { from: "node-orchestrator", to: "node-db-agent" },
      { from: "node-orchestrator", to: "node-sim-agent" },
      { from: "node-db-agent", to: "node-db-function" },
      { from: "node-db-function", to: "node-frontend-trigger" },
      { from: "node-sim-agent", to: "node-api-function" },
      { from: "node-api-function", to: "node-frontend-trigger" },
    ],
  };
}

exportButton.addEventListener("click", () => {
  updateFlowJson();
  flowJson.focus();
  flowJson.select();
});

copyButton.addEventListener("click", async () => {
  updateFlowJson();
  try {
    await navigator.clipboard.writeText(flowJson.value);
    copyButton.textContent = "복사됨";
    setTimeout(() => {
      copyButton.textContent = "복사";
    }, 1500);
  } catch (error) {
    copyButton.textContent = "복사 실패";
  }
});

loadSampleButton.addEventListener("click", loadSample);
clearButton.addEventListener("click", resetCanvas);

window.addEventListener("resize", updateConnections);
canvas.addEventListener("click", () => {
  pendingConnection = null;
  clearPortHighlights();
  clearSelection();
});

nodeLabelInput.addEventListener("input", updateSelectedNode);
nodeSubtypeSelect.addEventListener("change", updateSelectedNode);
nodeExecutionSelect.addEventListener("change", updateSelectedNode);
nodeKeywordsInput.addEventListener("input", updateSelectedNode);
nodeInputFormat.addEventListener("input", updateSelectedNode);
nodeOutputFormat.addEventListener("input", updateSelectedNode);
if (nodeDeleteButton) {
  nodeDeleteButton.addEventListener("click", deleteSelectedNode);
}

saveWorkflowButton.addEventListener("click", saveWorkflow);
loadWorkflowButton.addEventListener("click", loadWorkflow);
if (validateWorkflowButton) {
  validateWorkflowButton.addEventListener("click", validateWorkflow);
}
if (previewWorkflowButton) {
  previewWorkflowButton.addEventListener("click", previewWorkflow);
}
workflowNameInput.addEventListener("input", () => {
  updateFlowJson();
  markDirty();
});
workflowDescInput.addEventListener("input", () => {
  updateFlowJson();
  markDirty();
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Delete" && event.key !== "Backspace") {
    return;
  }
  const target = event.target;
  if (
    target &&
    (target.tagName === "INPUT" || target.tagName === "TEXTAREA")
  ) {
    return;
  }
  deleteSelectedNode();
});

initMarkers();
buildPalette();
loadWorkflow();
