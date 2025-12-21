const paletteList = document.getElementById("palette-list");
const canvas = document.getElementById("canvas");
const connections = document.getElementById("connections");
const flowJson = document.getElementById("flow-json");
const exportButton = document.getElementById("export-json");
const copyButton = document.getElementById("copy-json");
const loadSampleButton = document.getElementById("load-sample");
const clearButton = document.getElementById("clear-canvas");

const NODE_TYPES = [
  {
    key: "user",
    title: "사용자 요청",
    desc: "자연어로 공정 데이터를 요청하거나 질문",
    color: "#2f7cff",
  },
  {
    key: "triage",
    title: "요청 분류",
    desc: "DB 조회 vs 시뮬레이션을 판단",
    color: "#ff6b35",
  },
  {
    key: "db_agent",
    title: "DB 에이전트",
    desc: "공정 데이터 쿼리 수행",
    color: "#264653",
  },
  {
    key: "sim_agent",
    title: "시뮬레이션 에이전트",
    desc: "입력 수집 및 시뮬레이션 실행",
    color: "#2a9d8f",
  },
  {
    key: "tool_db",
    title: "DB 함수 도구",
    desc: "DB 접근 및 이벤트 트리거",
    color: "#7b5cff",
  },
  {
    key: "tool_sim",
    title: "시뮬레이션 함수 도구",
    desc: "시뮬레이션 호출 및 이벤트 트리거",
    color: "#f4a261",
  },
  {
    key: "ui",
    title: "프론트 트리거",
    desc: "웹 UI로 이벤트 전달",
    color: "#8d99ae",
  },
];

const NODE_TYPE_MAP = Object.fromEntries(
  NODE_TYPES.map((type) => [type.key, type])
);

let nodeCounter = 1;
const nodes = new Map();
const edges = [];
let pendingConnection = null;

function buildPalette() {
  NODE_TYPES.forEach((type) => {
    const item = document.createElement("div");
    item.className = "palette-item";
    item.innerHTML = `<strong>${type.title}</strong><span>${type.desc}</span>`;
    item.addEventListener("click", () => {
      const rect = canvas.getBoundingClientRect();
      const x = rect.width / 2 - 80 + Math.random() * 80;
      const y = rect.height / 2 - 60 + Math.random() * 80;
      createNode(type.key, x, y);
    });
    paletteList.appendChild(item);
  });
}

function createNode(typeKey, x, y, label) {
  const type = NODE_TYPE_MAP[typeKey];
  const id = `node-${nodeCounter++}`;

  const nodeEl = document.createElement("div");
  nodeEl.className = "node";
  nodeEl.style.left = `${x}px`;
  nodeEl.style.top = `${y}px`;
  nodeEl.style.setProperty("--node-color", type.color);
  nodeEl.dataset.nodeId = id;

  const header = document.createElement("div");
  header.className = "node-header";
  header.textContent = label || type.title;

  const body = document.createElement("div");
  body.className = "node-body";
  body.textContent = type.desc;

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
    label: label || type.title,
    x,
    y,
    color: type.color,
    element: nodeEl,
    inPort,
    outPort,
  };

  nodes.set(id, node);

  enableDrag(node, header);
  enablePorts(node);
  updateFlowJson();
  updateConnections();

  return node;
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
  const payload = {
    updated_at: new Date().toISOString(),
    nodes: Array.from(nodes.values()).map((node) => ({
      id: node.id,
      type: node.type,
      label: node.label,
      position: { x: Math.round(node.x), y: Math.round(node.y) },
    })),
    edges: edges.map((edge) => ({ from: edge.from, to: edge.to })),
  };

  flowJson.value = JSON.stringify(payload, null, 2);
}

function resetCanvas() {
  pendingConnection = null;
  clearPortHighlights();
  nodes.forEach((node) => node.element.remove());
  nodes.clear();
  edges.splice(0, edges.length);
  connections.innerHTML = "";
  initMarkers();
  updateFlowJson();
}

function loadSample() {
  resetCanvas();
  const user = createNode("user", 80, 120);
  const triage = createNode("triage", 280, 120);
  const dbAgent = createNode("db_agent", 500, 40);
  const simAgent = createNode("sim_agent", 500, 200);
  const dbTool = createNode("tool_db", 720, 40, "DB 함수 도구");
  const simTool = createNode("tool_sim", 720, 200, "시뮬레이션 함수 도구");
  const uiTrigger = createNode("ui", 940, 120, "프론트 UI 트리거");

  createEdge(user.id, triage.id);
  createEdge(triage.id, dbAgent.id);
  createEdge(triage.id, simAgent.id);
  createEdge(dbAgent.id, dbTool.id);
  createEdge(simAgent.id, simTool.id);
  createEdge(dbTool.id, uiTrigger.id);
  createEdge(simTool.id, uiTrigger.id);
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
});

initMarkers();
buildPalette();
updateFlowJson();
