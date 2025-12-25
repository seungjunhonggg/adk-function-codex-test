const paletteList = document.getElementById("palette-list");
const canvas = document.getElementById("canvas");
const canvasWrap = canvas ? canvas.closest(".canvas-wrap") : null;
const connections = document.getElementById("connections");
const flowJson = document.getElementById("flow-json");
const exportButton = document.getElementById("export-json");
const toggleAllNodesButton = document.getElementById("toggle-all-nodes");
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
const inspectorPanel = document.getElementById("inspector-panel");
const inspectorCloseButton = document.getElementById("inspector-close");
const inspectorTabButtons = Array.from(
  document.querySelectorAll(".inspector-tab")
);

const workflowPanel = document.querySelector('[data-panel="workflow"]');
const dbPanel = document.querySelector('[data-panel="db"]');
const nodePanel = document.querySelector('[data-panel="node"]');
const previewPanel = document.querySelector('[data-panel="preview"]');
const jsonPanel = document.querySelector('[data-panel="json"]');
const guidePanel = document.querySelector('[data-panel="guide"]');

const dbNameInput = document.getElementById("db-name");
const dbTypeSelect = document.getElementById("db-type");
const dbHostInput = document.getElementById("db-host");
const dbPortInput = document.getElementById("db-port");
const dbDatabaseInput = document.getElementById("db-database");
const dbUserInput = document.getElementById("db-user");
const dbPasswordInput = document.getElementById("db-password");
const dbConnectButton = document.getElementById("db-connect");
const dbRefreshButton = document.getElementById("db-refresh");
const dbStatus = document.getElementById("db-status");
const dbConnectionsList = document.getElementById("db-connections");

const nodeEmpty = document.getElementById("node-empty");
const nodeForm = document.getElementById("node-form");
const nodeIdEl = document.getElementById("node-id");
const nodeLabelInput = document.getElementById("node-label");
const nodeTypeInput = document.getElementById("node-type");
const nodeExecutionField = document.getElementById("node-execution-field");
const nodeExecutionSelect = document.getElementById("node-execution");
const nodeFields = document.getElementById("node-fields");
const nodeDeleteButton = document.getElementById("node-delete");

const NODE_GROUPS = [
  { key: "core", title: "Core", nodes: ["start", "agent", "end"] },
  {
    key: "tools",
    title: "Tools",
    nodes: ["guardrail", "mcp", "file_search", "function_tool"],
  },
  { key: "logic", title: "Logic", nodes: ["if_else", "while", "user_approval"] },
  { key: "data", title: "Data", nodes: ["state", "transform", "note"] },
];

const NODE_LIBRARY = {
  start: {
    title: "Start",
    desc: "워크플로우 시작점 · 입력 변수를 정의",
    color: "#2f7cff",
    icon: "S",
  },
  agent: {
    title: "Agent",
    desc: "LLM 실행 · 도구/핸드오프 구성",
    color: "#ff6b35",
    icon: "A",
  },
  guardrail: {
    title: "Guardrail",
    desc: "PII/Moderation/Jailbreak 체크",
    color: "#f4a261",
    icon: "G",
  },
  mcp: {
    title: "MCP",
    desc: "MCP 서버 연결 · 도구 제공",
    color: "#2a9d8f",
    icon: "M",
  },
  file_search: {
    title: "File Search",
    desc: "벡터 스토어 기반 검색",
    color: "#3a86ff",
    icon: "F",
  },
  function_tool: {
    title: "Function Tool",
    desc: "사용자 정의 함수 · DB 쿼리 자동 생성",
    color: "#10b981",
    icon: "T",
  },
  end: {
    title: "End",
    desc: "워크플로우 종료 · 출력 포맷",
    color: "#6b7280",
    icon: "E",
  },
  note: {
    title: "Note",
    desc: "메모/설명용 스티키 노트",
    color: "#b08968",
    icon: "N",
  },
  if_else: {
    title: "If / Else",
    desc: "조건 분기 (CEL)",
    color: "#f59e0b",
    icon: "IF",
  },
  while: {
    title: "While",
    desc: "조건 반복 (CEL)",
    color: "#ef4444",
    icon: "W",
  },
  user_approval: {
    title: "User Approval",
    desc: "사람 승인 단계 추가",
    color: "#db2777",
    icon: "UA",
  },
  transform: {
    title: "Transform",
    desc: "데이터 변환/매핑",
    color: "#0ea5e9",
    icon: "X",
  },
  state: {
    title: "State",
    desc: "전역 상태 업데이트",
    color: "#6b705c",
    icon: "ST",
  },
};

const NODE_PORTS = {
  start: { in: [], out: ["out"] },
  agent: { in: ["in"], out: ["out"] },
  guardrail: { in: ["in"], out: ["pass", "fail"] },
  mcp: { in: ["in"], out: ["out"] },
  file_search: { in: ["in"], out: ["out"] },
  function_tool: { in: ["in"], out: ["out"] },
  end: { in: ["in"], out: [] },
  note: { in: [], out: [] },
  if_else: { in: ["in"], out: ["true", "false"] },
  while: { in: ["in"], out: ["loop", "done"] },
  user_approval: { in: ["in"], out: ["approved", "rejected"] },
  transform: { in: ["in"], out: ["out"] },
  state: { in: ["in"], out: ["out"] },
};

const NODE_DEFAULTS = {
  start: {
    config: {
      input_variables: ["input_as_text"],
      state_variables: [],
    },
  },
  agent: {
    config: {
      agent_profile: "custom",
      instructions: "",
      include_chat_history: true,
      response_format: "text",
      verbosity: "medium",
      summary: "off",
      write_conversation_history: false,
    },
    keywords: [],
  },
  guardrail: {
    config: {
      input: "input_as_text",
      pii: true,
      moderation: true,
      jailbreak: true,
      hallucination: false,
      vector_store_id: "",
    },
  },
  mcp: {
    config: {
      toolset: "custom",
      server_name: "",
      server_url: "",
      auth_type: "none",
      auth_token: "",
      tool_names: [],
      message: "",
    },
  },
  file_search: {
    config: {
      vector_store_id: "",
      top_k: 3,
    },
  },
  function_tool: {
    config: {
      tool_type: "db_query",
      connection_id: "",
      schema: "",
      table: "",
      columns: [],
      filter_column: "",
      filter_operator: "ilike",
      limit: 50,
      tool_name: "",
      description: "",
    },
  },
  end: {
    config: {
      return_type: "text",
      schema: "",
    },
  },
  note: {
    config: {
      note: "",
    },
  },
  if_else: {
    config: {
      condition: "contains(input_as_text, \"예측\")",
    },
  },
  while: {
    config: {
      condition: "",
      max_iterations: 3,
    },
  },
  user_approval: {
    config: {
      approval_message: "승인 후 다음 단계로 진행합니다.",
      default_decision: "approved",
      auto_approve: false,
    },
  },
  transform: {
    config: {
      template: "{{input_as_text}}",
    },
  },
  state: {
    config: {
      assignments: "",
    },
  },
};

const NODE_FIELDS = {
  start: [
    {
      key: "input_variables",
      label: "Input Variables",
      type: "list",
      placeholder: "input_as_text",
    },
    {
      key: "state_variables",
      label: "State Variables",
      type: "list",
      placeholder: "conversation_history:text",
    },
  ],
  agent: [
    {
      key: "agent_profile",
      label: "Agent Profile",
      type: "select",
      options: [
        { value: "custom", label: "Custom" },
        { value: "orchestrator", label: "Orchestrator" },
        { value: "db_agent", label: "DB Agent" },
        { value: "simulation_agent", label: "Simulation Agent" },
      ],
    },
    {
      key: "instructions",
      label: "Instructions",
      type: "textarea",
      placeholder: "에이전트 역할과 제약을 설명하세요.",
    },
    {
      key: "include_chat_history",
      label: "Include Chat History",
      type: "checkbox",
    },
    { key: "model", label: "Model", type: "text", placeholder: "gpt-5" },
    {
      key: "reasoning",
      label: "Reasoning",
      type: "select",
      options: [
        { value: "minimum", label: "minimum" },
        { value: "low", label: "low" },
        { value: "medium", label: "medium" },
        { value: "high", label: "high" },
      ],
    },
    {
      key: "response_format",
      label: "Response Format",
      type: "select",
      options: [
        { value: "text", label: "text" },
        { value: "json", label: "json" },
        { value: "widgets", label: "widgets" },
      ],
    },
    {
      key: "verbosity",
      label: "Verbosity",
      type: "select",
      options: [
        { value: "low", label: "low" },
        { value: "medium", label: "medium" },
        { value: "high", label: "high" },
      ],
    },
    {
      key: "summary",
      label: "Summary",
      type: "select",
      options: [
        { value: "off", label: "off" },
        { value: "auto", label: "auto" },
        { value: "on", label: "on" },
      ],
    },
    {
      key: "write_conversation_history",
      label: "Write Conversation History",
      type: "checkbox",
    },
    {
      key: "keywords",
      label: "Routing Keywords",
      type: "list",
      placeholder: "공정, 예측, 라인",
      scope: "root",
    },
    {
      key: "input_format",
      label: "Input Schema",
      type: "textarea",
      placeholder: "예: { message: string }",
      scope: "root",
    },
    {
      key: "output_format",
      label: "Output Schema",
      type: "textarea",
      placeholder: "예: { answer: string }",
      scope: "root",
    },
  ],
  guardrail: [
    { key: "input", label: "Input Variable", type: "text" },
    { key: "pii", label: "PII", type: "checkbox" },
    { key: "moderation", label: "Moderation", type: "checkbox" },
    { key: "jailbreak", label: "Jailbreak", type: "checkbox" },
    { key: "hallucination", label: "Hallucination", type: "checkbox" },
    { key: "vector_store_id", label: "Vector Store Id", type: "text" },
  ],
  mcp: [
    {
      key: "toolset",
      label: "Toolset",
      type: "select",
      options: [
        { value: "custom", label: "Custom" },
        { value: "process_db", label: "Process DB" },
        { value: "simulation", label: "Simulation" },
        { value: "frontend_trigger", label: "Frontend Trigger" },
      ],
    },
    { key: "server_name", label: "Server Name", type: "text" },
    { key: "server_url", label: "Server URL", type: "text" },
    {
      key: "auth_type",
      label: "Auth Type",
      type: "select",
      options: [
        { value: "none", label: "No Auth" },
        { value: "api_key", label: "API Key" },
        { value: "headers", label: "Custom Headers" },
      ],
    },
    { key: "auth_token", label: "Auth Token", type: "text" },
    {
      key: "tool_names",
      label: "Tool Names",
      type: "list",
      placeholder: "예: search, create, update",
    },
    {
      key: "message",
      label: "Frontend Message",
      type: "textarea",
      placeholder: "Frontend Trigger용 메시지",
    },
    {
      key: "input_format",
      label: "Input Schema",
      type: "textarea",
      scope: "root",
    },
    {
      key: "output_format",
      label: "Output Schema",
      type: "textarea",
      scope: "root",
    },
  ],
  file_search: [
    { key: "vector_store_id", label: "Vector Store Id", type: "text" },
    { key: "top_k", label: "Top K", type: "number" },
    {
      key: "input_format",
      label: "Input Schema",
      type: "textarea",
      scope: "root",
    },
    {
      key: "output_format",
      label: "Output Schema",
      type: "textarea",
      scope: "root",
    },
  ],
  function_tool: [
    {
      key: "tool_type",
      label: "Tool Type",
      type: "select",
      options: [{ value: "db_query", label: "DB Query" }],
    },
    {
      key: "connection_id",
      label: "DB Connection",
      type: "select",
      options: getDbConnectionOptions,
      refreshOnChange: true,
    },
    {
      key: "schema",
      label: "Schema",
      type: "select",
      options: getSchemaOptions,
      refreshOnChange: true,
    },
    {
      key: "table",
      label: "Table",
      type: "select",
      options: getTableOptions,
      refreshOnChange: true,
    },
    {
      key: "columns",
      label: "Columns",
      type: "multiselect",
      options: getColumnOptions,
    },
    {
      key: "filter_column",
      label: "Filter Column",
      type: "select",
      options: getFilterColumnOptions,
    },
    {
      key: "filter_operator",
      label: "Filter Operator",
      type: "select",
      options: [
        { value: "ilike", label: "ilike" },
        { value: "like", label: "like" },
        { value: "=", label: "=" },
        { value: ">", label: ">" },
        { value: "<", label: "<" },
        { value: ">=", label: ">=" },
        { value: "<=", label: "<=" },
      ],
    },
    {
      key: "limit",
      label: "Limit",
      type: "number",
    },
    {
      key: "tool_name",
      label: "Tool Name",
      type: "text",
      placeholder: "db_query_tool",
    },
    {
      key: "description",
      label: "Description",
      type: "textarea",
      placeholder: "도구의 목적과 출력 형식을 설명하세요.",
    },
    {
      key: "input_format",
      label: "Input Schema",
      type: "textarea",
      scope: "root",
    },
    {
      key: "output_format",
      label: "Output Schema",
      type: "textarea",
      scope: "root",
    },
  ],
  end: [
    {
      key: "return_type",
      label: "Return Type",
      type: "select",
      options: [
        { value: "text", label: "text" },
        { value: "json", label: "json" },
      ],
    },
    {
      key: "schema",
      label: "JSON Schema",
      type: "textarea",
      placeholder: "{ \"type\": \"object\" }",
    },
  ],
  note: [{ key: "note", label: "Note", type: "textarea" }],
  if_else: [
    {
      key: "condition",
      label: "Condition (CEL)",
      type: "text",
      placeholder: "contains(input_as_text, \"예측\")",
    },
  ],
  while: [
    {
      key: "condition",
      label: "Condition (CEL)",
      type: "text",
    },
    { key: "max_iterations", label: "Max Iterations", type: "number" },
  ],
  user_approval: [
    {
      key: "approval_message",
      label: "Approval Message",
      type: "textarea",
    },
    {
      key: "default_decision",
      label: "Default Decision",
      type: "select",
      options: [
        { value: "approved", label: "approved" },
        { value: "rejected", label: "rejected" },
      ],
    },
    { key: "auto_approve", label: "Auto Approve", type: "checkbox" },
  ],
  transform: [
    {
      key: "template",
      label: "Template",
      type: "textarea",
      placeholder: "{{input_as_text}}",
    },
  ],
  state: [
    {
      key: "assignments",
      label: "Assignments (key=value)",
      type: "textarea",
      placeholder: "foo=bar",
    },
  ],
};

const LEGACY_TYPE_MAP = { user: "start", function: "mcp" };
const LEGACY_TOOLSET_MAP = {
  db_function: "process_db",
  api_function: "simulation",
  frontend_trigger: "frontend_trigger",
};

let nodeCounter = 1;
const nodes = new Map();
const edges = [];
let pendingConnection = null;
let selectedNodeId = null;
let isDirty = false;
let isLoading = false;
let previewedNodeIds = new Set();
let activeFieldDefs = [];
let fieldInputMap = new Map();
let currentScale = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panOriginX = 0;
let panOriginY = 0;
let panMoved = false;
let suppressCanvasClick = false;

function toRgba(color, alpha) {
  if (!color || typeof color !== "string") {
    return `rgba(0, 0, 0, ${alpha})`;
  }
  const trimmed = color.trim();
  if (!trimmed.startsWith("#")) {
    return trimmed;
  }
  let hex = trimmed.slice(1);
  if (hex.length === 3) {
    hex = hex
      .split("")
      .map((ch) => ch + ch)
      .join("");
  }
  if (hex.length !== 6) {
    return `rgba(0, 0, 0, ${alpha})`;
  }
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

let dbConnections = [];
const dbSchemas = {};

const inspectorPanels = {
  workflow: workflowPanel,
  db: dbPanel,
  node: nodePanel,
  preview: previewPanel,
  json: jsonPanel,
  guide: guidePanel,
};

let activeInspectorTab = "node";

function setInspectorOpen(open) {
  if (!inspectorPanel) {
    return;
  }
  inspectorPanel.classList.toggle("is-open", open);
}

function setActiveTab(tabKey) {
  if (!tabKey) {
    return;
  }
  activeInspectorTab = tabKey;
  inspectorTabButtons.forEach((button) => {
    const isActive = button.dataset.tab === tabKey;
    button.classList.toggle("active", isActive);
  });
  Object.keys(inspectorPanels).forEach((key) => {
    setPanelVisible(key, key === tabKey);
  });
}

function setTabEnabled(tabKey, enabled) {
  inspectorTabButtons.forEach((button) => {
    if (button.dataset.tab !== tabKey) {
      return;
    }
    button.disabled = !enabled;
    button.classList.toggle("disabled", !enabled);
  });
}

function setPanelVisible(panelKey, visible) {
  const panel = inspectorPanels[panelKey];
  if (!panel) {
    return;
  }
  panel.classList.toggle("hidden", !visible);
}

function updateInspectorVisibility(node) {
  if (node) {
    setInspectorOpen(true);
    setTabEnabled("db", node.type === "function_tool");
    setActiveTab("node");
  } else {
    setInspectorOpen(false);
    setTabEnabled("db", false);
    setActiveTab("node");
  }
}

function setDbStatus(message, isError = false) {
  if (!dbStatus) {
    return;
  }
  dbStatus.textContent = message;
  dbStatus.classList.toggle("error", isError);
}

function renderDbConnectionsList() {
  if (!dbConnectionsList) {
    return;
  }
  if (!dbConnections.length) {
    dbConnectionsList.textContent = "연결된 DB 없음";
    return;
  }
  const lines = dbConnections.map((conn) => {
    const name = conn.name || conn.id;
    const host = conn.host || "-";
    const port = conn.port ? `:${conn.port}` : "";
    const database = conn.database || "-";
    const user = conn.user ? `/${conn.user}` : "";
    return `${name} (${conn.id}) - ${host}${port}/${database}${user}`;
  });
  dbConnectionsList.textContent = lines.join("\n");
}

function refreshFunctionToolNodes() {
  nodes.forEach((node) => {
    if (node.type === "function_tool") {
      updateNodeVisual(node);
    }
  });
}

function getConnectionById(connectionId) {
  return dbConnections.find((conn) => conn.id === connectionId);
}

function getDbConnectionLabel(connectionId) {
  const conn = getConnectionById(connectionId);
  if (conn) {
    return conn.name || conn.id;
  }
  return connectionId || "";
}

function withPlaceholder(options, label) {
  if (!label) {
    return options;
  }
  return [{ value: "", label }, ...options];
}

function getDbSchema(connectionId) {
  return dbSchemas[connectionId] || null;
}

async function ensureDbSchema(connectionId) {
  if (!connectionId || dbSchemas[connectionId]) {
    return;
  }
  try {
    const response = await fetch(`/api/db/schema/${connectionId}`);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    dbSchemas[connectionId] = data.schema || {};
    if (selectedNodeId) {
      const node = nodes.get(selectedNodeId);
      if (node && node.type === "function_tool") {
        renderNodeFields(node);
      }
    }
  } catch (error) {
    // ignore background schema refresh
  }
}

function getDbConnectionOptions() {
  const options = dbConnections.map((conn) => {
    const label = `${conn.name || conn.id} (${conn.host}:${conn.port}/${conn.database})`;
    return { value: conn.id, label };
  });
  return withPlaceholder(options, "연결 선택");
}

function getSchemaOptions(node) {
  const connectionId = node.config?.connection_id;
  if (!connectionId) {
    return withPlaceholder([], "연결을 선택하세요");
  }
  const schema = getDbSchema(connectionId);
  if (!schema) {
    ensureDbSchema(connectionId);
    return withPlaceholder([], "스키마 로딩 중");
  }
  const schemas = schema.schemas || {};
  const options = Object.keys(schemas).map((name) => ({
    value: name,
    label: name,
  }));
  return withPlaceholder(options, "스키마 선택");
}

function getTableOptions(node) {
  const connectionId = node.config?.connection_id;
  const schemaName = node.config?.schema;
  if (!connectionId || !schemaName) {
    return withPlaceholder([], "스키마를 선택하세요");
  }
  const schema = getDbSchema(connectionId);
  if (!schema) {
    ensureDbSchema(connectionId);
    return withPlaceholder([], "스키마 로딩 중");
  }
  const tables = schema.schemas?.[schemaName]?.tables || {};
  const options = Object.keys(tables).map((name) => ({
    value: name,
    label: name,
  }));
  return withPlaceholder(options, "테이블 선택");
}

function getColumnOptions(node) {
  const connectionId = node.config?.connection_id;
  const schemaName = node.config?.schema;
  const tableName = node.config?.table;
  if (!connectionId || !schemaName || !tableName) {
    return [];
  }
  const schema = getDbSchema(connectionId);
  if (!schema) {
    ensureDbSchema(connectionId);
    return [];
  }
  const columns = schema.schemas?.[schemaName]?.tables?.[tableName]?.columns || [];
  return columns.map((column) => ({
    value: column.name,
    label: `${column.name} (${column.type})`,
  }));
}

function getFilterColumnOptions(node) {
  const options = getColumnOptions(node).map((option) => ({
    value: option.value,
    label: option.value,
  }));
  return withPlaceholder(options, "필터 없음");
}

function normalizeFunctionToolConfig(node) {
  const config = node.config || {};
  const connectionId = config.connection_id;
  if (!connectionId || !dbSchemas[connectionId]) {
    return false;
  }
  let changed = false;
  const schemas = dbSchemas[connectionId].schemas || {};
  if (config.schema && !schemas[config.schema]) {
    config.schema = "";
    config.table = "";
    config.columns = [];
    config.filter_column = "";
    changed = true;
  }
  const tables = config.schema ? schemas[config.schema]?.tables || {} : {};
  if (config.table && !tables[config.table]) {
    config.table = "";
    config.columns = [];
    config.filter_column = "";
    changed = true;
  }
  if (config.table) {
    const columns = tables[config.table]?.columns || [];
    const available = new Set(columns.map((col) => col.name));
    if (Array.isArray(config.columns)) {
      const filtered = config.columns.filter((col) => available.has(col));
      if (filtered.length !== config.columns.length) {
        config.columns = filtered;
        changed = true;
      }
    }
    if (config.filter_column && !available.has(config.filter_column)) {
      config.filter_column = "";
      changed = true;
    }
  }
  return changed;
}

async function refreshDbConnections() {
  if (!dbStatus) {
    return;
  }
  setDbStatus("DB 연결 목록 불러오는 중...");
  try {
    const response = await fetch("/api/db/connections");
    if (!response.ok) {
      setDbStatus("DB 연결 목록 불러오기 실패", true);
      return;
    }
    const data = await response.json();
    dbConnections = Array.isArray(data.connections) ? data.connections : [];
    renderDbConnectionsList();
    refreshFunctionToolNodes();
    setDbStatus(`DB 연결 ${dbConnections.length}개`);
  } catch (error) {
    setDbStatus("DB 연결 목록 불러오기 실패: 네트워크 오류", true);
  }
}

async function connectDb() {
  if (!dbConnectButton) {
    return;
  }
  const payload = {
    name: dbNameInput?.value.trim() || "",
    db_type: dbTypeSelect?.value || "postgresql",
    host: dbHostInput?.value.trim() || "",
    port: Number(dbPortInput?.value) || 5432,
    database: dbDatabaseInput?.value.trim() || "",
    user: dbUserInput?.value.trim() || "",
    password: dbPasswordInput?.value || "",
  };
  if (!payload.host || !payload.database || !payload.user) {
    setDbStatus("Host/Database/User를 입력하세요.", true);
    return;
  }
  setDbStatus("DB 연결 중...");
  try {
    const response = await fetch("/api/db/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const detail = await response.json();
      setDbStatus(`연결 실패: ${detail?.detail?.error || "오류"}`, true);
      return;
    }
    const data = await response.json();
    const connection = data.connection || {};
    if (connection.id) {
      const index = dbConnections.findIndex((item) => item.id === connection.id);
      if (index >= 0) {
        dbConnections[index] = connection;
      } else {
        dbConnections.push(connection);
      }
    }
    if (connection.id && data.schema) {
      dbSchemas[connection.id] = data.schema;
    }
    renderDbConnectionsList();
    refreshFunctionToolNodes();
    setDbStatus(`연결 완료: ${connection.name || connection.id}`);
    if (dbPasswordInput) {
      dbPasswordInput.value = "";
    }
    if (selectedNodeId) {
      const node = nodes.get(selectedNodeId);
      if (node && node.type === "function_tool") {
        renderNodeFields(node);
      }
    }
  } catch (error) {
    setDbStatus("연결 실패: 네트워크 오류", true);
  }
}

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

function applyCanvasTransform() {
  canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${currentScale})`;
}

function setCanvasScale(scale) {
  const next = Math.max(0.5, Math.min(1.8, scale));
  currentScale = Number(next.toFixed(3));
  applyCanvasTransform();
  updateConnections();
}

function handleCanvasWheel(event) {
  event.preventDefault();
  const zoomFactor = event.deltaY < 0 ? 1.08 : 0.92;
  setCanvasScale(currentScale * zoomFactor);
}

function isBackgroundPanTarget(target) {
  if (!target || !(target instanceof Element)) {
    return false;
  }
  if (target.closest(".node")) {
    return false;
  }
  if (target.closest(".palette")) {
    return false;
  }
  if (target.closest(".inspector-panel")) {
    return false;
  }
  if (target.closest(".canvas-toolbar")) {
    return false;
  }
  return true;
}

function startCanvasPan(event) {
  if (event.button !== 0 || !isBackgroundPanTarget(event.target)) {
    return;
  }
  event.preventDefault();
  isPanning = true;
  panMoved = false;
  panStartX = event.clientX;
  panStartY = event.clientY;
  panOriginX = panX;
  panOriginY = panY;
  canvas.classList.add("panning");
  if (canvasWrap) {
    canvasWrap.classList.add("panning");
  }
  document.addEventListener("mousemove", moveCanvasPan);
  document.addEventListener("mouseup", endCanvasPan);
}

function moveCanvasPan(event) {
  if (!isPanning) {
    return;
  }
  const deltaX = event.clientX - panStartX;
  const deltaY = event.clientY - panStartY;
  if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) {
    panMoved = true;
  }
  panX = panOriginX + deltaX;
  panY = panOriginY + deltaY;
  applyCanvasTransform();
}

function endCanvasPan() {
  if (!isPanning) {
    return;
  }
  isPanning = false;
  canvas.classList.remove("panning");
  if (canvasWrap) {
    canvasWrap.classList.remove("panning");
  }
  if (panMoved) {
    suppressCanvasClick = true;
  }
  document.removeEventListener("mousemove", moveCanvasPan);
  document.removeEventListener("mouseup", endCanvasPan);
}

function buildPalette() {
  paletteList.innerHTML = "";
  NODE_GROUPS.forEach((group) => {
    const section = document.createElement("div");
    section.className = "palette-section";
    const heading = document.createElement("h4");
    heading.textContent = group.title;
    section.appendChild(heading);

    group.nodes.forEach((typeKey) => {
      const type = NODE_LIBRARY[typeKey];
      if (!type) {
        return;
      }
      const item = document.createElement("div");
      item.className = "palette-item";
      const iconText = type.icon || type.title.slice(0, 1);
      const iconStyle = type.color
        ? `style="background:${type.color}1a;color:${type.color};"`
        : "";
      item.innerHTML = `<div class="palette-icon" ${iconStyle}>${iconText}</div><div class="palette-copy"><strong>${type.title}</strong><span>${type.desc}</span></div>`;
      item.addEventListener("click", () => {
        const rect = (canvasWrap || canvas).getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const baseCenterX = (centerX - panX) / currentScale;
        const baseCenterY = (centerY - panY) / currentScale;
        const x = baseCenterX - 80 + Math.random() * 80;
        const y = baseCenterY - 60 + Math.random() * 80;
        createNode(typeKey, x, y);
      });
      section.appendChild(item);
    });

    paletteList.appendChild(section);
  });
}

function normalizeLoadedNode(rawNode) {
  const normalized = { ...rawNode };
  normalized.type = LEGACY_TYPE_MAP[normalized.type] || normalized.type;
  normalized.config = normalized.config ? { ...normalized.config } : {};

  if (rawNode.type === "function") {
    const toolset = LEGACY_TOOLSET_MAP[rawNode.subtype];
    if (toolset) {
      normalized.config.toolset = normalized.config.toolset || toolset;
      normalized.subtype = toolset;
    }
  }

  if (normalized.type === "agent" && normalized.config.agent_profile) {
    normalized.subtype = normalized.config.agent_profile;
  }
  if (normalized.type === "mcp" && normalized.config.toolset) {
    normalized.subtype = normalized.config.toolset;
  }
  if (normalized.type === "function_tool" && normalized.config.tool_type) {
    normalized.subtype = normalized.config.tool_type;
  }

  const defaults = NODE_DEFAULTS[normalized.type];
  if (defaults && defaults.config) {
    normalized.config = { ...defaults.config, ...normalized.config };
  }
  if (normalized.type === "function_tool" && !normalized.subtype) {
    normalized.subtype = normalized.config.tool_type || "db_query";
  }
  if (!normalized.keywords) {
    normalized.keywords = [];
  }
  return normalized;
}

function createNode(typeKey, x, y, options = {}) {
  const type = NODE_LIBRARY[typeKey];
  if (!type) {
    return null;
  }

  const defaults = NODE_DEFAULTS[typeKey] || {};
  const id = options.id || `node-${nodeCounter++}`;
  const label = options.label || type.title;
  const executionMode =
    options.execution_mode || (typeKey === "agent" ? "handoff" : "");
  const config = { ...(defaults.config || {}), ...(options.config || {}) };
  const subtype =
    options.subtype ||
    defaults.subtype ||
    (typeKey === "function_tool" ? config.tool_type || "" : "");
  const keywords = options.keywords || defaults.keywords || [];
  const inputFormat = options.input_format || "";
  const outputFormat = options.output_format || "";

  const nodeEl = document.createElement("div");
  nodeEl.className = "node";
  nodeEl.style.left = `${x}px`;
  nodeEl.style.top = `${y}px`;
  nodeEl.style.setProperty("--node-color", type.color);
  nodeEl.style.setProperty("--node-header-bg", toRgba(type.color, 0.14));
  nodeEl.dataset.nodeId = id;

  const header = document.createElement("div");
  header.className = "node-header";
  const titleEl = document.createElement("span");
  titleEl.className = "node-title";
  const toggleButton = document.createElement("button");
  toggleButton.type = "button";
  toggleButton.className = "node-toggle";
  toggleButton.textContent = "v";
  toggleButton.title = "세부 내용 접기/펴기";
  header.appendChild(titleEl);
  header.appendChild(toggleButton);

  const body = document.createElement("div");
  body.className = "node-body";

  const portsConfig = NODE_PORTS[typeKey] || { in: ["in"], out: ["out"] };
  const ports = document.createElement("div");
  ports.className = "node-ports";

  const inGroup = document.createElement("div");
  inGroup.className = "ports-group in";
  const outGroup = document.createElement("div");
  outGroup.className = "ports-group out";

  const nodePorts = { in: {}, out: {} };

  portsConfig.in.forEach((portName) => {
    const port = document.createElement("div");
    port.className = "port";
    port.dataset.port = portName;
    port.dataset.direction = "in";
    port.dataset.label = portName;
    inGroup.appendChild(port);
    nodePorts.in[portName] = port;
  });

  portsConfig.out.forEach((portName) => {
    const port = document.createElement("div");
    port.className = "port";
    port.dataset.port = portName;
    port.dataset.direction = "out";
    port.dataset.label = portName;
    outGroup.appendChild(port);
    nodePorts.out[portName] = port;
  });

  if (portsConfig.in.length || portsConfig.out.length) {
    ports.appendChild(inGroup);
    ports.appendChild(outGroup);
  }

  nodeEl.appendChild(header);
  nodeEl.appendChild(body);
  if (portsConfig.in.length || portsConfig.out.length) {
    nodeEl.appendChild(ports);
  }
  canvas.appendChild(nodeEl);

  const node = {
    id,
    type: typeKey,
    subtype,
    execution_mode: executionMode,
    label,
    keywords,
    input_format: inputFormat,
    output_format: outputFormat,
    config,
    x,
    y,
    color: type.color,
    element: nodeEl,
    header,
    titleEl,
    toggleButton,
    body,
    ports: nodePorts,
    inPorts: Object.values(nodePorts.in),
    outPorts: Object.values(nodePorts.out),
    collapsed: false,
  };

  nodes.set(id, node);

  updateNodeVisual(node);
  enableDrag(node, nodeEl);
  enablePorts(node);

  nodeEl.addEventListener("click", (event) => {
    event.stopPropagation();
    selectNode(node);
  });
  toggleButton.addEventListener("click", (event) => {
    event.stopPropagation();
    selectNode(node);
    toggleNodeCollapse(node);
  });

  updateFlowJson();
  updateConnections();
  selectNode(node);
  markDirty();
  updateToggleAllLabel();

  return node;
}

function updateNodeVisual(node) {
  const title = NODE_LIBRARY[node.type]?.title || node.type;
  const metaLines = buildNodeMeta(node);
  node.element.classList.toggle("collapsed", Boolean(node.collapsed));
  node.titleEl.textContent = node.label;
  if (node.toggleButton) {
    const collapsed = Boolean(node.collapsed);
    node.toggleButton.textContent = collapsed ? ">" : "v";
    node.toggleButton.title = collapsed ? "세부 내용 펼치기" : "세부 내용 접기";
  }
  node.body.innerHTML = `
    <div class="node-type">${title}</div>
    ${metaLines.map((line) => `<div class="node-meta">${line}</div>`).join("")}
  `;
  updateToggleAllLabel();
}

function toggleNodeCollapse(node) {
  if (!node) {
    return;
  }
  node.collapsed = !node.collapsed;
  node.element.classList.toggle("collapsed", node.collapsed);
  updateNodeVisual(node);
  updateConnections();
}

function updateToggleAllLabel() {
  if (!toggleAllNodesButton) {
    return;
  }
  const hasExpanded = Array.from(nodes.values()).some((node) => !node.collapsed);
  toggleAllNodesButton.textContent = hasExpanded ? "전체 접기" : "전체 펼치기";
}

function setAllNodesCollapsed(collapsed) {
  nodes.forEach((node) => {
    node.collapsed = collapsed;
    node.element.classList.toggle("collapsed", collapsed);
    updateNodeVisual(node);
  });
  updateConnections();
  updateToggleAllLabel();
}

function buildNodeMeta(node) {
  const config = node.config || {};
  const lines = [];
  if (node.type === "start") {
    const inputs = formatList(config.input_variables);
    const states = formatList(config.state_variables);
    lines.push(`입력: ${inputs || "input_as_text"}`);
    if (states) {
      lines.push(`상태: ${states}`);
    }
  }
  if (node.type === "agent") {
    const profile = config.agent_profile || node.subtype || "custom";
    lines.push(`모드: ${node.execution_mode || "handoff"}`);
    lines.push(`프로필: ${profile}`);
    if (config.model) {
      lines.push(`모델: ${config.model}`);
    }
  }
  if (node.type === "guardrail") {
    const toggles = [];
    if (config.pii) toggles.push("PII");
    if (config.moderation) toggles.push("Moderation");
    if (config.jailbreak) toggles.push("Jailbreak");
    if (config.hallucination) toggles.push("Hallucination");
    lines.push(`검사: ${toggles.length ? toggles.join(", ") : "없음"}`);
  }
  if (node.type === "mcp") {
    const toolset = config.toolset || node.subtype || "custom";
    lines.push(`Toolset: ${toolset}`);
    if (config.server_name) {
      lines.push(`Server: ${config.server_name}`);
    }
  }
  if (node.type === "file_search") {
    if (config.vector_store_id) {
      lines.push(`Vector: ${config.vector_store_id}`);
    }
    lines.push(`TopK: ${config.top_k || 3}`);
  }
  if (node.type === "function_tool") {
    const toolType = config.tool_type || node.subtype || "db_query";
    lines.push(`Tool: ${toolType}`);
    if (toolType === "db_query") {
      const connectionLabel = getDbConnectionLabel(config.connection_id);
      if (connectionLabel) {
        lines.push(`DB: ${connectionLabel}`);
      }
      const target = [config.schema, config.table].filter(Boolean).join(".");
      if (target) {
        lines.push(`Target: ${target}`);
      }
      if (config.filter_column) {
        const op = config.filter_operator || "ilike";
        lines.push(`Filter: ${config.filter_column} ${op}`);
      }
    }
  }
  if (node.type === "end") {
    lines.push(`Return: ${config.return_type || "text"}`);
  }
  if (node.type === "note" && config.note) {
    lines.push(truncate(config.note, 36));
  }
  if (node.type === "if_else") {
    lines.push(`조건: ${truncate(config.condition || "", 36) || "없음"}`);
  }
  if (node.type === "while") {
    lines.push(`조건: ${truncate(config.condition || "", 36) || "없음"}`);
    lines.push(`반복: ${config.max_iterations || 3}회`);
  }
  if (node.type === "user_approval") {
    if (config.auto_approve) {
      lines.push("자동 승인");
    } else {
      lines.push(`기본: ${config.default_decision || "approved"}`);
    }
  }
  if (node.type === "transform") {
    lines.push(`템플릿: ${truncate(config.template || "", 36) || "없음"}`);
  }
  if (node.type === "state") {
    const summary = formatList(config.assignments);
    lines.push(`State: ${summary || "없음"}`);
  }
  if (node.keywords && node.keywords.length && node.type === "agent") {
    lines.push(`키워드: ${node.keywords.join(", ")}`);
  }
  return lines;
}

function formatList(value) {
  if (!value) {
    return "";
  }
  if (Array.isArray(value)) {
    return value.filter(Boolean).join(", ");
  }
  if (typeof value === "string") {
    return value
      .split(/[,\n]/)
      .map((item) => item.trim())
      .filter(Boolean)
      .join(", ");
  }
  return String(value);
}

function truncate(value, maxLength) {
  const text = String(value || "");
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1)}…`;
}

function enableDrag(node, handle) {
  handle.addEventListener("mousedown", (event) => {
    if (event.button !== 0) {
      return;
    }
    if (
      event.target.closest(".port") ||
      event.target.closest(".node-toggle")
    ) {
      return;
    }
    event.preventDefault();
    selectNode(node);
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = node.element.offsetLeft;
    const startTop = node.element.offsetTop;

    function onMove(moveEvent) {
      const deltaX = (moveEvent.clientX - startX) / currentScale;
      const deltaY = (moveEvent.clientY - startY) / currentScale;
      const nextLeft = startLeft + deltaX;
      const nextTop = startTop + deltaY;
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
  node.outPorts.forEach((port) => {
    port.addEventListener("click", (event) => {
      event.stopPropagation();
      clearPortHighlights();
      port.classList.add("active");
      pendingConnection = { from: node.id, fromPort: port.dataset.port };
    });
  });

  node.inPorts.forEach((port) => {
    port.addEventListener("click", (event) => {
      event.stopPropagation();
      if (!pendingConnection) {
        return;
      }
      if (pendingConnection.from === node.id) {
        clearPortHighlights();
        pendingConnection = null;
        return;
      }
      createEdge(
        pendingConnection.from,
        node.id,
        pendingConnection.fromPort,
        port.dataset.port
      );
      clearPortHighlights();
      pendingConnection = null;
    });
  });
}

function clearPortHighlights() {
  nodes.forEach((node) => {
    node.outPorts.forEach((port) => port.classList.remove("active"));
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

function createEdge(
  fromId,
  toId,
  fromPort = "out",
  toPort = "in",
  kindOverride = ""
) {
  const fromNode = nodes.get(fromId);
  const toNode = nodes.get(toId);
  if (!fromNode || !toNode) {
    return;
  }

  const kind = kindOverride || getEdgeKind(fromNode, toNode);
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("stroke", fromNode.color);
  path.setAttribute("stroke-width", "2");
  path.setAttribute("fill", "none");
  path.setAttribute("marker-end", "url(#arrow)");
  path.dataset.kind = kind;
  connections.appendChild(path);

  edges.push({
    id: `edge-${edges.length + 1}`,
    from: fromId,
    to: toId,
    from_port: fromPort,
    to_port: toPort,
    kind,
    element: path,
  });

  updateConnections();
  updateFlowJson();
  markDirty();
}

function getEdgeKind(fromNode, toNode) {
  if (fromNode.type === "agent" && toNode.type === "agent") {
    return "handoff";
  }
  if (
    fromNode.type === "agent" &&
    (toNode.type === "mcp" ||
      toNode.type === "file_search" ||
      toNode.type === "function_tool")
  ) {
    return "tool";
  }
  return "flow";
}

function updateConnections() {
  edges.forEach((edge) => {
    const fromNode = nodes.get(edge.from);
    const toNode = nodes.get(edge.to);
    if (!fromNode || !toNode) {
      return;
    }

    const fromPort =
      fromNode.ports.out[edge.from_port || "out"] || fromNode.outPorts[0];
    const toPort =
      toNode.ports.in[edge.to_port || "in"] || toNode.inPorts[0];
    if (!fromPort || !toPort) {
      return;
    }
    const start = getPortCenter(fromPort);
    const end = getPortCenter(toPort);
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
    x:
      (portRect.left - canvasRect.left + portRect.width / 2) / currentScale,
    y:
      (portRect.top - canvasRect.top + portRect.height / 2) / currentScale,
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
      config: node.config || {},
    })),
    edges: edges.map((edge) => ({
      from: edge.from,
      to: edge.to,
      from_port: edge.from_port || "out",
      to_port: edge.to_port || "in",
      kind: edge.kind || "flow",
    })),
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
  updateToggleAllLabel();
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
    const normalized = normalizeLoadedNode(node);
    const position = normalized.position || { x: 80, y: 80 };
    const created = createNode(normalized.type, position.x, position.y, {
      id: normalized.id,
      subtype: normalized.subtype,
      execution_mode: normalized.execution_mode,
      label: normalized.label,
      keywords: normalized.keywords || [],
      input_format: normalized.input_format || "",
      output_format: normalized.output_format || "",
      config: normalized.config || {},
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
    const fromNode = nodes.get(edge.from);
    const toNode = nodes.get(edge.to);
    const inferredKind =
      edge.kind ||
      (fromNode && toNode ? getEdgeKind(fromNode, toNode) : "flow");
    createEdge(
      edge.from,
      edge.to,
      edge.from_port || "out",
      edge.to_port || "in",
      inferredKind
    );
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
  updateToggleAllLabel();
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
  nodeTypeInput.value = NODE_LIBRARY[node.type]?.title || node.type;
  if (node.type === "agent") {
    nodeExecutionField.classList.remove("hidden");
    nodeExecutionSelect.value = node.execution_mode || "handoff";
  } else {
    nodeExecutionField.classList.add("hidden");
  }
  renderNodeFields(node);
  updateInspectorVisibility(node);
}

function clearSelection() {
  if (selectedNodeId && nodes.has(selectedNodeId)) {
    nodes.get(selectedNodeId).element.classList.remove("selected");
  }
  selectedNodeId = null;
  nodeEmpty.classList.remove("hidden");
  nodeForm.classList.add("hidden");
  if (nodeFields) {
    nodeFields.innerHTML = "";
  }
  fieldInputMap = new Map();
  activeFieldDefs = [];
  updateInspectorVisibility(null);
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
  updateToggleAllLabel();
}

function updateSelectedNode() {
  if (!selectedNodeId || !nodes.has(selectedNodeId)) {
    return;
  }
  const node = nodes.get(selectedNodeId);
  node.label = nodeLabelInput.value.trim() || node.label;
  if (node.type === "agent") {
    node.execution_mode = nodeExecutionSelect.value;
  }
  if (!node.config) {
    node.config = {};
  }

  fieldInputMap.forEach(({ field, input }) => {
    const value = parseFieldValue(field, input);
    if (field.scope === "root") {
      node[field.key] = value;
    } else {
      node.config[field.key] = value;
    }
  });

  if (node.type === "agent" && node.config.agent_profile) {
    node.subtype = node.config.agent_profile;
  }
  if (node.type === "mcp" && node.config.toolset) {
    node.subtype = node.config.toolset;
  }
  if (node.type === "function_tool" && node.config.tool_type) {
    node.subtype = node.config.tool_type;
    normalizeFunctionToolConfig(node);
  }
  updateNodeVisual(node);
  updateFlowJson();
  markDirty();
}

function renderNodeFields(node) {
  if (!nodeFields) {
    return;
  }
  nodeFields.innerHTML = "";
  fieldInputMap = new Map();
  activeFieldDefs = NODE_FIELDS[node.type] || [];
  node.config = node.config || {};
  if (node.type === "function_tool") {
    const changed = normalizeFunctionToolConfig(node);
    if (changed && !isLoading) {
      updateNodeVisual(node);
      updateFlowJson();
      markDirty();
    }
  }

  activeFieldDefs.forEach((field) => {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.textContent = field.label;
    wrapper.appendChild(label);

    let input;
    if (field.type === "textarea" || field.type === "list") {
      input = document.createElement("textarea");
      if (field.placeholder) {
        input.placeholder = field.placeholder;
      }
    } else if (field.type === "select" || field.type === "multiselect") {
      input = document.createElement("select");
      if (field.type === "multiselect") {
        input.multiple = true;
      }
      const options =
        typeof field.options === "function" ? field.options(node) : field.options || [];
      options.forEach((optionDef) => {
        const option = document.createElement("option");
        option.value = optionDef.value;
        option.textContent = optionDef.label;
        input.appendChild(option);
      });
      if (field.type === "multiselect") {
        const size = Math.min(8, Math.max(4, options.length || 4));
        input.size = size;
      }
    } else if (field.type === "checkbox") {
      input = document.createElement("input");
      input.type = "checkbox";
    } else {
      input = document.createElement("input");
      if (field.type === "number") {
        input.type = "number";
      } else if (field.type === "password") {
        input.type = "password";
      } else {
        input.type = "text";
      }
      if (field.placeholder) {
        input.placeholder = field.placeholder;
      }
    }

    const value =
      field.scope === "root" ? node[field.key] : node.config[field.key];
    applyFieldValue(field, input, value);

    const handleUpdate = () => {
      updateSelectedNode();
    };
    const handleRefresh = () => {
      updateSelectedNode();
      const currentNode = nodes.get(selectedNodeId);
      if (currentNode) {
        renderNodeFields(currentNode);
      }
    };
    if (field.refreshOnChange) {
      input.addEventListener("input", handleRefresh);
      input.addEventListener("change", handleRefresh);
    } else {
      input.addEventListener("input", handleUpdate);
      input.addEventListener("change", handleUpdate);
    }

    wrapper.appendChild(input);
    nodeFields.appendChild(wrapper);
    fieldInputMap.set(field.key, { field, input });
  });
}

function applyFieldValue(field, input, value) {
  if (field.type === "checkbox") {
    input.checked = Boolean(value);
    return;
  }
  if (field.type === "list") {
    input.value = Array.isArray(value) ? value.join(", ") : value || "";
    return;
  }
  if (field.type === "multiselect") {
    const values = Array.isArray(value)
      ? value.map((item) => String(item))
      : [];
    Array.from(input.options).forEach((option) => {
      option.selected = values.includes(option.value);
    });
    return;
  }
  if (field.type === "number") {
    input.value = value ?? "";
    return;
  }
  input.value = value ?? "";
}

function parseFieldValue(field, input) {
  if (field.type === "checkbox") {
    return input.checked;
  }
  if (field.type === "number") {
    const parsed = Number(input.value);
    return Number.isNaN(parsed) ? 0 : parsed;
  }
  if (field.type === "list") {
    return input.value
      .split(/[,\n]/)
      .map((chunk) => chunk.trim())
      .filter(Boolean);
  }
  if (field.type === "multiselect") {
    return Array.from(input.selectedOptions || [])
      .map((option) => option.value)
      .filter(Boolean);
  }
  return input.value;
}

function formatNodeTitle(node) {
  if (!node) {
    return "알 수 없음";
  }
  const label = node.label || node.id || "노드";
  const subtype =
    node.subtype ||
    node.config?.agent_profile ||
    node.config?.toolset ||
    node.config?.tool_type ||
    "-";
  const typeText = node.type ? `${node.type}/${subtype}` : "노드";
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
  const existingName = workflowNameInput?.value.trim() || "";
  const nextName = window.prompt("워크플로우 이름을 입력하세요.", existingName);
  if (!nextName || !nextName.trim()) {
    workflowStatus.textContent = "저장 취소: 이름이 필요합니다.";
    return;
  }
  workflowNameInput.value = nextName.trim();
  const payload = buildWorkflowPayload();
  workflowStatus.textContent = "저장 중...";
  try {
    const response = await fetch("/api/workflows", {
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
    workflowStatus.textContent = `저장 완료: ${data.workflow?.meta?.updated_at || ""}`;
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
      name: "OpenAI Agent Builder 스타일 워크플로우",
      description: "Start → Guardrail → Agent → End, MCP/File Search는 에이전트 도구로 연결",
      updated_at: new Date().toISOString(),
    },
    nodes: [
      {
        id: "node-start",
        type: "start",
        subtype: "start",
        label: "Start",
        keywords: [],
        input_format: "",
        output_format: "",
        position: { x: 80, y: 160 },
        config: {
          input_variables: ["input_as_text"],
          state_variables: ["conversation_history:text"],
        },
      },
      {
        id: "node-guardrail",
        type: "guardrail",
        subtype: "guardrail",
        label: "Guardrail",
        keywords: [],
        input_format: "",
        output_format: "",
        position: { x: 300, y: 160 },
        config: {
          input: "input_as_text",
          pii: true,
          moderation: true,
          jailbreak: true,
        },
      },
      {
        id: "node-orchestrator",
        type: "agent",
        subtype: "orchestrator",
        execution_mode: "handoff",
        label: "Orchestrator Agent",
        keywords: [
          "lot",
          "로트",
          "line",
          "라인",
          "status",
          "상태",
          "data",
          "데이터",
          "조회",
          "예측",
          "시뮬레이션",
          "simulation",
          "forecast",
          "what-if",
          "risk",
          "수율",
        ],
        input_format: "user message (Korean)",
        output_format: "routing decision + tool/agent call",
        position: { x: 520, y: 160 },
        config: {
          agent_profile: "orchestrator",
          instructions:
            "You are the conversation lead for a manufacturing monitoring demo. Always respond in Korean. Do not mention internal routing, tools, or intent labels. Decide whether the user wants process/LOT data or a simulation. If data lookup, call the db_agent tool with the user message. If simulation/prediction, call the simulation_agent tool. If both are requested, call db_agent first, then simulation_agent. After tool output, respond naturally in 1-3 sentences. Do not expose the tool report format; translate it into natural Korean. If the tool reports missing fields, ask a single clear question for those fields. If status is ok, summarize the result and suggest one optional next step. Do not invent LOT IDs or parameters.",
          include_chat_history: true,
          response_format: "text",
        },
      },
      {
        id: "node-end",
        type: "end",
        subtype: "end",
        label: "End",
        keywords: [],
        input_format: "",
        output_format: "",
        position: { x: 760, y: 160 },
        config: { return_type: "text" },
      },
      {
        id: "node-db-agent",
        type: "agent",
        subtype: "db_agent",
        execution_mode: "as_tool",
        label: "DB Agent",
        keywords: [
          "lot",
          "로트",
          "line",
          "라인",
          "status",
          "상태",
          "data",
          "데이터",
          "조회",
          "process",
          "공정",
        ],
        input_format: "lot_id or line/status query",
        output_format: "process rows + UI update",
        position: { x: 520, y: 20 },
        config: {
          agent_profile: "db_agent",
          instructions:
            "You are an internal DB helper used by the orchestrator. Do not address the user directly. Always respond in Korean. If a LOT ID is present, call get_lot_info(lot_id, limit=12). If no LOT ID but the user mentions line/status/conditions, call get_process_data(query=original_message, limit=12). If required info is missing, do not ask the user; report missing fields. Do not invent LOT IDs or data. Prefer tool calls over direct answers. Return exactly four lines: status: <ok|missing|error>. summary: <Korean 1 sentence>. missing: <comma-separated fields or 'none'>. next: <Korean follow-up question or 'none'>.",
        },
      },
      {
        id: "node-sim-agent",
        type: "agent",
        subtype: "simulation_agent",
        execution_mode: "as_tool",
        label: "Simulation Agent",
        keywords: [
          "예측",
          "시뮬레이션",
          "simulation",
          "forecast",
          "what-if",
          "risk",
          "리스크",
          "수율",
          "throughput",
        ],
        input_format: "lot_id or temperature/voltage/size/capacity/production_mode",
        output_format: "simulation result + UI update",
        position: { x: 520, y: 300 },
        config: {
          agent_profile: "simulation_agent",
          instructions:
            "You are an internal simulation helper used by the orchestrator. Do not address the user directly. Always respond in Korean. When the user requests a simulation, call open_simulation_form first to open the UI panel (unless it is already open). If a LOT ID is provided, call run_lot_simulation(lot_id). Otherwise collect the five required params: temperature, voltage, size, capacity, production_mode. If any params are provided, call update_simulation_params with those values or with message=original_message to extract. Do not ask the user directly; report missing fields. Never ask for values that are already filled. When all five params are available, call run_simulation. Return exactly four lines: status: <ok|missing|error>. summary: <Korean 1 sentence>. missing: <comma-separated fields or 'none'>. next: <Korean follow-up question or 'none'>.",
        },
      },
      {
        id: "node-file-search",
        type: "file_search",
        subtype: "file_search",
        label: "File Search",
        keywords: [],
        input_format: "query",
        output_format: "results",
        position: { x: 520, y: 420 },
        config: { top_k: 3 },
      },
      {
        id: "node-mcp-db",
        type: "mcp",
        subtype: "process_db",
        label: "MCP: Process DB",
        keywords: [],
        input_format: "query",
        output_format: "rows",
        position: { x: 760, y: 20 },
        config: { toolset: "process_db", server_name: "process_db" },
      },
      {
        id: "node-mcp-sim",
        type: "mcp",
        subtype: "simulation",
        label: "MCP: Simulation",
        keywords: [],
        input_format: "params",
        output_format: "result",
        position: { x: 760, y: 300 },
        config: { toolset: "simulation", server_name: "simulation_api" },
      },
      {
        id: "node-mcp-ui",
        type: "mcp",
        subtype: "frontend_trigger",
        label: "MCP: Frontend Trigger",
        keywords: [],
        input_format: "message",
        output_format: "ui_event",
        position: { x: 760, y: 420 },
        config: { toolset: "frontend_trigger", server_name: "frontend_trigger" },
      },
    ],
    edges: [
      {
        from: "node-start",
        to: "node-guardrail",
        from_port: "out",
        to_port: "in",
        kind: "flow",
      },
      {
        from: "node-guardrail",
        to: "node-orchestrator",
        from_port: "pass",
        to_port: "in",
        kind: "flow",
      },
      {
        from: "node-guardrail",
        to: "node-end",
        from_port: "fail",
        to_port: "in",
        kind: "flow",
      },
      {
        from: "node-orchestrator",
        to: "node-end",
        from_port: "out",
        to_port: "in",
        kind: "flow",
      },
      {
        from: "node-orchestrator",
        to: "node-db-agent",
        from_port: "out",
        to_port: "in",
        kind: "handoff",
      },
      {
        from: "node-orchestrator",
        to: "node-sim-agent",
        from_port: "out",
        to_port: "in",
        kind: "handoff",
      },
      {
        from: "node-orchestrator",
        to: "node-file-search",
        from_port: "out",
        to_port: "in",
        kind: "tool",
      },
      {
        from: "node-db-agent",
        to: "node-mcp-db",
        from_port: "out",
        to_port: "in",
        kind: "tool",
      },
      {
        from: "node-db-agent",
        to: "node-mcp-ui",
        from_port: "out",
        to_port: "in",
        kind: "tool",
      },
      {
        from: "node-sim-agent",
        to: "node-mcp-sim",
        from_port: "out",
        to_port: "in",
        kind: "tool",
      },
      {
        from: "node-sim-agent",
        to: "node-mcp-ui",
        from_port: "out",
        to_port: "in",
        kind: "tool",
      },
    ],
  };
}

async function copyFlowJson(triggerButton) {
  if (!flowJson) {
    return;
  }
  updateFlowJson();
  const payload = flowJson.value || "";
  if (!payload) {
    return;
  }
  const button = triggerButton || null;
  const originalLabel = button ? button.textContent : "";
  try {
    await navigator.clipboard.writeText(payload);
    if (button) {
      button.textContent = "복사됨";
      setTimeout(() => {
        button.textContent = originalLabel;
      }, 1500);
    }
  } catch (error) {
    try {
      flowJson.focus();
      flowJson.select();
    } catch (selectError) {
    }
    if (button) {
      button.textContent = "복사 실패";
      setTimeout(() => {
        button.textContent = originalLabel;
      }, 1500);
    }
  }
}

exportButton.addEventListener("click", () => {
  copyFlowJson(exportButton);
});

if (toggleAllNodesButton) {
  toggleAllNodesButton.addEventListener("click", () => {
    const hasExpanded = Array.from(nodes.values()).some((node) => !node.collapsed);
    setAllNodesCollapsed(hasExpanded);
  });
}

if (copyButton) {
  copyButton.addEventListener("click", () => {
    copyFlowJson(copyButton);
  });
}

loadSampleButton.addEventListener("click", loadSample);
clearButton.addEventListener("click", resetCanvas);

window.addEventListener("resize", updateConnections);
canvas.addEventListener("click", () => {
  if (suppressCanvasClick) {
    suppressCanvasClick = false;
    return;
  }
  pendingConnection = null;
  clearPortHighlights();
  clearSelection();
});
canvas.addEventListener("wheel", handleCanvasWheel, { passive: false });
if (canvasWrap) {
  canvasWrap.addEventListener("mousedown", startCanvasPan);
} else {
  canvas.addEventListener("mousedown", startCanvasPan);
}

nodeLabelInput.addEventListener("input", updateSelectedNode);
nodeExecutionSelect.addEventListener("change", updateSelectedNode);
if (nodeDeleteButton) {
  nodeDeleteButton.addEventListener("click", deleteSelectedNode);
}
if (inspectorCloseButton) {
  inspectorCloseButton.addEventListener("click", clearSelection);
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

if (dbConnectButton) {
  dbConnectButton.addEventListener("click", connectDb);
}
if (dbRefreshButton) {
  dbRefreshButton.addEventListener("click", refreshDbConnections);
}

if (inspectorTabButtons.length) {
  inspectorTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (button.disabled) {
        return;
      }
      setActiveTab(button.dataset.tab);
    });
  });
}

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
refreshDbConnections();
loadWorkflow();
