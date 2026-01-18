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
const newChatButtonEl = document.getElementById("newChatButton");
const insightsToggleEl = document.getElementById("insightsToggle");
const appEl = document.querySelector(".app");
const insightsEl = document.querySelector(".insights");
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
  insightsHidden: false,
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
  // 차트 카드 래퍼를 만든다.
  const card = document.createElement("div");
  card.className = "block chart-card";
  const label = document.createElement("div");
  label.className = "block__label";
  label.textContent = chartId || "chart";
  card.appendChild(label);

  // 요청한 차트 데이터를 찾는다.
  const chart =
    charts && chartId ? charts.find((item) => item.chart_id === chartId) : null;
  if (!chart) {
    const empty = document.createElement("div");
    empty.className = "block__text";
    empty.textContent = "No chart data.";
    card.appendChild(empty);
    return card;
  }

  // 차트 헤더(타이틀/서브텍스트)를 만든다.
  const header = document.createElement("div");
  header.className = "chart-header";
  const title = document.createElement("div");
  title.className = "chart-title";
  title.textContent = chart.title || "Chart";
  header.appendChild(title);
  const subtitleText = chart.subtitle || chart.notes;
  if (subtitleText) {
    const subtitle = document.createElement("div");
    subtitle.className = "chart-subtitle";
    subtitle.textContent = subtitleText;
    header.appendChild(subtitle);
  }
  card.appendChild(header);

  // 시리즈 색상을 정리한다.
  const chartSeries = normalizeChartSeries(chart.series || []);
  const seriesVisibility = chartSeries.map(() => true);
  const chartPayload = { ...chart, series: chartSeries };

  // 시리즈 토글이 필요하면 범례를 만든다.
  if (chartSeries.length > 1) {
    const legend = buildChartLegend(
      chartSeries,
      seriesVisibility,
      () => renderSvg()
    );
    card.appendChild(legend);
  }

  // 차트 프레임과 툴팁을 만든다.
  const frame = document.createElement("div");
  frame.className = "chart-frame";
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  frame.appendChild(tooltip);
  card.appendChild(frame);

  // SVG를 다시 그리는 함수를 만든다.
  function renderSvg() {
    const prev = frame.querySelector("svg");
    if (prev) {
      prev.remove();
    }
    const svg = buildSvgChart(chartPayload, seriesVisibility);
    frame.insertBefore(svg, tooltip);
    bindChartTooltip(svg, tooltip, frame);
  }

  // 첫 렌더를 수행한다.
  renderSvg();
  return card;
}

// SVG 차트를 생성한다.
function buildSvgChart(chart, seriesVisibility) {
  // 기본 차트 영역 치수를 정의한다.
  const width = 640;
  const height = 300;
  const padding = { top: 26, right: 24, bottom: 54, left: 66 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  svg.classList.add("chart-canvas");

  // 표시할 시리즈와 값 범위를 계산한다.
  const seriesList = Array.isArray(chart.series) ? chart.series : [];
  const visibleSeries = seriesList.filter(
    (_series, index) => !seriesVisibility || seriesVisibility[index]
  );
  const baseSeries = visibleSeries[0] || seriesList[0] || { points: [] };
  const basePoints = Array.isArray(baseSeries.points) ? baseSeries.points : [];
  const pointCount = Math.max(basePoints.length, 1);
  const values = [];
  visibleSeries.forEach((series) => {
    const points = Array.isArray(series.points) ? series.points : [];
    points.forEach((point) => {
      const numeric = Number(point.y);
      if (Number.isFinite(numeric)) {
        values.push(numeric);
      }
    });
  });
  const maxValue = Math.max(...values, 1);

  // 그리드와 Y축 눈금을 그린다.
  const yTickCount = 5;
  for (let i = 0; i <= yTickCount; i += 1) {
    const y = padding.top + (plotHeight * i) / yTickCount;
    const value = maxValue - (maxValue * i) / yTickCount;
    const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
    grid.setAttribute("x1", padding.left);
    grid.setAttribute("y1", y);
    grid.setAttribute("x2", width - padding.right);
    grid.setAttribute("y2", y);
    grid.setAttribute("class", "chart-grid-line");
    svg.appendChild(grid);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", padding.left - 10);
    label.setAttribute("y", y + 4);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "chart-axis-text");
    label.textContent = formatChartValue(value);
    svg.appendChild(label);
  }

  // X축 눈금을 그린다.
  const xTickCount = Math.min(6, basePoints.length || 1);
  const tickStep =
    basePoints.length > 1 ? (basePoints.length - 1) / Math.max(xTickCount - 1, 1) : 1;
  for (let i = 0; i < xTickCount; i += 1) {
    const index = Math.round(i * tickStep);
    const x =
      padding.left + (plotWidth * index) / Math.max(basePoints.length - 1, 1);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", x);
    tick.setAttribute("y1", height - padding.bottom);
    tick.setAttribute("x2", x);
    tick.setAttribute("y2", height - padding.bottom + 6);
    tick.setAttribute("class", "chart-axis-line");
    svg.appendChild(tick);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x);
    label.setAttribute("y", height - padding.bottom + 22);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "chart-axis-text");
    label.textContent = formatChartLabel(basePoints[index]?.x, index);
    svg.appendChild(label);
  }

  // 축 라인을 그린다.
  const axisX = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axisX.setAttribute("x1", padding.left);
  axisX.setAttribute("y1", height - padding.bottom);
  axisX.setAttribute("x2", width - padding.right);
  axisX.setAttribute("y2", height - padding.bottom);
  axisX.setAttribute("class", "chart-axis-line");
  svg.appendChild(axisX);

  const axisY = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axisY.setAttribute("x1", padding.left);
  axisY.setAttribute("y1", padding.top);
  axisY.setAttribute("x2", padding.left);
  axisY.setAttribute("y2", height - padding.bottom);
  axisY.setAttribute("class", "chart-axis-line");
  svg.appendChild(axisY);

  // 축 라벨을 그린다.
  if (chart.x_label) {
    const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    xLabel.setAttribute("x", padding.left + plotWidth / 2);
    xLabel.setAttribute("y", height - 10);
    xLabel.setAttribute("text-anchor", "middle");
    xLabel.setAttribute("class", "chart-axis-label");
    xLabel.textContent = chart.x_label;
    svg.appendChild(xLabel);
  }
  if (chart.y_label) {
    const yLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    yLabel.setAttribute(
      "transform",
      `translate(16 ${padding.top + plotHeight / 2}) rotate(-90)`
    );
    yLabel.setAttribute("text-anchor", "middle");
    yLabel.setAttribute("class", "chart-axis-label");
    yLabel.textContent = chart.y_label;
    svg.appendChild(yLabel);
  }

  // 차트 타입에 맞게 시리즈를 그린다.
  const unit = chart.unit || chart.y_unit || "";
  if (chart.type === "line") {
    visibleSeries.forEach((series, seriesIndex) => {
      const points = Array.isArray(series.points) ? series.points : [];
      let pathData = "";
      points.forEach((point, index) => {
        const x =
          padding.left +
          (plotWidth * index) / Math.max(points.length - 1, 1);
        const y =
          height -
          padding.bottom -
          ((Number(point.y) || 0) / maxValue) * plotHeight;
        pathData += index === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
      });
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", pathData);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", series.color || "#0d6c63");
      path.setAttribute("stroke-width", "2");
      path.setAttribute("class", "chart-line");
      svg.appendChild(path);

      points.forEach((point, index) => {
        const value = Number(point.y) || 0;
        const x =
          padding.left +
          (plotWidth * index) / Math.max(points.length - 1, 1);
        const y =
          height -
          padding.bottom -
          (value / maxValue) * plotHeight;
        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        dot.setAttribute("cx", x);
        dot.setAttribute("cy", y);
        dot.setAttribute("r", "4");
        dot.setAttribute("fill", series.color || "#0d6c63");
        dot.setAttribute("class", "chart-point");
        setPointDataset(dot, point, series, seriesIndex, index, unit);
        svg.appendChild(dot);
      });
    });
  } else if (chart.type === "scatter") {
    visibleSeries.forEach((series, seriesIndex) => {
      const points = Array.isArray(series.points) ? series.points : [];
      points.forEach((point, index) => {
        const value = Number(point.y) || 0;
        const x =
          padding.left +
          (plotWidth * index) / Math.max(points.length - 1, 1);
        const y =
          height -
          padding.bottom -
          (value / maxValue) * plotHeight;
        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        dot.setAttribute("cx", x);
        dot.setAttribute("cy", y);
        dot.setAttribute("r", "4.5");
        dot.setAttribute("fill", series.color || "#b5842f");
        dot.setAttribute("class", "chart-point");
        setPointDataset(dot, point, series, seriesIndex, index, unit);
        svg.appendChild(dot);
      });
    });
  } else {
    const seriesCount = Math.max(visibleSeries.length, 1);
    const groupWidth = plotWidth / pointCount;
    const barGroupWidth = groupWidth * 0.72;
    const barWidth = barGroupWidth / seriesCount;
    visibleSeries.forEach((series, seriesIndex) => {
      const points = Array.isArray(series.points) ? series.points : [];
      points.forEach((point, index) => {
        const value = Number(point.y) || 0;
        const barHeight = (value / maxValue) * plotHeight;
        const x =
          padding.left +
          index * groupWidth +
          (groupWidth - barGroupWidth) / 2 +
          seriesIndex * barWidth;
        const y = height - padding.bottom - barHeight;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", x);
        rect.setAttribute("y", y);
        rect.setAttribute("width", barWidth * 0.9);
        rect.setAttribute("height", barHeight);
        rect.setAttribute("rx", "4");
        rect.setAttribute("fill", series.color || "#0d6c63");
        rect.setAttribute("class", "chart-bar");
        setPointDataset(rect, point, series, seriesIndex, index, unit);
        svg.appendChild(rect);
      });
    });
  }

  return svg;
}

// 시리즈 색상을 정리한다.
function normalizeChartSeries(seriesList) {
  const palette = ["#0d6c63", "#b5842f", "#2f6db5", "#8c2f5b", "#2f8c6e"];
  return seriesList.map((series, index) => ({
    ...series,
    color: series.color || palette[index % palette.length],
  }));
}

// 시리즈 토글 범례를 만든다.
function buildChartLegend(seriesList, visibility, onToggle) {
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  seriesList.forEach((series, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chart-legend__item";
    const swatch = document.createElement("span");
    swatch.className = "chart-legend__swatch";
    swatch.style.backgroundColor = series.color || "#0d6c63";
    const text = document.createElement("span");
    text.textContent = series.name || `Series ${index + 1}`;
    button.appendChild(swatch);
    button.appendChild(text);
    button.addEventListener("click", () => {
      const activeCount = visibility.filter(Boolean).length;
      if (visibility[index] && activeCount <= 1) {
        return;
      }
      visibility[index] = !visibility[index];
      button.classList.toggle("is-off", !visibility[index]);
      onToggle();
    });
    legend.appendChild(button);
  });
  return legend;
}

// 툴팁 이벤트를 연결한다.
function bindChartTooltip(svg, tooltipEl, frameEl) {
  function hideTooltip() {
    tooltipEl.classList.remove("is-visible");
  }

  svg.addEventListener("mousemove", (event) => {
    const target = event.target.closest(".chart-point, .chart-bar");
    if (!target || !target.dataset) {
      hideTooltip();
      return;
    }
    const series = target.dataset.series || "-";
    const xLabel = target.dataset.xLabel || "-";
    const yValue = formatChartValue(Number(target.dataset.yValue));
    const unit = target.dataset.unit || "";
    const rank = target.dataset.rank || "-";
    tooltipEl.innerHTML = `
      <div class="chart-tooltip__title">${series}</div>
      <div class="chart-tooltip__row">값: ${yValue}${unit ? ` ${unit}` : ""}</div>
      <div class="chart-tooltip__row">X: ${xLabel}</div>
      <div class="chart-tooltip__row">rank: ${rank}</div>
    `;
    const rect = frameEl.getBoundingClientRect();
    tooltipEl.style.left = `${event.clientX - rect.left}px`;
    tooltipEl.style.top = `${event.clientY - rect.top}px`;
    tooltipEl.classList.add("is-visible");
  });

  svg.addEventListener("mouseleave", hideTooltip);
}

// 데이터 포인트 메타를 설정한다.
function setPointDataset(element, point, series, seriesIndex, index, unit) {
  const rawX = point.x ?? `#${index + 1}`;
  const seriesName = series.name || `Series ${seriesIndex + 1}`;
  const rankFromPoint =
    point && point.rank !== null && point.rank !== undefined ? point.rank : null;
  const rankFromSeries = extractRank(seriesName, null);
  const rankFromX = extractRank(rawX, index + 1);
  let rankValue = rankFromPoint;
  if (rankValue === null || rankValue === undefined) {
    rankValue = rankFromSeries;
  }
  if (rankValue === null || rankValue === undefined) {
    rankValue = rankFromX;
  }
  element.dataset.series = seriesName;
  element.dataset.xLabel = String(rawX);
  element.dataset.yValue = String(point.y ?? 0);
  element.dataset.rank = String(rankValue);
  element.dataset.unit = unit || "";
}

// X축 라벨을 정리한다.
function formatChartLabel(value, index) {
  if (value === null || value === undefined || value === "") {
    return `#${index + 1}`;
  }
  return String(value);
}

// 숫자 값을 보기 좋게 포맷한다.
function formatChartValue(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const rounded = Math.round(value * 100) / 100;
  if (Number.isInteger(rounded)) {
    return String(rounded);
  }
  return rounded.toFixed(2);
}

// rank 정보를 추출한다.
function extractRank(rawValue, fallback) {
  if (typeof rawValue === "number" && Number.isFinite(rawValue)) {
    return rawValue;
  }
  if (typeof rawValue === "string") {
    const match = rawValue.match(/\d+/);
    if (match) {
      return Number(match[0]);
    }
  }
  return fallback !== undefined ? fallback : null;
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

// 존재하는 요소만 세션 정보를 반영한다.
function updateSessionUi() {
  if (sessionIdEl) {
    sessionIdEl.textContent = state.sessionId;
  }
  if (demoToggleEl) {
    demoToggleEl.checked = state.demoMode;
  }
  if (apiKeyInputEl) {
    apiKeyInputEl.value = state.apiKey;
  }
  if (modelInputEl) {
    modelInputEl.value = state.model;
  }
  if (baseUrlInputEl) {
    baseUrlInputEl.value = state.baseUrl;
  }
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

// 데모 모드 토글이 있을 때만 상태를 저장한다.
if (demoToggleEl) {
  demoToggleEl.addEventListener("change", () => {
    state.demoMode = demoToggleEl.checked;
    saveState();
  });
}

// 옵션 입력 필드가 있을 때만 상태를 저장한다.
if (apiKeyInputEl) {
  apiKeyInputEl.addEventListener("input", () => {
    state.apiKey = apiKeyInputEl.value.trim();
    saveState();
  });
}

if (modelInputEl) {
  modelInputEl.addEventListener("input", () => {
    state.model = modelInputEl.value.trim();
    saveState();
  });
}

if (baseUrlInputEl) {
  baseUrlInputEl.addEventListener("input", () => {
    state.baseUrl = baseUrlInputEl.value.trim();
    saveState();
  });
}

// 오른쪽 패널 토글을 처리한다.
function setInsightsHidden(hidden) {
  state.insightsHidden = hidden;
  if (appEl) {
    appEl.classList.toggle("is-insights-hidden", hidden);
  }
  if (insightsEl) {
    insightsEl.style.display = hidden ? "none" : "flex";
  }
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
  state.sessionId = createSessionId();
  state.messages = [];
  saveState();
  updateSessionUi();
  renderMessages();
}

// 버튼이 있을 때만 대화 초기화를 처리한다.
if (clearButtonEl) {
  clearButtonEl.addEventListener("click", () => {
    resetConversation();
  });
}

if (newChatButtonEl) {
  newChatButtonEl.addEventListener("click", () => {
    resetConversation();
  });
}

// 초기 로딩을 수행한다.
loadState();
updateSessionUi();
renderMessages();
resizeInput();
setInsightsHidden(state.insightsHidden);
