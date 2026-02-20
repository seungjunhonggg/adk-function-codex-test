// EChart 렌더러 모듈 — ECharts option JSON을 받아 인터랙티브 차트를 렌더링한다.
const EChartRenderer = (function () {
  // 활성 차트 인스턴스를 관리한다.
  const _instances = new Map();
  let _idCounter = 0;

  // 테마 색상 팔레트
  const PALETTE = [
    "#0d6c63",
    "#b5842f",
    "#3a7bd5",
    "#e06c5f",
    "#6bb36b",
    "#8e6cba",
    "#2aa198",
    "#d4843e",
    "#5c7cba",
    "#c75c76",
  ];

  // 기본 인터랙티브 옵션을 병합한다.
  function _applyDefaults(option) {
    var merged = Object.assign({}, option);

    // 색상 팔레트
    if (!merged.color) {
      merged.color = PALETTE;
    }

    // 툴팁 기본 설정
    if (!merged.tooltip) {
      merged.tooltip = {};
    }
    if (merged.tooltip.trigger === undefined) {
      merged.tooltip.trigger = "axis";
    }
    Object.assign(merged.tooltip, {
      backgroundColor: "rgba(255,255,255,0.96)",
      borderColor: "rgba(15,18,17,0.12)",
      borderWidth: 1,
      textStyle: Object.assign(
        { color: "#1d1f1e", fontSize: 12 },
        merged.tooltip.textStyle || {}
      ),
      extraCssText:
        "box-shadow: 0 8px 24px rgba(20,20,20,0.12); border-radius: 8px; padding: 10px 14px;",
    });

    // 축 교차선
    if (merged.tooltip.trigger === "axis" && !merged.tooltip.axisPointer) {
      merged.tooltip.axisPointer = {
        type: "cross",
        lineStyle: { color: "rgba(13,108,99,0.25)", type: "dashed" },
        crossStyle: { color: "rgba(13,108,99,0.25)" },
        label: {
          backgroundColor: "rgba(13,108,99,0.85)",
          borderRadius: 4,
          padding: [4, 8],
          fontSize: 11,
        },
      };
    }

    // 범례 기본 설정
    if (merged.legend === undefined && _hasMultipleSeries(merged)) {
      merged.legend = {};
    }
    if (merged.legend && typeof merged.legend === "object") {
      Object.assign(merged.legend, {
        type: "scroll",
        bottom: 0,
        textStyle: { fontSize: 12, color: "#6f6b64" },
        pageTextStyle: { color: "#6f6b64" },
        icon: "roundRect",
        itemWidth: 14,
        itemHeight: 10,
        itemGap: 16,
      });
    }

    // 그리드 기본 설정
    if (!merged.grid) {
      merged.grid = {};
    }
    Object.assign(
      merged.grid,
      {
        left: "3%",
        right: "4%",
        bottom: merged.legend ? "14%" : "3%",
        top: merged.title ? "15%" : "8%",
        containLabel: true,
      },
      merged.grid
    );

    // 툴박스 (줌, 저장, 리셋)
    if (!merged.toolbox) {
      merged.toolbox = {
        show: true,
        orient: "horizontal",
        right: 16,
        top: 8,
        itemSize: 14,
        itemGap: 10,
        iconStyle: {
          borderColor: "#9aa0a6",
          borderWidth: 1.2,
        },
        emphasis: {
          iconStyle: { borderColor: "#0d6c63" },
        },
        feature: {
          dataZoom: { show: true, title: { zoom: "영역 확대", back: "되돌리기" } },
          restore: { show: true, title: "초기화" },
          saveAsImage: {
            show: true,
            title: "이미지 저장",
            pixelRatio: 2,
            backgroundColor: "#fff",
          },
        },
      };
    }

    // 데이터 줌 (x축 슬라이더)
    if (!merged.dataZoom && _hasAxis(merged, "xAxis")) {
      merged.dataZoom = [
        {
          type: "inside",
          xAxisIndex: 0,
          filterMode: "none",
        },
        {
          type: "slider",
          xAxisIndex: 0,
          height: 20,
          bottom: merged.legend ? "8%" : 4,
          borderColor: "rgba(13,108,99,0.15)",
          fillerColor: "rgba(13,108,99,0.08)",
          handleStyle: { color: "#0d6c63", borderColor: "#0d6c63" },
          textStyle: { fontSize: 10, color: "#6f6b64" },
          filterMode: "none",
        },
      ];
      // 그리드 하단 여유를 늘린다
      if (merged.legend) {
        merged.grid.bottom = "22%";
      } else {
        merged.grid.bottom = "14%";
      }
    }

    // 축 스타일 기본값
    _styleAxis(merged, "xAxis");
    _styleAxis(merged, "yAxis");

    // 시리즈별 기본 인터랙티브 설정
    if (Array.isArray(merged.series)) {
      merged.series.forEach(function (s) {
        // 라인 차트: 부드러운 곡선 + 포인트 강조
        if (s.type === "line") {
          if (s.smooth === undefined) s.smooth = true;
          if (!s.emphasis) s.emphasis = {};
          s.emphasis.focus = s.emphasis.focus || "series";
          if (!s.lineStyle) s.lineStyle = {};
          s.lineStyle.width = s.lineStyle.width || 2.5;
          // 영역 채우기
          if (s.areaStyle === undefined) {
            s.areaStyle = {
              opacity: 0.06,
            };
          }
        }
        // 바 차트: 둥근 모서리 + 강조
        if (s.type === "bar") {
          if (!s.itemStyle) s.itemStyle = {};
          if (s.itemStyle.borderRadius === undefined) {
            s.itemStyle.borderRadius = [4, 4, 0, 0];
          }
          if (!s.emphasis) s.emphasis = {};
          s.emphasis.focus = s.emphasis.focus || "series";
        }
        // 파이 차트: 강조
        if (s.type === "pie") {
          if (!s.emphasis) s.emphasis = {};
          s.emphasis.focus = s.emphasis.focus || "self";
          if (!s.emphasis.itemStyle) s.emphasis.itemStyle = {};
          s.emphasis.itemStyle.shadowBlur =
            s.emphasis.itemStyle.shadowBlur || 10;
          s.emphasis.itemStyle.shadowOffsetX =
            s.emphasis.itemStyle.shadowOffsetX || 0;
          s.emphasis.itemStyle.shadowColor =
            s.emphasis.itemStyle.shadowColor || "rgba(0,0,0,0.15)";
          // 라벨 포맷
          if (!s.label) s.label = {};
          if (!s.label.formatter) {
            s.label.formatter = "{b}: {d}%";
          }
        }
        // 산점도: 강조
        if (s.type === "scatter") {
          if (!s.emphasis) s.emphasis = {};
          s.emphasis.focus = s.emphasis.focus || "series";
          if (s.symbolSize === undefined) s.symbolSize = 8;
        }
      });
    }

    // 애니메이션 기본 설정
    if (merged.animation === undefined) {
      merged.animation = true;
      merged.animationDuration = 800;
      merged.animationEasing = "cubicInOut";
    }

    return merged;
  }

  // 여러 시리즈가 있는지 확인한다.
  function _hasMultipleSeries(option) {
    return Array.isArray(option.series) && option.series.length > 1;
  }

  // 특정 축이 있는지 확인한다.
  function _hasAxis(option, axisKey) {
    return option[axisKey] !== undefined;
  }

  // 축 스타일을 적용한다.
  function _styleAxis(option, axisKey) {
    var axis = option[axisKey];
    if (!axis) return;
    var axes = Array.isArray(axis) ? axis : [axis];
    axes.forEach(function (a) {
      if (!a.axisLine) a.axisLine = {};
      a.axisLine.lineStyle = Object.assign(
        { color: "rgba(15,18,17,0.15)" },
        a.axisLine.lineStyle || {}
      );
      if (!a.axisTick) a.axisTick = {};
      a.axisTick.lineStyle = Object.assign(
        { color: "rgba(15,18,17,0.1)" },
        a.axisTick.lineStyle || {}
      );
      if (!a.axisLabel) a.axisLabel = {};
      a.axisLabel = Object.assign(
        { fontSize: 11, color: "#6f6b64" },
        a.axisLabel
      );
      if (!a.splitLine) a.splitLine = {};
      a.splitLine.lineStyle = Object.assign(
        { color: "rgba(13,108,99,0.06)", type: "dashed" },
        a.splitLine.lineStyle || {}
      );
    });
  }

  // 차트 DOM 컨테이너를 생성한다.
  function _createContainer(chartId) {
    var wrapper = document.createElement("div");
    wrapper.className = "echart-wrapper";
    wrapper.dataset.chartId = chartId;

    var chartDiv = document.createElement("div");
    chartDiv.className = "echart-container";
    chartDiv.id = "echart-" + chartId;
    wrapper.appendChild(chartDiv);

    return { wrapper: wrapper, chartDiv: chartDiv };
  }

  // 차트를 렌더링한다.
  function render(container, option, chartId) {
    if (!window.echarts) {
      console.warn("ECharts library not loaded");
      var fallback = document.createElement("div");
      fallback.className = "echart-fallback";
      fallback.textContent = "차트 라이브러리를 불러올 수 없습니다.";
      container.appendChild(fallback);
      return null;
    }

    var id = chartId || "echart-" + ++_idCounter;
    var elements = _createContainer(id);
    container.appendChild(elements.wrapper);

    // 머지된 옵션 적용
    var mergedOption = _applyDefaults(option);

    // ECharts 인스턴스 생성 (약간의 딜레이로 DOM이 렌더된 후 초기화)
    requestAnimationFrame(function () {
      var chart = echarts.init(elements.chartDiv, null, {
        renderer: "canvas",
      });
      chart.setOption(mergedOption);

      // 인스턴스 저장
      _instances.set(id, {
        chart: chart,
        option: mergedOption,
        container: elements.chartDiv,
      });

      // 리사이즈 관찰
      var resizeObserver = new ResizeObserver(function () {
        chart.resize({ animation: { duration: 200, easing: "cubicInOut" } });
      });
      resizeObserver.observe(elements.wrapper);

      // 클릭 이벤트 (데이터 포인트 클릭 시 콘솔 로그)
      chart.on("click", function (params) {
        console.log("[EChart click]", params.name, params.value, params);
      });
    });

    return id;
  }

  // 기존 차트의 옵션을 업데이트한다.
  function update(chartId, newOption) {
    var instance = _instances.get(chartId);
    if (!instance) return;
    var mergedOption = _applyDefaults(newOption);
    instance.chart.setOption(mergedOption, { notMerge: false });
    instance.option = mergedOption;
  }

  // 차트를 제거한다.
  function dispose(chartId) {
    var instance = _instances.get(chartId);
    if (!instance) return;
    instance.chart.dispose();
    _instances.delete(chartId);
  }

  // 모든 차트를 리사이즈한다.
  function resizeAll() {
    _instances.forEach(function (inst) {
      inst.chart.resize({ animation: { duration: 200, easing: "cubicInOut" } });
    });
  }

  // 윈도우 리사이즈 이벤트
  window.addEventListener("resize", function () {
    resizeAll();
  });

  return {
    render: render,
    update: update,
    dispose: dispose,
    resizeAll: resizeAll,
    instances: _instances,
  };
})();
