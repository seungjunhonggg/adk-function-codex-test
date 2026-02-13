/**
 * A2UI v0.10 경량 렌더러 (vanilla JS)
 *
 * Google A2UI 프로토콜의 server-to-client 메시지를 받아
 * DOM 트리로 변환한다. 기존 form CSS 클래스를 재활용한다.
 */

// eslint-disable-next-line no-unused-vars
const A2UIRenderer = (() => {
  "use strict";

  /** @type {Record<string, A2UISurface>} */
  const surfaces = {};

  // -----------------------------------------------------------------------
  // JSON Pointer 헬퍼 (단일 레벨만 지원: "/key")
  // -----------------------------------------------------------------------

  function getByPath(obj, path) {
    if (!path || !path.startsWith("/")) return undefined;
    const key = path.slice(1);
    return obj[key];
  }

  function setByPath(obj, path, value) {
    if (!path || !path.startsWith("/")) return;
    const key = path.slice(1);
    obj[key] = value;
  }

  // -----------------------------------------------------------------------
  // 메시지 핸들러
  // -----------------------------------------------------------------------

  function handleMessage(msg) {
    if (msg.createSurface) return _createSurface(msg.createSurface);
    if (msg.updateComponents) return _updateComponents(msg.updateComponents);
    if (msg.updateDataModel) return _updateDataModel(msg.updateDataModel);
    if (msg.deleteSurface) return _deleteSurface(msg.deleteSurface);
    return null;
  }

  function _createSurface(payload) {
    const { surfaceId, catalogId, sendDataModel } = payload;
    const rootEl = document.createElement("div");
    rootEl.className = "a2ui-surface";
    rootEl.dataset.surfaceId = surfaceId;

    surfaces[surfaceId] = {
      surfaceId,
      catalogId,
      sendDataModel: !!sendDataModel,
      components: [],
      dataModel: {},
      rootEl,
      onAction: null,
      // dataPath → DOM input 요소 매핑 (검증용)
      _inputEls: {},
    };
    return rootEl;
  }

  function _updateComponents(payload) {
    const surface = surfaces[payload.surfaceId];
    if (!surface) return null;
    surface.components = payload.components;
    _render(surface);
    return null;
  }

  function _updateDataModel(payload) {
    const surface = surfaces[payload.surfaceId];
    if (!surface) return null;
    if (!payload.path || payload.path === "/") {
      surface.dataModel = payload.value || {};
    } else {
      setByPath(surface.dataModel, payload.path, payload.value);
    }
    // 데이터 모델이 나중에 도착할 수 있으므로 리렌더
    _render(surface);
    return null;
  }

  function _deleteSurface(payload) {
    const surface = surfaces[payload.surfaceId];
    if (!surface) return null;
    if (surface.rootEl.parentNode) {
      surface.rootEl.parentNode.removeChild(surface.rootEl);
    }
    delete surfaces[payload.surfaceId];
    return null;
  }

  // -----------------------------------------------------------------------
  // 렌더링
  // -----------------------------------------------------------------------

  function _render(surface) {
    surface.rootEl.innerHTML = "";
    surface._inputEls = {};

    const componentMap = {};
    surface.components.forEach((c) => {
      componentMap[c.id] = c;
    });

    // root 컴포넌트 찾기 (id === "root" 우선, 없으면 childIds에 포함되지 않은 첫 번째)
    let rootComp = componentMap["root"];
    if (!rootComp) {
      const childSet = new Set();
      surface.components.forEach((c) => {
        (c.childIds || []).forEach((id) => childSet.add(id));
      });
      rootComp = surface.components.find((c) => !childSet.has(c.id));
    }
    if (rootComp) {
      surface.rootEl.appendChild(
        _renderComponent(rootComp, componentMap, surface),
      );
    }
  }

  function _renderComponent(comp, componentMap, surface) {
    switch (comp.type) {
      case "Row":
        return _renderRow(comp, componentMap, surface);
      case "Column":
        return _renderColumn(comp, componentMap, surface);
      case "Text":
        return _renderText(comp);
      case "TextField":
        return _renderTextField(comp, surface);
      case "ChoicePicker":
        return _renderChoicePicker(comp, surface);
      case "Button":
        return _renderButton(comp, surface);
      case "Divider":
        return _renderDivider();
      case "Card":
        return _renderCard(comp, componentMap, surface);
      default: {
        const el = document.createElement("div");
        el.textContent = "[A2UI: unknown " + comp.type + "]";
        return el;
      }
    }
  }

  // -----------------------------------------------------------------------
  // 컨테이너 렌더러
  // -----------------------------------------------------------------------

  function _renderChildren(comp, componentMap, surface) {
    const frag = document.createDocumentFragment();
    (comp.childIds || []).forEach((childId) => {
      const child = componentMap[childId];
      if (child) {
        frag.appendChild(_renderComponent(child, componentMap, surface));
      }
    });
    return frag;
  }

  function _renderRow(comp, componentMap, surface) {
    const el = document.createElement("div");
    el.className = "form-box-container";
    el.appendChild(_renderChildren(comp, componentMap, surface));
    return el;
  }

  function _renderColumn(comp, componentMap, surface) {
    const el = document.createElement("div");
    el.className = "input-form-card";
    el.appendChild(_renderChildren(comp, componentMap, surface));
    return el;
  }

  function _renderCard(comp, componentMap, surface) {
    const el = document.createElement("div");
    el.className = "input-form-card";
    el.appendChild(_renderChildren(comp, componentMap, surface));
    return el;
  }

  // -----------------------------------------------------------------------
  // 리프 렌더러
  // -----------------------------------------------------------------------

  function _renderText(comp) {
    const el = document.createElement("div");
    if (comp.variant === "h5") {
      el.className = "form-box-header";
      const label = document.createElement("div");
      label.className = "form-box-label";
      // "방법 N" 추출
      const match = (comp.content || "").match(/^(방법\s*\d+)/);
      label.textContent = match ? match[1] : "";
      const title = document.createElement("div");
      title.className = "form-box-title";
      // "— " 이후 텍스트를 제목으로
      const dashIdx = (comp.content || "").indexOf("—");
      title.textContent =
        dashIdx >= 0 ? comp.content.slice(dashIdx + 1).trim() : comp.content;
      el.appendChild(label);
      el.appendChild(title);
    } else {
      el.className = "a2ui-text";
      el.textContent = comp.content || "";
    }
    return el;
  }

  function _renderTextField(comp, surface) {
    const group = document.createElement("div");
    group.className = "form-group";
    group.dataset.dataPath = comp.dataPath || "";

    const label = document.createElement("label");
    label.className = "form-label";
    label.textContent = comp.label || "";
    group.appendChild(label);

    const input = document.createElement("input");
    input.className = "form-input";
    input.type = comp.inputType || "text";
    input.name = comp.id;
    if (comp.placeholder) input.placeholder = comp.placeholder;
    input.value = getByPath(surface.dataModel, comp.dataPath) || "";

    input.addEventListener("input", () => {
      setByPath(surface.dataModel, comp.dataPath, input.value);
    });
    group.appendChild(input);

    // 에러 텍스트 (숨겨진 상태)
    const errorText = document.createElement("div");
    errorText.className = "form-error-text";
    errorText.textContent = "입력이 필요합니다";
    group.appendChild(errorText);

    // 검증용 매핑 저장
    if (comp.dataPath) {
      surface._inputEls[comp.dataPath] = { input, errorText };
    }

    return group;
  }

  function _renderChoicePicker(comp, surface) {
    const group = document.createElement("div");
    group.className = "form-group";
    group.dataset.dataPath = comp.dataPath || "";

    const label = document.createElement("label");
    label.className = "form-label";
    label.textContent = comp.label || "";
    group.appendChild(label);

    const select = document.createElement("select");
    select.className = "form-input";
    select.name = comp.id;

    const currentVal = getByPath(surface.dataModel, comp.dataPath) || "";

    // 빈 값이면 placeholder 옵션 추가
    if (!currentVal) {
      const placeholder = document.createElement("option");
      placeholder.textContent = "선택해주세요";
      placeholder.value = "";
      placeholder.disabled = true;
      placeholder.selected = true;
      select.appendChild(placeholder);
    }

    (comp.options || []).forEach((opt) => {
      const option = document.createElement("option");
      option.value = opt.value;
      option.textContent = opt.label;
      if (opt.value === currentVal) option.selected = true;
      select.appendChild(option);
    });

    select.addEventListener("change", () => {
      setByPath(surface.dataModel, comp.dataPath, select.value);
    });
    group.appendChild(select);

    // 에러 텍스트
    const errorText = document.createElement("div");
    errorText.className = "form-error-text";
    errorText.textContent = "선택이 필요합니다";
    group.appendChild(errorText);

    if (comp.dataPath) {
      surface._inputEls[comp.dataPath] = { input: select, errorText };
    }

    return group;
  }

  function _renderButton(comp, surface) {
    const actions = document.createElement("div");
    actions.className = "form-actions";

    const btn = document.createElement("button");
    btn.className = "form-submit-btn";
    btn.textContent = comp.label || "Submit";
    btn.type = "button";

    btn.addEventListener("click", () => {
      if (!surface.onAction) return;
      const eventInfo = (comp.action && comp.action.event) || {};
      const actionPayload = {
        version: "v0.10",
        action: {
          name: eventInfo.name || "",
          surfaceId: surface.surfaceId,
          sourceComponentId: comp.id,
          timestamp: new Date().toISOString(),
          context: surface.sendDataModel
            ? { dataModel: Object.assign({}, surface.dataModel) }
            : eventInfo.context || {},
        },
      };
      surface.onAction(actionPayload);
    });

    actions.appendChild(btn);
    return actions;
  }

  function _renderDivider() {
    const el = document.createElement("div");
    el.className = "form-or-divider";
    const text = document.createElement("span");
    text.className = "form-or-text";
    text.textContent = "또는";
    el.appendChild(text);
    return el;
  }

  // -----------------------------------------------------------------------
  // 검증 헬퍼
  // -----------------------------------------------------------------------

  /**
   * 주어진 dataPath 목록에 대해 필수 값 검증을 수행한다.
   * 빈 값이면 에러 스타일을 표시하고 false를 반환한다.
   */
  function validateRequired(surfaceId, requiredPaths) {
    const surface = surfaces[surfaceId];
    if (!surface) return false;

    let isValid = true;

    // 먼저 모든 에러 초기화
    Object.values(surface._inputEls).forEach(({ input, errorText }) => {
      input.classList.remove("has-error");
      if (errorText) errorText.classList.remove("is-visible");
    });

    requiredPaths.forEach((path) => {
      const val = getByPath(surface.dataModel, path);
      if (!val || !String(val).trim()) {
        isValid = false;
        const els = surface._inputEls[path];
        if (els) {
          els.input.classList.add("has-error");
          if (els.errorText) els.errorText.classList.add("is-visible");
        }
      }
    });

    return isValid;
  }

  /**
   * surface를 제출 완료 상태로 만든다.
   */
  function markSubmitted(surfaceId) {
    const surface = surfaces[surfaceId];
    if (!surface) return;
    surface.rootEl.classList.add("is-submitted");
    surface.rootEl.style.opacity = "0.7";
    surface.rootEl.style.pointerEvents = "none";
    // 버튼 비활성화
    surface.rootEl.querySelectorAll(".form-submit-btn").forEach((btn) => {
      btn.disabled = true;
      btn.textContent = "처리 중...";
    });
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  return {
    surfaces,
    handleMessage,
    validateRequired,
    markSubmitted,
  };
})();
