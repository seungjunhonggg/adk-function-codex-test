# 상태 스키마 (확장안 v0)

## 목적
- 사용자가 어느 단계에서든 변경 요청을 해도 최소 재실행으로 처리한다.
- 긴 raw 결과는 저장만 하고, 요약만 컨텍스트에 주입한다.
- 값 없는 변경 요청은 pending으로 보류하고 재질문한다.

## 세션 상태 구조
```json
{
  "session_id": "string",
  "input_params": {
    "temperature": "string|null",
    "voltage": "string|null",
    "size": "string|null",
    "capacity": "string|null",
    "dev_flag": "string|null",
    "powder_size": "string|null",
    "chip_type": "string|null"
  },
  "selections": {
    "chip_type_id": "string|null",
    "reference_lot_id": "string|null"
  },
  "configs": {
    "top_k": 5,
    "top_k_sort": "rank",
    "core_match_params": [
      "active_powder_base",
      "active_powder_additives",
      "ldn_avr_value",
      "cast_dsgn_thk"
    ]
  },
  "stage_status": {
    "1-1": {"done": true, "dirty": false, "updated_at": "ts"},
    "1-2": {"done": false, "dirty": true, "updated_at": "ts"},
    "1-3": {"done": false, "dirty": true, "updated_at": "ts"},
    "1-4": {"done": false, "dirty": true, "updated_at": "ts"},
    "1-5": {"done": false, "dirty": true, "updated_at": "ts"},
    "1-6": {"done": false, "dirty": true, "updated_at": "ts"},
    "1-7": {"done": false, "dirty": true, "updated_at": "ts"},
    "1-8": {"done": false, "dirty": true, "updated_at": "ts"}
  },
  "stage_outputs": {
    "tables": {
      "input_params_table": [],
      "chip_type_candidates_table": [],
      "reference_lot_candidates_table": [],
      "reference_lot_table": [],
      "top_k_table": [],
      "recent_similar_table": [],
      "defect_rate_table": []
    },
    "charts": [],
    "briefing_blocks": []
  },
  "stage_notes": {
    "1-1": "string",
    "1-2": "string",
    "1-3": "string",
    "1-4": "string",
    "1-5": "string",
    "1-6": "string",
    "1-7": "string",
    "1-8": "string"
  },
  "raw_refs": {
    "top_k_raw_id": "string|null",
    "defect_raw_id": "string|null"
  },
  "pending_action": {
    "action": "collect_input",
    "target_stage": "1-1",
    "missing_fields": ["temperature"],
    "requested_at": "ts"
  },
  "last_explain_stage": "string|null",
  "history": [
    {"action": "update_input", "payload": {}, "at": "ts"}
  ],
  "user_prefs": {
    "chart_type": "bar|line|scatter",
    "language": "ko"
  },
  "last_error": {
    "stage": "1-4",
    "message": "string"
  }
}
```

## pending_action 규칙
- 값 없는 변경 요청이면 pending_action을 만든다.
- pending_action이 있으면 **자동 재실행하지 않고** 재질문한다.
- 사용자가 값을 주면 pending_action을 해소하고 해당 단계부터 재실행한다.
- action 예시:
  - collect_input
  - update_reference_lot
  - update_chip_type
  - update_top_k
  - update_chart_type
  - update_input_params

## 입력 병합 규칙
- 새 입력값이 있으면 기존 값을 덮어쓴다.
- 새 입력값이 비어 있으면 기존 값을 유지한다.

## 무효화 규칙 (DAG)
- 1-1 변경 → 1-2~1-7 dirty
- 1-2 변경 → 1-3~1-7 dirty
- 1-3 변경 → 1-4~1-7 dirty
- 1-5(k 변경/필터 변경) → 1-6~1-7 dirty

## 저장 원칙
- raw 결과는 raw_refs로만 저장하고, 요약만 stage_outputs에 저장한다.
- 브리핑은 stage_outputs.briefing_blocks로 저장한다.
- 단계 근거 요약은 stage_notes에 저장한다.
- 설명 후속 처리를 위해 last_explain_stage를 저장한다.
