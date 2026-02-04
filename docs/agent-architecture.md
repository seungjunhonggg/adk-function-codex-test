# 에이전트 구조 설계 (Google ADK)

## 목적
- MLCC 시뮬레이션 대화를 위한 최소 구조
- 브리핑 없이 짧은 대화형 응답
- 단계별 재실행(DAG dirty)과 gap 대응을 유연하게 처리

## 에이전트 계층
### 1) RootAgent (메인)
- 사용자 메시지 진입점
- 라우팅/커맨드 결정, 입력·변경 파싱
- SimulationAgent 실행 및 최종 응답 스트리밍

### 2) SimulationAgent (시뮬레이션, LlmAgent)
- 5개 단계 툴을 순차 호출
- 단계 완료마다 사용자 확인/수정 요청을 처리
- gap 발생 시 즉시 중단하고 질문을 반환

### 3) Tool Functions (단계 실행)
- 각 단계는 단일 툴 함수로 실행
- 결과는 요약만 stage_outputs에 저장
- raw 결과는 별도 ref로 보관

## LLM 역할
- RouteAgent: simulation vs casual 분기
- CommandAgent: run/reset 결정
- InputAgent: 입력 파라미터 추출
- UpdateAgent: 변경 요청 추출
- GapAgent: 데이터 공백 질문 생성
- SelectionAgent: 텍스트에서 후보 선택 추출
- SimulationReplyAgent: 짧은 최종 응답 생성
- CasualAgent: 일반 대화 응답

## 상태/메모리 구조
- 영속 상태: `state._get_session_state`로 로드/저장
- 런타임 상태: ADK session.state의 temp 키로 유지
- 핵심 필드
  - input_params, selections, configs, user_prefs
  - stage_status, stage_outputs, stage_notes
  - pending_action, last_gap, history

## 실행 흐름 요약
1. RootAgent가 상태 로드
2. RouteAgent로 casual/simulation 분기
3. simulation이면 CommandAgent로 run/reset 결정
4. Input/Update 파싱 후 dirty_stages 계산
5. SimulationAgent가 단계별 툴을 순차 호출
6. gap이면 질문 반환 + pending_action 저장
7. 완료 시 SimulationReplyAgent로 짧은 답변 생성

## 단계 설계 (5개 툴)
- 1-1 입력 수집(LLM): 온도/전압/크기/용량 4개 또는 chip_prod_id 1개
- 1-2 칩기종 후보 조회 (tool: `find_chip_prod_id`)
- 1-3 REF LOT 선정 (tool: `find_ref_lot_candidate`)
- 1-4 API payload/설계값 구성 (tool: `get_ref_lot_info`)
- 1-5 Top-K 선정 (tool: `run_grid`)
- 1-6 최근 6개월 유사 LOT + 불량률 요약(1-7 포함) (tool: `fetch_recent_defect_summary`)

## DAG dirty 규칙
- 변경된 입력/설정/선택에 따라 dirty_stages 계산
- dirty 단계부터 후속 단계를 재실행
- 중간 단계 변경도 전체 구조를 깨지 않고 재연산

## gap/pending 처리
- 데이터 공백(gap)이면 즉시 질문 반환
- 후보 선택이 필요하면 후보 ID 목록을 텍스트로 제공
- 사용자의 텍스트 선택을 SelectionAgent가 추출

## 스트리밍 규칙
- SSE는 `event: delta`로 텍스트만 전달
- 블록/테이블/차트 응답은 사용하지 않는다

## 확장 포인트
- 단계별 툴 DB 쿼리 및 근거 노트 확장
- pending_action 타입 확장(새 입력, 추가 선택)
- 장기 메모리(리포트/결과 캐시) 연동
