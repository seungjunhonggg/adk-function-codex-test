# 시뮬레이션(추천 → 예측) 워크플로우 가이드

## 목적
사용자가 인접기종 추천을 요청하면 **추천 파라미터 수집 → 추천 실행 → 결과 브리핑**까지 진행하고,  
사용자가 예측을 확인하면 **예측 시뮬레이션 실행 → 결과 UI 표시 → 브리핑**까지 이어집니다.

## 주요 구성 요소
- 에이전트
  - 오케스트레이터(`triage_agent`)
  - 시뮬레이션 에이전트(`simulation_agent`)
- 상태 저장소
  - `simulation_store` (추천 입력 파라미터 저장)
  - `recommendation_store` (추천 결과 + 30개 파라미터 저장)
- 도구/함수
  - `open_simulation_form`, `update_simulation_params`, `run_simulation`
  - `run_lot_simulation`, `run_prediction_simulation`
  - 내부 호출: `execute_simulation`, `call_simulation_api`, `call_prediction_api`
- UI 이벤트
  - `simulation_form`, `simulation_result`, `prediction_result`, `frontend_trigger`, `chat_message`

## 전체 흐름 (요약)
1) 사용자 → `/api/chat`  
2) 오케스트레이터 → `simulation_agent` 호출  
3) 파라미터 수집/입력 UI 오픈  
4) 추천 실행 → 추천 결과 UI 표시  
5) 사용자 확인(“네/응/예측해줘”) → 예측 실행  
6) 예측 결과 UI 표시 + 브리핑

## 추천 단계 상세
### 1) 사용자 요청
- 오케스트레이터가 “추천/시뮬레이션” 의도로 판단하면 `simulation_agent`를 호출합니다.

### 2) 입력 UI 열기
- `simulation_agent` → `open_simulation_form`  
  - 현재 `simulation_store` 상태로 `simulation_form` 이벤트를 보냅니다.

### 3) 파라미터 수집
필수 파라미터(5개):
`temperature`, `voltage`, `size`, `capacity`, `production_mode`

입력 경로:
- **대화 입력**: `update_simulation_params(message=...)`가 텍스트에서 값 추출
- **UI 입력**: `/api/simulation/params`로 값 반영 (실행은 하지 않음)

### 4) 추천 실행
실행 트리거:
- UI에서 “추천 실행” 버튼 클릭 → `/api/simulation/run`
- 또는 에이전트가 `run_simulation` 호출

내부 동작:
1) `execute_simulation` 호출  
2) `call_simulation_api` 실행  
   - `SIM_API_URL`이 있으면 API 호출  
   - 없으면 로컬 시뮬레이션
3) 결과를 `recommendation_store`에 저장  
   - `awaiting_prediction = True`  
   - `recommended_model`, `representative_lot`, `params(param1~param30)`
4) `simulation_result` 이벤트로 UI 갱신  
5) 자동 브리핑 생성(`auto_message_agent`) → `chat_message`

## 추천 결과 파라미터 수정
추천 결과의 30개 파라미터는 UI에서 수정 가능:
- UI “파라미터 반영” 버튼 → `/api/recommendation/params`
- `recommendation_store.update_params`로 저장
- 다시 `simulation_result` 이벤트를 보내 UI를 최신 상태로 갱신
- 예측은 자동 실행되지 않음

## 예측 단계 상세
### 1) 예측 실행 조건
예측은 **명시적 요청**이 있을 때만 실행:
- 사용자가 “예측/시뮬레이션/forecast” 등의 키워드를 말함
- 또는 `awaiting_prediction=True` 상태에서 짧은 긍정(“네/응/예”)

### 2) 예측 실행
`run_prediction_simulation` 내부 동작:
1) `recommendation_store`에서 `param1~param30` 확인
2) 누락 시 에러 반환 + UI 트리거
3) `call_prediction_api` 실행  
   - `PREDICT_API_URL` 있으면 API 호출  
   - 없으면 로컬 예측
4) 결과를 `prediction_result` 이벤트로 UI 출력
5) `awaiting_prediction=False`로 상태 종료

## UI 이벤트 흐름
- `simulation_form` → 입력 패널 표시
- `simulation_result` → 추천 결과 카드 표시
- `prediction_result` → 예측 결과 카드 표시
- `frontend_trigger` → 상태/오류 메시지 표시

## 실패/예외 처리
- 추천 입력 누락 → `missing` 반환 + 입력 패널 강조
- 추천 결과 없음 → 예측 실행 불가 메시지
- 예측 파라미터(30개) 누락 → 실행 차단 및 누락 항목 안내
- API 실패 → 로컬 로직 fallback(환경변수 미설정 시)

## 커스터마이징 포인트
- 추천 입력 항목 변경: `extract_simulation_params`, `simulation_store`
- 추천 결과 구조 변경: `call_simulation_api` 반환 구조
- 예측 파라미터 개수 변경: `run_prediction_simulation` 검증 로직
- UI 표시 카드 구조 변경: `frontend/app.js`
