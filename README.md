# 공정 모니터링 에이전트 데모

이 데모는 **OpenAI Agents SDK**를 활용해 두 가지 워크플로우를 보여줍니다.

1) **공정 데이터 조회**: DB 조회 에이전트가 SQLite를 조회하고, 함수 도구가 프론트엔드로 트리거 이벤트를 전송합니다.
2) **예측 시뮬레이션**: 4가지 입력(온도/전압/크기/용량)을 수집한 뒤 시뮬레이션 API(또는 로컬 스텁) 호출 → 결과를 UI 트리거로 전송합니다.

추가로, **워크플로우 빌더 페이지**에서 노드 기반으로 에이전트 흐름을 시각화할 수 있습니다.

## 전체 구조

```
사용자 → /api/chat → 분류 에이전트
                 ├─ DB 에이전트 → get_process_data → WebSocket 이벤트 → UI 테이블
                 └─ 시뮬레이션 에이전트 → update_simulation_params → run_simulation → WebSocket 이벤트 → UI 카드
```

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# OPENAI_API_KEY 설정 후 실행
uvicorn backend.app:app --reload
```

브라우저에서 `http://localhost:8000` 접속하세요.

## 환경 변수

- `OPENAI_API_KEY` : Agents SDK 호출에 필요
- `OPENAI_MODEL` : 기본 모델 (기본값: gpt-4.1-mini)
- `SIM_API_URL` : 외부 시뮬레이션 API가 있을 때 POST 대상
- `SESSION_DB_PATH` : 에이전트 세션 저장용 SQLite (기본값: sessions.db)
- `DB_PATH` : 공정 데이터 SQLite (기본값: process_data.db)

## 간이 테스트 (API 키 없이)

에이전트 호출 없이 **함수 도구 → 프론트 트리거**만 검증할 수 있습니다.

1. UI 좌측의 **“간이 테스트 실행”** 버튼을 클릭
2. `/api/test/trigger`가 호출되어 DB 조회 및 시뮬레이션 트리거를 전송
3. 이벤트 패널에서 결과 렌더링 확인

또는 직접 호출할 수 있습니다.

```bash
curl -X POST http://localhost:8000/api/test/trigger ^
  -H \"Content-Type: application/json\" ^
  -d \"{\\\"session_id\\\":\\\"demo\\\"}\"
```

## 워크플로우 빌더 페이지

`http://localhost:8000/builder` 에서 노드 기반 데모를 확인할 수 있습니다.
- 왼쪽 팔레트에서 노드 추가
- 드래그로 위치 이동
- 포트를 클릭해 연결선 생성
- 오른쪽 패널에서 JSON으로 내보내기 가능

## 폴더 구조

```
backend/
  app.py           # FastAPI 엔트리, WebSocket, API 라우팅
  agents.py        # 분류/DB/시뮬레이션 에이전트 구성
  tools.py         # 함수 도구 (DB 조회, 시뮬레이션, 트리거)
  db.py            # SQLite 스키마 및 샘플 데이터
  simulation.py    # 시뮬레이션 스토어 + 로컬 스텁
  config.py        # 환경 변수 설정
frontend/
  index.html       # ChatGPT 스타일 UI
  styles.css       # 공통 스타일
  app.js           # 프론트 동작 및 WebSocket 처리
  builder.html     # 워크플로우 빌더 페이지
  builder.css      # 빌더 전용 스타일
  builder.js       # 빌더 인터랙션
```

## 참고
- Agents SDK 공식 문서: https://openai.github.io/openai-agents-python/
- SDK 리포지토리: https://github.com/openai/openai-agents-python
