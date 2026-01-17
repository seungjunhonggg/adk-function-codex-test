# 에이전트/툴/메모리/컨텍스트 전략 (v0)

## 목표
- 10k 컨텍스트 한도에서 2k~3k 브리핑을 안정적으로 생성한다.
- 표/차트는 결정적 처리, LLM은 서술만 담당한다.
- 중간 단계 변경 요청을 빠르게 반영한다.

## 기본 아키텍처
### 단일 오케스트레이터 + 작업 함수
초기에는 단일 에이전트가 전체를 오케스트레이션한다.
- 장점: 컨텍스트 단순, 구현 속도 빠름.
- 단점: 긴 대화에서 컨텍스트 과밀 가능.

### 필요 시 멀티 에이전트 확장
컨텍스트가 길어질 때만 분리한다.
- DB 조회 요약 에이전트
- 최적화 API 요약 에이전트
- 브리핑 서술 에이전트
핵심은 역할이 아니라 컨텍스트 분리다.

## 툴 설계 (최소 세트)
### 1) simulation_run
설명: 1-1~1-7의 결정적 처리를 수행하고 표/차트 데이터를 반환.
입력:
- input_params (온도/전압/크기/용량/개발품여부/파우더사이즈)
- optional chip_type
- overrides (중간 단계 변경 시)
출력:
- tables (1-1~1-7)
- chart_data
- stage_status

### 2) simulation_explain
설명: 특정 단계의 상세 원인/근거 데이터 요청에 응답.
입력:
- stage_id (1-3/1-5/1-6/1-7)
- query (사용자 질문)
출력:
- detail_notes
- optional expanded_tables (children 지표 등)

### 3) label_mapping
설명: 영문 컬럼명을 한글 라벨로 변환.
입력:
- columns
출력:
- korean_labels

## 메모리 설계
### 세션 메모리 (필수)
- session_id
- input_params
- chip_type
- stage_outputs (tables, chart_data)
- last_updated_stage
- invalidation_rules

### 장기 메모리 (선택)
- 사용자 선호 입력값(기본 온도/전압 등)
- 선호 차트 타입

## 컨텍스트 예산
- LLM 입력 4k 이내 유지.
- 표/차트는 요약 형태로만 주입.
- top-k raw 배열은 저장만 하고 필요 시 재조회.

## 중간 단계 변경 규칙
- 1-1 변경: 1-2~1-7 재계산
- 1-3 변경: 1-4~1-7 재계산
- 1-5 변경(top-k k값 변경): 1-6~1-7 재계산

## 브리핑 생성 정책
- 서술은 표/차트에 있는 값만 인용.
- children 지표는 기본 숨김, 요청 시만 확장.
