# MCP 설계 초안 (v0)

## 목적
- DB 조회/최적화 API를 에이전트 툴로 안정적으로 제공한다.
- 툴 수는 최소화하고, 워크플로우 단위로 통합한다.

## 권장 도구 세트
1) `mlcc_simulation_run`
   - 1-1~1-7 실행 + 표/차트 반환
2) `mlcc_simulation_explain`
   - 특정 단계 상세 질의 응답
3) `mlcc_label_mapping`
   - 영문 컬럼 → 한글 라벨 변환

## 응답 형식
- text + structuredContent 병행
- 대량 데이터는 요약만 반환

## 보안/안정성
- readOnlyHint: true (조회성)
- idempotentHint: true
- destructiveHint: false
