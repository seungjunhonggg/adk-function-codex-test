# 데모 모드 설계 (v0)

## 목적
- DB 연결 없이도 전체 워크플로우를 테스트한다.
- 표/차트/브리핑이 실제처럼 보이도록 구성한다.

## 원칙
- 실제 스키마와 동일한 필드명/형식을 사용한다.
- 시드 고정 랜덤으로 재현 가능하게 생성한다.
- 값 범위는 현실적인 구간으로 제한한다.

## 구성
1. 입력값 더미
   - temperature, voltage, size, capacity, dev_flag, powder_size
2. 칩기종 더미
   - 4~5개 후보를 고정 목록으로 제공
3. 레퍼런스 LOT 더미
   - mdh_base_view_total_4 스키마 일부를 모사
   - 최종 선정 1개 LOT만 제공
   - 후보 LOT 10개를 함께 제공
4. 최적화 API 더미 응답
   - datas.ref, datas.sim만 포함
   - sim은 rank 순으로 k개 생성
5. 최근 6개월 유사 설계 더미
   - design_input_date 범위를 최근 6개월로 설정
   - candidate_rank 기준 4~5개 제공
6. 불량률 지표 더미
   - report_defect_groups/children 설정을 그대로 사용
   - candidate_rank 기준 4~5개 제공
   - metric은 전부 포함

## 출력 제약
- 브리핑은 한글 라벨만 사용한다.
- 표/차트는 결정적 처리로 생성한다.
- 초기 데모는 입력 요약 표 + top-k 표를 채운다.
- top-k는 4~5개 행을 기본으로 채운다.
- 나머지 표/차트는 빈 값으로 두고 점진적으로 확장한다.
