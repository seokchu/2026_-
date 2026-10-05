# 27. 제출 패키징 체크포인트 — 공식 horizon 검증 이후

작성: 2026-10-05

## 현재 판정

- 공식 forecast horizon 정합성: 완료. 보고서 26 근거로 현행 t+15/30/45/60분 유지.
- horizon 변경에 따른 모델 재설계: 불필요.
- 다만 코드 감사에서 평가와 배포 artifact 간 정합성 문제 2건을 수정했으므로 Baseline v1.1은 로컬 원본 데이터 환경에서 재실행해야 한다.
- 재실행 전 기존 joblib / baseline_summary.json / 생성 표·그림은 최종 제출본으로 확정하지 않는다.

## 이번 코드 감사에서 수정한 배포 정합성 문제

### P1. 조건부 conformal

평가 pipeline은 v1.1에서 현재수요 TRAIN 3분위별 조건부 conformal을 사용했지만, 기존 deployment_fit은 전역 conformal artifact를 저장했다.
run_baseline.py와 inference.py를 수정하여 배포 artifact도 평가와 동일한 level3 그룹 cut과 그룹별 Q를 저장·적용하도록 했다.

### P2. 기본 경보 정책

기존 inference.py는 q를 명시하지 않으면 threshold grid 최댓값을 사용했다. 이는 평가에서 선택된 기본 운용점과 달랐다.
이제 평가의 primary window / default cost ratio에서 선택된 policy, early_q, confirm_q, gate를 artifact에 저장하고 inference 기본값으로 사용한다.

### P3. 회귀 방지 테스트

test_baseline.py에 다음 검사를 추가했다.
- config가 level3 conformal이면 배포 artifact도 level3이어야 한다.
- 배포 기본 early_q / confirm_q / gate가 recommended_operating_point와 같아야 한다.

## 반드시 다시 실행할 명령

프로젝트 루트에서:

    python3 analysis/final_baseline/run_baseline.py --config analysis/final_baseline/config.yaml
    python3 analysis/final_baseline/test_baseline.py

그 후 git status와 생성 산출물을 확인한다.

## 재실행 후 제출 전 확인

1. 수용 테스트 전부 PASS.
2. baseline_summary.json의 git_commit이 실행 시점 HEAD와 일치.
3. model artifact가 level3 조건부 conformal과 선택된 기본 경보정책을 포함.
4. 보고서 23 수치와 23_baseline CSV가 일치.
5. 보고서 24·26의 정정사항이 최종 결과보고서에 반영.
6. day-ahead 96-step을 공식 요구사항이라고 쓰지 않음.
7. 원본 15분/30분/45분/60분 열을 미래 forecast horizon이라고 설명하지 않음.
8. 물리적 설비 조치나 실제 요금 절감액을 근거 없이 주장하지 않음.

## 제출 패키지 핵심 구성

- 최종 결과보고서: 문제정의 → 데이터 진단 → 다중시점 예측 → 신뢰도 → 경보정책 → 정보부족 진단 → 한계
- 코드: analysis/final_baseline 전체 + config + 수용테스트 + 실행환경
- 그림: 시스템 구조, actual/prediction interval, 신뢰도 밴드, 비용×선행 Pareto, CORE vs PUBLIC, 2×2 라우팅
- 근거: 보고서 17, 23, 24, 26 및 생성 테이블

## 현재 ChatGPT가 여기서 직접 끝낼 수 없는 부분

원본 KAMP CSV가 GitHub에 커밋되어 있지 않고 현재 연결 파일에도 원본 CSV 자체가 없으므로, 수정된 파이프라인의 전체 재실행은 로컬 원본 데이터가 있는 환경에서 해야 한다.
재실행 결과가 push되면 최종 보고서·코드 패키지 정합성 검수를 진행한다.