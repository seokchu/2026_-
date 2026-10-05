# 저계측 제조환경을 위한 신뢰도 기반 다중시점 최대수요전력 사전경보 및 정보부족 진단

제6회 K-인공지능 제조데이터 분석 경진대회 — KAMP 「자원 최적화 AI 데이터셋」(2021)

집계 전력 이력만 있는 공장에서 **t+15/30/45/60분 수요를 예측**하고, 각 예측의 **신뢰도**와
**운영상 피크 초과 확률**을 함께 산출해, 목표시각을 추적하는 **순차 경보**로 올리며, 불확실성의 원인을
현재 변수로 지목할 수 없는 구간을 **정보부족**으로 분류해 작업자에게 넘긴다.
설비 제어·조치 지시는 하지 않는다.

## 최종 결과보고서

**→ [`analysis/reports/29_FINAL_SUBMISSION_REPORT.md`](analysis/reports/29_FINAL_SUBMISSION_REPORT.md)** (제출용, 여기부터 읽는다)
- 최근 수정·철회 내역: [`28_REVIEW_R1_RESPONSE.md`](analysis/reports/28_REVIEW_R1_RESPONSE.md)
- 상세 색인: [`analysis/reports/README.md`](analysis/reports/README.md)

## 한 명령 실행

```bash
# 원본 데이터 배치: "5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv" (재배포 금지, KAMP 포털에서 수령)
pip install pandas numpy scipy scikit-learn matplotlib joblib pyyaml holidays

python3 analysis/final_baseline/run_baseline.py --config analysis/final_baseline/config.yaml  # 522초, CPU
python3 analysis/final_baseline/test_baseline.py                                              # 수용테스트 24항
```
외부 공공데이터 캐시가 없으면 자동으로 CORE 모드로 내려간다(`--no-external` 로 강제 가능).
후속 검증·1차 모의평가 대응 스크립트: `analysis/final_baseline/followup_*.py`, `sequential_policy.py`,
`peak_probability.py`, `error_conditions.py`.

## 핵심 성능 (원본구간 7~9월, rolling-origin OOF)

| 항목 | 값 |
|---|---|
| MAE h15 / h60 | **5.668 / 8.764 kw** (persistence 대비 −28.3% / −47.9%) |
| h60 베이스라인 | persistence 16.827 / Ridge 15.461 / RF 9.445 / HGB 8.764 |
| 피크 초과 확률 h60 | PR-AUC 0.357 · F1 0.444 · Brier 0.0523 · ECE 0.0460 |
| 신뢰도 밴드 h60 MAE | HIGH 4.70 / MEDIUM 12.60 / LOW 13.85 (OOD 9.34) |
| 구간 coverage h60 | 0.867 (피크구간 0.234 — 조건부 보정으로 0.119에서 개선) |
| 경보(선행 ≥30분 제약) | h60 q0.78 → h45 q0.90: 포착 0.529 · 선행 45분 · 오경보 5.0 스텝/일 |
| 작업자 검토율 | 11.98% (11.5건/일), 금지 문구 0건 |

## 하지 않는 주장

설비 조치 지시 / 셋업·고장 탐지 / 전기요금 절감액 / 계약전력 초과 회피 / 외부데이터 무용론 /
전이학습 불가 단정 / 장비 단위 분해(NILM). 철회된 주장은 보고서 28 §6 에 기록한다.

## 구조

```
analysis/
  final_baseline/     최종 파이프라인(config·features·models·pipeline·policy·diagnosis·report·inference)
                      + 수용테스트 + 후속검증 + 1차 평가 대응 스크립트
  01~07_*/            데이터 감사 / 4개 데이터셋 EDA / 저계측 검증(Task A~Q)
  reports/            보고서 01~29 (29 = 제출용)
  state/              가설 원장 · 근거 레지스트리 · 상태판 · 모의평가 로그
  tables/ figures/    생성 산출물 (행 단위·모델 산출물은 비커밋)
```

고정 seed 20260926 / CPU 단독(M1) / 원본·외부 원본 데이터는 커밋하지 않는다.
