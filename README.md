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

## 핵심 성능 (원본구간 7~9월, rolling-origin OOF) — v2

| 항목 | v1.1 | **v2** |
|---|---|---|
| MAE h15 / h60 | 5.668 / 8.764 kw | **5.094 / 7.907 kw** (persistence 대비 −35.6% / −53.0%) |
| 고사용량 구간 MAE h60 (ablation 기준) | 20.878 | **15.649 kw (−25.0%)** |
| 구간 coverage h60 (전체 / 피크구간) | 0.867 / 0.234 | **0.872 / 0.556** |
| 평균 구간폭 h60 | — | **39.87 kw** |
| 신뢰도 밴드 h60 MAE | 4.70 / 12.60 / 13.85 | **4.72 / 10.32 / 13.58** (OOD 5.80) |
| 사건 포착 / 오경보 / 평균 선행 | — | **0.884 / 3.21 스텝·일 / 43.2분** |
| 작업자 검토율 / 정보부족 비율 | 11.98% | **22.74% / 14.04%** |

v2 에서 바꾼 것: 잔차 타깃(`y−kw(t)`) · 푸리에 주기 인코딩 · SERA 계열 극단 가중 ·
휴무/재가동 특성. 분위모델과 conformal 보정도 같은 잔차 스케일로 옮겼다
(→ 피크구간 coverage 0.234 → 0.556).
근거: [`analysis/reports/32_V2_FORECAST_IMPROVEMENT.md`](analysis/reports/32_V2_FORECAST_IMPROVEMENT.md),
표 `analysis/tables/31_*.csv`.


## 경보 성능 — 숨기지 않고 적는 수치 (clean 7~9월, h60)

| 기준 | F1 |
|---|---:|
| 사전확률(모두 경보) | 0.111 |
| persistence 규칙 | 0.305 |
| 회귀 마진(점예측 그대로) | 0.322 |
| 이전 주 모델 HGB | 0.434 |
| **현재 ExtraTrees+isotonic** | **0.468** (h15 0.538) |
| 무편향 오라클 상한 | 0.693 |
| 완전 예측 | 1.000 |

사건 단위 recall **0.903** (186 사건 중 168).
**상한의 68% 지점이다. 좋다고 말하지 않는다.**
원인은 고사용량 구간의 체계적 과소예측 **+14.75 kw** — 고사용량 스텝의 81.6%에서 점예측이 임계 미달.
ROC-AUC 0.934 는 발생률 5.96%에서 과대평가되므로 근거로 쓰지 않는다.
근거: [`analysis/reports/33_MODEL_SELECTION_AND_F1_AUDIT.md`](analysis/reports/33_MODEL_SELECTION_AND_F1_AUDIT.md)

## 모델 선정 (대회 요건: 베이스라인 포함 2개 이상 비교)

- **수치 예측** 6종 비교(persistence / Ridge / RandomForest / ExtraTrees / HistGBM / HistGBM-deep)
  → ExtraTrees 가 전체 MAE 는 유의 개선이나 **고사용량 MAE 유의 악화(h60 +1.237, CI [+0.515, +1.911])**
  → 사전 선언 조건 위반으로 **기각**, HistGBM 유지
- **고사용량 경보** 9종 비교 → **ExtraTrees(balanced)+isotonic 채택** (F1 0.481, 사건 recall 0.903, 11.1건/일 ≤ 용량 12)
- 표: `33_model_compare_regression.csv`, `33_model_compare_alert_clean.csv`, `33_model_selection_verdict.csv`, `33_f1_ceiling_diagnosis.csv`

## 제출물 생성

```bash
python3 analysis/submission/rebuild_all.py        # 표·그림·보고서·발표자료 전부 재생성
python3 analysis/submission/make_submission.py    # 예측결과 파일 + 소스코드 zip
```
- 결과보고서: `submission/결과보고서_전력피크조기경보.hwpx` (주최측 양식 자동 채움)
- 발표자료: `submission/발표자료_전력피크조기경보.pptx`
- 수치는 전부 `analysis/tables/*.csv` 에서 읽어 생성한다. 손으로 옮겨 적지 않는다.

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
