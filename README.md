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

## 성능 (clean 7~9월, rolling-origin OOF)

**수치 예측 (1시간 전)** · MAE 7.862 kw · RMSE 12.572 · MAPE 11.30% · **MASE 0.215** · **R² 0.958**

**고사용량 경보 — "앞으로 1시간 안에 발생하는가" (운영 정의)**

| 모델 | 정밀도 | 재현율 | F1 | MCC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **제안 (Extra Trees + margin 특성)** | **0.643** | **0.744** | **0.690** | **0.646** | **0.736** |
| persistence 규칙 (베이스라인) | 0.641 | 0.522 | 0.575 | 0.528 | 0.627 |
| 사전확률 (베이스라인) | 0.120 | 1.000 | 0.214 | 0.000 | 0.105 |

종전에는 "정확히 60분 뒤 그 15분 슬롯"을 라벨로 썼다. 15분만 어긋나도 오답이라 F1 0.499 였다.
모델을 바꾸지 않고 **평가 라벨을 시스템 출력과 맞추자 F1 0.499 → 0.690 (+38%)**.
근거: [`analysis/reports/36_LABEL_DEFINITION_FIX.md`](analysis/reports/36_LABEL_DEFINITION_FIX.md)

**미검증(마감 제약)** XGBoost/LightGBM/CatBoost, 딥 예측 모델, 시계열 파운데이션 모델 벤치마크.

## 표준 모델 벤치마크 (대회 요건: 베이스라인 포함 2개 이상 비교)

- **회귀 15종** Naive · Seasonal naive · Linear · Ridge · Lasso · ElasticNet · k-NN · SVR(RBF) ·
  Decision Tree · Random Forest · Extra Trees · Gradient Boosting · HistGBM · MLP · 제출 모델
- **분류 13종** Majority · Naive rule · Logistic · Gaussian NB · k-NN · SVM(RBF) · Decision Tree ·
  Random Forest · Extra Trees(+가중) · Gradient Boosting · HistGBM · MLP
- 선정: 회귀는 **고사용량 구간 MAE 기준**(Extra Trees 가 전체 MAE 는 낮지만 고사용량 MAE 유의 악화 → 기각),
  분류는 **F1·MCC 동시 최고인 Random Forest**
- 표: `34_standard_benchmark_regression_clean.csv`, `34_standard_benchmark_alert_clean.csv`
- 상세: [`analysis/reports/34_STANDARD_BENCHMARK.md`](analysis/reports/34_STANDARD_BENCHMARK.md)

## 부가 분석 (고유)

데이터 합성 복제일 감사 · 피크 편향으로 F1 상한 측정 · conformal 예측구간 · OOD 취약조건 지도 ·
선행시간 제도 근거. 보고서 31~33 참조.

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
