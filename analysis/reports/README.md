# K-AI 제조데이터 분석 경진대회 — 탐색적 분석 (EDA) 산출물

목적: **주제 선정 전 근거 축적.** 최종 주제는 선정하지 않았다.

## 읽는 순서
0. `15_FINAL_DIRECTION_RECOMMENDATION.md` — **증거 기반 최종 방향 권고 15문 직답** (07 Sparse-FEMS 단계 결론)
1. `07_DISCOVERY_SUMMARY.md` — 전체 종합 + 부록의 정정 사항 (EDA 단계 종합)
2. `06_CROSS_DATASET_비교.md` — 4개 데이터셋 횡단 비교, 미해결 질문
3. `05_IDEA_INGREDIENT_BANK.md` — 조합 가능한 재료 목록(주제 아님)
4. Sparse-FEMS 단계: `10_SPARSE_FEMS_INFORMATION_ABLATION.md`, `11_HARD_CONDITION_MAP.md`,
   `12_NILM_SR_FEASIBILITY.md`, `13_RELIABILITY_AND_DECISION_POLICY.md`, `14_PROGRESSIVE_INSTRUMENTATION.md`
   / 상태판: `analysis/state/hypothesis_ledger.csv`(가설 12건), `analysis/state/evidence_registry.csv`(근거 24건)
5. 후속검증/문서조사: `08_후속검증_A_B_C.md`, `09_KAMP_공식문서_조사.md`
6. 개별 보고서: `01_사출성형_보고서.md`, `02_용접_보고서.md`, `03_프레스유압_보고서.md`(부록 A 정정 포함), `04_전력_보고서.md`(부록 A 정정 포함)

## 디렉터리
```
analysis/
  common.py                       공통 로더 / 경로 / seed / 전처리 결정
  01_data_audit/audit.py          파일 인벤토리, 데이터 딕셔너리, 구조 감사
  02_injection_molding/analyze.py 사출: 중복쌍·라벨충돌·버스트·레짐·베이스라인·오류·포켓·상호작용
                      tree_cn7.txt, tree_rg3.txt   실패 포켓 규칙
  03_robot_welding/analyze.py     용접: 라벨입도·삽입블록·파생변수 중복·관계안정성
  04_hydraulic_pump/analyze.py    프레스: 버스트구조·취득누출·특성군·오경보·Matrix Profile
                    leak_removed.py  누출 제거 후 공정 재측정 (부록 A)
  05_power/analyze.py             전력: 증강진단·피크정의·이벤트해부·원형군집·베이스라인·효율프런티어·전조
           clean_window.py        원본 76일 재검증 (부록 A)
  06_followup/A_state_split_regression.py  조업/비조업 분리 3모델 비교 + paired bootstrap
              B_duplication_handling.py    복제 처리 4방식 비교 (동일 원본 테스트셋)
              C_conformal_exceedance.py    QR/CQR coverage + 경보 운용곡선
  07_sparse_fems/                 저계측 FEMS 검증 (Task A~I). run_all.py 로 일괄 재현
      _fe.py                      공통 특성/정보수준/분할/지표/bootstrap
      feature_availability.py     특성 가용성 감사 + 누수 실증('평균' 열 적발)
      multi_horizon_baseline.py   t+15/30/45/60분 x persistence/Ridge/RF/HGB
      information_ablation.py     정보수준 L0~L4 ablation (MAE/피크AP/coverage/경보예산)
      hard_condition_discovery.py TimeSeriesSplit OOF 기반 난조건 발견 + 정보수준 교차
      latent_context.py           HMM(forward filtering)/GMM/NMF 잠재문맥 안정성·해석·가치
      symbolic_reliability.py     gplearn SR 3역할 + 채택 게이트
      reliability_meta.py         신뢰도 메타모델 vs 무학습 신호, 밴드 단조성
      confidence_gate.py          신뢰도 x 피크위험 등급 정책 운용곡선
      cost_aware_alert.py         FP/FN 비용비별 최적 경보 임계
      value_of_information.py     정보군 한계가치 / 최소 정보집합
  state/     analysis_state.md, hypothesis_ledger.csv, evidence_registry.csv
  (중간 캐시 07C_oof_predictions / 07D_latent_state_features / 07F_row_scores 는 원본 관측값을
   행 단위로 담으므로 .gitignore 처리. run_all.py 재실행으로 재생성된다.)
  tables/    CSV (utf-8-sig)
  figures/   PNG
  reports/   본 보고서들 + ENVIRONMENT.txt
```

## 실행
```bash
cd analysis
python3 01_data_audit/audit.py
python3 02_injection_molding/analyze.py
python3 03_robot_welding/analyze.py
python3 04_hydraulic_pump/analyze.py
python3 04_hydraulic_pump/leak_removed.py
python3 05_power/analyze.py
python3 05_power/clean_window.py
python3 06_followup/A_state_split_regression.py
python3 06_followup/B_duplication_handling.py
python3 06_followup/C_conformal_exceedance.py
python3 07_sparse_fems/run_all.py      # Task A~I 일괄 (M1 기준 약 12분)
```
전부 CPU 단독, M1 MacBook Air 기준 총 3분 이내. GPU 불필요.

## 재현성 기록
- **seed**: 20260926 (`common.SEED`). 사용처: StratifiedGroupKFold shuffle, RandomForest, IsolationForest, KMeans, HistGradientBoostingRegressor, 치환검정(200회), 배경 표본 추출.
- **라이브러리 버전**: `reports/ENVIRONMENT.txt` 참조 (python 3.9.6 / pandas 2.3.3 / numpy 2.0.2 / scipy 1.13.1 / scikit-learn 1.6.1 / matplotlib 3.9.4 / openpyxl 3.1.5)
- **한글 그림 폰트**: AppleGothic (`common.mpl()`)
- **추가 설치**: `openpyxl` (xlsx 읽기), `gplearn` 0.4.2 (Symbolic Regression), `hmmlearn` 0.3.3 (잠재문맥)
- 07 단계 seed 동일(20260926). 사용처 추가: HistGradientBoosting(회귀/분위/분류), RandomForest,
  TimeSeriesSplit(비랜덤), GaussianHMM/GaussianMixture/NMF, SymbolicRegressor, paired bootstrap 2,000회

## 전처리 결정 기록
| 데이터셋 | 결정 | 이유 |
|---|---|---|
| 사출 | 24개 특성이 동일한 행을 하나의 `pair`로 묶고 그룹 CV에 사용 | 전 행 2배 복제, 랜덤 CV 누출 |
| 사출 | 효과크기 계산 시 pair당 1행만 사용 | 표본 중복 제거 |
| 사출 | `Clamp_Open_Position` 제외 (labeled) | 상수 0 |
| 용접 | `scaled_data.csv` 미사용 | Raw data의 min-max 사본, 정보 중복 |
| 프레스 | Δt > 0.15 s 를 경계로 버스트 분할, 16샘플 미만 버스트 제외 | 실제 관측 단위는 5초 버스트 |
| 프레스 | 부록 A에서 진동 1e-3 / 전류 2.0 공통 격자 재양자화 | 취득경로 지문 제거 |
| 전력 | 2021-07-13, 07-15 의 `시간` 을 일자내 행 순서(0–23)로 복원 | 원값이 전력값으로 손상 |
| 전력 | 시간행을 15/30/45/60분 4개로 melt → 15분 시계열 | 최대수요 산정 단위 |
| 전력 | 원본 구간 = 2021-07-01 이후로 정의 | 일 프로파일 중복률 7월 0.065 / 8–9월 0.000 |
| 전력 | 결측(풍속 3, 강수량 1, 공장인원 17)은 모델 입력 시 −1 대체 | 결측 자체가 정보일 가능성 보존 |
| 전력(07) | `평균` 열 **전면 제외** | 15/30/45/60분 4개 타깃의 평균과 100% 일치 = 직접 누수 |
| 전력(07) | `공장인원`·`인건비` 제외 | 생산량과 spearman 0.994 / 고유값 2개 |
| 전력(07) | 날씨는 t 시점 관측값만 사용 | 미래 예보 미사용(누수 방지). 날씨 가치는 과소평가 가능 |
| 전력(07) | 생산량(t)은 '생산계획 사전확정' 가정. 엄격변형 L3s(1시간 지연) 병행 보고 | 가정 민감도 공개 |
| 전력(07) | HMM 은 forward filtering 만 사용(평활 posterior 금지) | 실시간 배치에서 미래 관측 불가 |
| 전력(07) | 피크 임계·스케일러·잠재모델·conformal 보정은 학습구간에서만 적합 | 임계 누수 방지 |

## 표기 규칙
- **[사실]** 데이터에서 직접 측정한 값 (표/그림에 근거 파일명 명시)
- **[해석]** 사실에 대한 추론
- **[가설]** 추가 검증이 필요한 주장
- 단위가 문서로 확인되지 않은 변수는 **unknown / needs verification** 으로 표기했다. 4개 데이터셋 중 단위 문서가 있는 것은 용접(`Welding Data Set_01.xlsx` → `data set` 시트)뿐이다.

## 제외한 것
- X-ray 이미지 데이터셋 (지시에 따라 분석 대상 아님)
- 하이퍼파라미터 탐색 (이번 단계 목적이 아님)
