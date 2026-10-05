# K-AI 제조데이터 분석 경진대회 — 탐색적 분석 (EDA) 산출물

목적: EDA·주제검증부터 최종 Baseline v1.1 및 제출 준비까지의 근거·재현 산출물을 관리한다.

## 읽는 순서
0. `23_FINAL_BASELINE_V1.md` — **최종 Baseline v1.1**(단일 재현 파이프라인 / 성능·신뢰도·경보·파레토·정보부족 진단 / 코드 감사 결과). 여기부터 읽는다
0-1. `24_BASELINE_V1_FOLLOWUP.md` — 후속 검증(조건부 conformal 채택 / 비용 단위 환산 / 채널 가치 **음성** / 타 공장 전이 **음성**)
0-2. `25_PRESENTATION_OUTLINE.md` — 발표 구성안 12슬라이드 + 금지 주장 목록
0-3. `26_OFFICIAL_TASK_HORIZON_VERIFICATION.md` — **공식 예측 정의 검증**(가이드북 next-step 예측 근거 / day-ahead 96-step 오해 정리 / 현행 h15~h60 유지 판정)
0-4. `27_SUBMISSION_PACKAGING_CHECKPOINT.md` — **재실행·제출 체크포인트**(평가-배포 정합성 수정 / 최종 재실행 조건 / 제출 패키지 점검)
1. `22_FINAL_TOPIC_GATE.md` — 최종 주제 게이트 7문 직답
2. `17_국내_저계측_제조환경_실무근거.md` — **도입부용 Problem Validity 자료**(국내 저계측 제조현황 공식통계 / 외부데이터 질문 방어논리 / 안전한 주장·금지 주장)
3. `16_조건부보정_피크정의_다중horizon_3상태.md` — **2단계 검증**(Mondrian conformal / 피크 정의 통일 / 다중 horizon 경보 / 3상태). 피크 임계 정의가 여기서 확정된다
4. `15_FINAL_DIRECTION_RECOMMENDATION.md` — **증거 기반 최종 방향 권고 15문 직답** (07 Sparse-FEMS 단계 결론)
5. `07_DISCOVERY_SUMMARY.md` — 전체 종합 + 부록의 정정 사항 (EDA 단계 종합)
6. `06_CROSS_DATASET_비교.md` — 4개 데이터셋 횡단 비교, 미해결 질문
7. `05_IDEA_INGREDIENT_BANK.md` — 조합 가능한 재료 목록(주제 아님)
8. Sparse-FEMS 단계: `10_SPARSE_FEMS_INFORMATION_ABLATION.md`, `11_HARD_CONDITION_MAP.md`,
   `12_NILM_SR_FEASIBILITY.md`, `13_RELIABILITY_AND_DECISION_POLICY.md`, `14_PROGRESSIVE_INSTRUMENTATION.md`
   / 상태판: `analysis/state/hypothesis_ledger.csv`(가설 25건), `analysis/state/evidence_registry.csv`(근거 52건)
9. 3단계(외부데이터·운영정책): `18_EXTERNAL_DATA_STRESS_TEST.md`, `19_COST_MULTI_HORIZON_POLICY.md`,
   `20_THREE_STATE_ROBUSTNESS.md`, `21_OPERATOR_DIAGNOSTIC_LAYER.md` / 문제타당성: `17_국내_저계측_제조환경_실무근거.md`
10. 후속검증/문서조사: `08_후속검증_A_B_C.md`, `09_KAMP_공식문서_조사.md`
11. 개별 보고서: `01_사출성형_보고서.md`, `02_용접_보고서.md`, `03_프레스유압_보고서.md`(부록 A 정정 포함), `04_전력_보고서.md`(부록 A 정정 포함)

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
  07_sparse_fems/                 저계측 FEMS 검증 (Task A~I) + 2단계(J~M) + 3단계(N~Q). run_all.py 로 일괄 재현
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
      mondrian_conformal.py       조건부(그룹별) conformal 보정 — 피크 coverage 검증
      peak_definition.py          피크 임계 정의 6종 비교 + 통일 권고
      multihorizon_alert.py       15/30/45/60분 경보 정책 (이벤트 포착률·선행시간)
      three_state.py              정지/전환/정상조업 3상태 검증
      fetch_external.py           외부 공공데이터 수집·캐싱(공휴일/NOAA ISD/태양기하/요금 시간대)
      external_data_stress_test.py 외부데이터 E0~E3 ablation + CASE 판정
      integrated_alert_policy.py  비용 x 다중 horizon 통합 경보 정책 스윕
      three_state_robustness.py   3레짐 구조의 정의 독립성(규칙/변화점/GMM/HMM)
      operator_diagnostic.py      작업자 진단 계층 + 정보부족 플래그 검증
  final_baseline/                 **최종 Baseline v1** — 단일 명령 end-to-end 파이프라인
      config.yaml                 seed·horizon·외부데이터 규칙·신뢰도·레짐·피크·정책·비용비 설정
      features.py                 CORE/PUBLIC 특성 빌더 + 메타데이터 + 외부캐시 없을 때 CORE 강등
      models.py                   점예측/분위예측/CQR/OOD/밴드
      pipeline.py                 rolling-origin 평가(TRAIN/CAL/TEST) + 모드 게이트 + 레짐 + 지표
      policy.py                   h60->h15 에스컬레이션 스윕 + 비용비 + 비용x선행시간 파레토
      diagnosis.py                2x2 라우팅 + 정보부족 플래그 + 작업자 출력(물리조치 금지)
      report.py                   통합 지표표 + 그림 6종 + baseline_summary.json
      run_baseline.py             단일 진입점 (--no-external / --quick)
      inference.py                배포 산출물 로딩 + 다음 시점 추론
      test_baseline.py            수용 테스트 16항 (누수/정렬/임계/금지문구/재현성)
      followup_mondrian.py        조건부(Mondrian) conformal — 피크구간 coverage 개선 검증 (v1.1 채택 근거)
      followup_cost_units.py      비용비 -> 실제 단위 환산(측정 ΔkW + 파라미터 시나리오)
      followup_added_channel.py   조업·생산 채널 추가 가치 대리 측정 (음성 결과)
      followup_transfer_categoryB.py  타 공장 공개데이터(figshare CC BY 4.0) 전이 검증 (음성 결과)
  state/     analysis_state.md, hypothesis_ledger.csv, evidence_registry.csv
  (중간 캐시 07C_oof_predictions / 07D_latent_state_features / 07F_row_scores / 07O_policy_rows 는 원본 관측값을
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
python3 07_sparse_fems/run_all.py      # Task A~I + J~M + N~Q 일괄 (M1 기준 약 25분, 외부데이터 최초 1회 다운로드 포함)

# 최종 Baseline v1 (end-to-end 단일 명령, M1 기준 472초)
python3 final_baseline/run_baseline.py --config final_baseline/config.yaml
python3 final_baseline/test_baseline.py          # 수용 테스트 16항
python3 final_baseline/run_baseline.py --no-external --quick   # CORE 전용 축약 재현

# 후속 검증 4종 (보고서 24)
python3 final_baseline/followup_mondrian.py            # 17초
python3 final_baseline/followup_cost_units.py          # 2초 (run_baseline 산출물 필요)
python3 final_baseline/followup_added_channel.py       # 77초
python3 final_baseline/followup_transfer_categoryB.py  # 34초 (최초 1회 figshare 11MB 다운로드)
```
전부 CPU 단독, M1 MacBook Air 기준 총 3분 이내. GPU 불필요.

## 재현성 기록
- **seed**: 20260926 (`common.SEED`). 사용처: StratifiedGroupKFold shuffle, RandomForest, IsolationForest, KMeans, HistGradientBoostingRegressor, 치환검정(200회), 배경 표본 추출.
- **라이브러리 버전**: `reports/ENVIRONMENT.txt` 참조 (python 3.9.6 / pandas 2.3.3 / numpy 2.0.2 / scipy 1.13.1 / scikit-learn 1.6.1 / matplotlib 3.9.4 / openpyxl 3.1.5)
- **한글 그림 폰트**: AppleGothic (`common.mpl()`)
- **추가 설치**: `openpyxl` (xlsx 읽기), `gplearn` 0.4.2 (Symbolic Regression), `hmmlearn` 0.3.3 (잠재문맥)
- 3단계(N~Q) 런타임(캐시 있음): fetch_external 1.4s / external_data_stress_test 88.2s /
  integrated_alert_policy 37.7s / three_state_robustness 2.1s / operator_diagnostic 1.4s
  (fetch_external 최초 실행은 NOAA ISD 10개 관측소 다운로드로 약 60초, 캐시 19MB)
- **외부데이터**: 공휴일(python holidays, MIT), NOAA NCEI ISD global-hourly(미국 정부 공개), 태양기하(계산),
  한전 산업용 시간대 구분(구조만). 출처·라이선스·가용성 전체는 `analysis/tables/07N_external_data_sources.csv`.
  원본 캐시 `analysis/external_cache/` 는 커밋하지 않는다(재다운로드 가능).
- **Category B 외부데이터**: Lee·Baek·Kim (2022) Scientific Data, 한국 제조공장 10개소 1분 전력,
  figshare DOI 10.6084/m9.figshare.14822256.v9, **CC BY 4.0**. 캐시 `analysis/external_cache/categoryB/`(11MB, 비커밋).
  인용·출처 표기 의무가 있으므로 발표 시 표기한다.
- **추가 설치**: `holidays` 0.83 (공휴일 달력), `PyYAML` 6.0.3 / `joblib` 1.5.3 (Baseline v1 설정·모델 산출물)
- 4단계 Baseline v1 런타임: 전체 472초(특성·5폴드 평가 약 420초 / 정책 스윕 1,118개 4초 / 진단·그림·배포적합 약 45초).
  행 단위 산출물(`23_baseline_oof_rows.csv` 15MB, `23_baseline_operator_output.csv` 2MB)과
  모델 산출물(`analysis/final_baseline/artifacts/` 4.7MB)은 커밋하지 않는다(재실행으로 재생성).
- 2단계(J~M) 런타임: mondrian 152.7s / peak_definition 3.2s / multihorizon_alert 21.3s / three_state 13.8s
- **피크 임계 정의(확정)**: `p95_trailing_30d` — 직전 30일 p95, 적응형·leak-free (근거 `07K_peak_definitions.csv`).
  08/13 보고서의 기존 수치는 각자의 정의(전체구간 p95 / 첫 폴드 p95)를 병기한 채 보존한다.
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
