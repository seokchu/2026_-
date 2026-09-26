# K-AI 제조데이터 분석 경진대회 — 탐색적 분석 (EDA) 산출물

목적: **주제 선정 전 근거 축적.** 최종 주제는 선정하지 않았다.

## 읽는 순서
1. `07_DISCOVERY_SUMMARY.md` — 전체 종합 + 부록의 정정 사항 (여기부터 읽을 것)
2. `06_CROSS_DATASET_비교.md` — 4개 데이터셋 횡단 비교, 미해결 질문
3. `05_IDEA_INGREDIENT_BANK.md` — 조합 가능한 재료 목록(주제 아님)
4. 개별 보고서: `01_사출성형_보고서.md`, `02_용접_보고서.md`, `03_프레스유압_보고서.md`(부록 A 정정 포함), `04_전력_보고서.md`(부록 A 정정 포함)

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
  tables/    62개 CSV (utf-8-sig)
  figures/   6개 PNG
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
```
전부 CPU 단독, M1 MacBook Air 기준 총 3분 이내. GPU 불필요.

## 재현성 기록
- **seed**: 20260926 (`common.SEED`). 사용처: StratifiedGroupKFold shuffle, RandomForest, IsolationForest, KMeans, HistGradientBoostingRegressor, 치환검정(200회), 배경 표본 추출.
- **라이브러리 버전**: `reports/ENVIRONMENT.txt` 참조 (python 3.9.6 / pandas 2.3.3 / numpy 2.0.2 / scipy 1.13.1 / scikit-learn 1.6.1 / matplotlib 3.9.4 / openpyxl 3.1.5)
- **한글 그림 폰트**: AppleGothic (`common.mpl()`)
- **추가 설치**: `openpyxl` (xlsx 읽기)

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

## 표기 규칙
- **[사실]** 데이터에서 직접 측정한 값 (표/그림에 근거 파일명 명시)
- **[해석]** 사실에 대한 추론
- **[가설]** 추가 검증이 필요한 주장
- 단위가 문서로 확인되지 않은 변수는 **unknown / needs verification** 으로 표기했다. 4개 데이터셋 중 단위 문서가 있는 것은 용접(`Welding Data Set_01.xlsx` → `data set` 시트)뿐이다.

## 제외한 것
- X-ray 이미지 데이터셋 (지시에 따라 분석 대상 아님)
- 하이퍼파라미터 탐색 (이번 단계 목적이 아님)
