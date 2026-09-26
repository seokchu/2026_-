# 제6회 K-AI 제조데이터 분석 경진대회 — 분석 저장소

현재 단계: **탐색적 분석(EDA) / 주제 발굴**. 최종 주제는 아직 선정하지 않았다.

## 빠른 시작

```bash
pip install pandas numpy scipy scikit-learn matplotlib openpyxl
cd analysis
python3 01_data_audit/audit.py
python3 02_injection_molding/analyze.py
python3 03_robot_welding/analyze.py
python3 04_hydraulic_pump/analyze.py
python3 04_hydraulic_pump/leak_removed.py
python3 05_power/analyze.py
python3 05_power/clean_window.py
```

CPU 단독, M1 MacBook Air 기준 총 3분 이내. GPU 불필요.

## 원본 데이터 배치

대회 제공 데이터는 **재배포하지 않으므로 이 저장소에 포함되어 있지 않다**(`.gitignore`).
KAMP 포털에서 내려받아 저장소 루트에 아래 이름 그대로 배치해야 스크립트가 동작한다.

```
1. 사출성형기 AI 데이터셋/moldset_labeled_cn7.csv
                        moldset_labeled_rg3.csv
                        moldset_unlabeled_cn7.csv
                        moldset_unlabeled_rg3.csv
2. 용접기 AI 데이터셋/Welding Data Set_01.xlsx
                     scaled_data.csv
3. 소성가공 예지보전 AI 데이터셋/press_data_normal.csv
                              outlier_data.csv
5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv
```

경로는 `analysis/common.py` 의 `DATA` 딕셔너리에 정의되어 있다.
파일 무결성은 `analysis/tables/01_file_inventory.csv` 의 md5(앞 12자)로 대조할 수 있다.

## 보고서

`analysis/reports/` — 서술은 한국어, 코드·변수·방법명은 영어.

| 순서 | 파일 |
|---|---|
| 1 | `07_DISCOVERY_SUMMARY.md` — 전체 종합 + 검증 후 정정 사항 |
| 2 | `06_CROSS_DATASET_비교.md` — 4개 데이터셋 횡단 비교, 미해결 질문 |
| 3 | `05_IDEA_INGREDIENT_BANK.md` — 조합 가능한 재료 목록(주제 아님) |
| 4 | `01_사출성형_보고서.md` / `02_용접_보고서.md` / `03_프레스유압_보고서.md` / `04_전력_보고서.md` |

`analysis/reports/README.md` 에 디렉터리 구조, seed, 라이브러리 버전, 전처리 결정 기록이 있다.

## 산출물

- `analysis/tables/` — 63개 CSV (utf-8-sig)
- `analysis/figures/` — 6개 PNG

## 표기 규칙

- **[사실]** 데이터에서 직접 측정 (근거 표 파일명 명시)
- **[해석]** 사실에 대한 추론
- **[가설]** 추가 검증 필요

단위가 문서로 확인되지 않은 변수는 `unknown / needs verification` 으로 표기했다.
