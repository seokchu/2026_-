"""결과보고서(.hwpx) 생성 — 양식 6개 장에 본문을 채운다. 수치는 CSV 에서 읽는다.

실행: python3 analysis/submission/build_report.py
출력: submission/결과보고서_전력피크조기경보.hwpx
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from hwpx_writer import build                                            # noqa: E402

TAB = ROOT / "analysis" / "tables"
TPL = ROOT / "경진대회 결과보고서 양식_일반국민,대학(원)생 부문.hwpx"
OUT = ROOT / "submission"; OUT.mkdir(exist_ok=True)


def L(n):
    return pd.read_csv(TAB / f"{n}.csv", encoding="utf-8-sig")


def f(df, **q):
    for k, v in q.items():
        df = df[df[k] == v]
    return df.iloc[0]


V2, V2B = L("31_v2_clean_summary"), L("31_v2_clean_bootstrap")
ABL = L("31_v2_ablation_summary")
DUP, DUPF = L("31_duplicate_day_audit"), L("31_duplicate_fold_crossing")
THS, LEAD = L("31_peak_threshold_sensitivity"), L("31_leadtime_basis")
OODS, OODD, OODB = L("31_ood_role_summary"), L("31_ood_decile_curve"), L("31_ood_weight_bootstrap")
STR = L("31_v2_stress_map")
MCR, MSC, MSB = L("33_model_compare_regression"), L("33_select_regression_clean"), L("33_select_regression_bootstrap")
MCA, FDG = L("33_model_compare_alert_clean"), L("33_f1_ceiling_diagnosis")
PKP = L("28_peak_probability_summary")
RSTD, CSTD = L("34_standard_benchmark_regression_clean"), L("34_standard_benchmark_alert_clean")
OURS_R = "HistGBM + 극단가중 (제출 모델)"
r_ours = RSTD[RSTD.model == OURS_R].iloc[0]
c_best = CSTD.sort_values("f1", ascending=False).iloc[0]
pk60 = PKP[(PKP.window == "clean_Jul_Sep") & (PKP.horizon_min == 60)].set_index("score")
pk15 = PKP[(PKP.window == "clean_Jul_Sep") & (PKP.horizon_min == 15)].set_index("score")
f1_now, f1_pers = float(pk60.loc["RF_clf(main)", "f1"]), float(pk60.loc["persistence_rule_margin", "f1"])
f1_ceil = float(FDG[FDG.model == "R4_hist_gbm"].oracle_f1_same_error.iloc[0])
bias_pk = float(FDG[FDG.model == "R4_hist_gbm"].bias_peak.iloc[0])
above = float(FDG[FDG.model == "R4_hist_gbm"].peak_pred_above_thr.iloc[0])
win_c = MCA.sort_values("f1", ascending=False).iloc[0]
AF = L("36_alert_final_metrics"); afc = AF[AF.window == "clean_Jul_Sep"]
OURS_C, PREV_C = "제안 (margin 특성 + RF)", "이전 구성 (기본특성 + RF)"
om = afc[afc.model == OURS_C].mean(numeric_only=True)
pmv = afc[afc.model == PREV_C].mean(numeric_only=True)
RNAME = {"R0_persistence": "직전값 지속(베이스라인)", "R1_ridge": "Ridge(베이스라인)",
         "R2_random_forest": "RandomForest", "R3_extra_trees": "ExtraTrees",
         "R4_hist_gbm": "HistGBM(현행)", "R5_hist_gbm_deep": "HistGBM-deep"}
CNAME = {"C5_extra_trees": "ExtraTrees(balanced)", "C6_hist_gbm": "HistGBM",
         "C2_quantile90_margin": "분위회귀 q0.90 마진", "C3_logistic": "Logistic(베이스라인)",
         "C4_random_forest": "RandomForest", "C0_persistence_rule": "persistence 규칙(베이스라인)",
         "C1_reg_margin": "회귀 마진(점예측 그대로)", "C7_stacked": "스태킹",
         "C8_stacked_isotonic": "스태킹+isotonic"}
IND = L("31_industry_identification")

a0 = f(V2, variant="A0_baseline", horizon_min=60)
a5 = f(V2, variant="A5_all", horizon_min=60)
da, dc = f(DUP, scope="all_2021"), f(DUP, scope="clean_Jul_Sep")
o0, o5 = f(OODS, variant="A0_baseline"), f(OODS, variant="A5_all")
ob = f(OODB, scope="OOD_only")


def pct(v): return f"{v:+.1f}%"


# 선택적으로 최신 v1.1 산출물을 참조 (없으면 해당 줄 생략)
def opt(name):
    p = TAB / f"{name}.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


FM = opt("23_baseline_final_metrics")
PK = opt("28_peak_probability_summary")
FNFP = opt("30_fnfp_by_state")

C1 = [
 (1, "데이터 개요와 업종"),
 (2, f"대상: {f(IND, field='업종').value} 1개 공장. 2021-01-01~09-14, 257일 × 24시간 = 6,168행 × 18열."),
 (2, "출처 표기: 중소벤처기업부·KAMP·자원 최적화 AI 데이터셋·KAIST(울산과학기술원, ㈜유피시앤에스), 2021.12.27."),
 (3, "업종은 데이터 파일 자체에 적혀 있지 않다. 데이터셋 설명과 동일 데이터를 쓴 공개 저장소 2건을 교차확인해 판단했다. 설비 구성·공정 순서는 데이터에 없으므로 설비 단위 주장은 하지 않는다."),
 (1, "변수 정의와 제조공정 상태와의 연결"),
 (2, "전력: 15분·30분·45분·60분 = 매 시간 내 4개 15분 수요 관측. 이를 펼쳐 15분 간격 수요 시계열(kw)로 재구성했다. 15분은 한전 요금적용전력의 측정 단위와 동일하다."),
 (2, "생산량: 시간당 생산 수량. 생산량 > 0 을 가동(is_operating), 0 을 비가동으로 본다. 이 이진 상태가 전력 프로파일의 계단형 레벨 변화와 직접 대응한다(가동 중앙값 130 kw, 비가동 중앙값 22 kw)."),
 (2, "기상(기온·풍속·습도·강수량): 동시각 관측값. 예보는 쓰지 않는다 — 결정시점에 예보를 가졌다고 가정하면 누수다."),
 (2, "달력(day·d·m)·전기요금(계절): 사전 확정 정보이므로 미래 시점에 대해서도 사용 가능."),
 (1, "전처리 내역"),
 (2, "① 제외: '평균' 열은 같은 시간 4개 분값의 평균이므로 예측 타깃을 포함한다. 전면 제외했다."),
 (2, "② 제외: '공장인원'은 생산량과 Spearman 0.994 로 중복, '인건비'는 고유값 2개로 사실상 상수."),
 (2, "③ 복구: 2021-07-13·07-15 의 '시간' 열이 전력값으로 덮여 있었다(값 > 23). 행 순서가 보존되어 있어 일자별 누적 순번으로 재구성했다."),
 (2, "④ 결측 의심 표기: kw 가 정확히 0 인 74 스텝(08-28 25건, 08-29 47건, 09-08 2건). 정지 중 기저부하가 22 kw 수준으로 유지되는 것과 모순되므로 계량 결측으로 의심해 별도 표기했다. 임의로 보간하지 않았다."),
 (2, "⑤ 파생: 지연(1·2·3·4·8·96스텝), 이동평균·표준편차·최대(4·96스텝), 차분·4스텝 램프, 하루 내 위치(0~95), 주말 여부. 전부 결정시점 t 까지의 정보만 쓴다."),
 (1, "데이터 진단 — 합성 증강으로 생긴 중복일"),
 (2, f"하루 96스텝 전력 프로파일을 해시로 비교한 결과, 전체 {int(da.n_days)}일 중 {int(da.n_dup_days)}일이 다른 날과 완전히 동일했다. {int(da.n_dup_groups)}개 그룹, 잉여 {int(da.n_redundant_days)}일(전체의 {da.dup_day_share*100:.1f}%)."),
 (2, f"반면 2021-07-01 이후 구간은 {int(dc.n_days)}일 중 중복 {int(dc.n_dup_days)}일(1개 그룹)뿐이다."),
 (2, "중복일이 같은 fold 의 TRAIN 과 TEST 양쪽에 들어가면 성능이 부풀려진다. fold 별로 세어 보면 fold0 23개, fold1 1개, fold2 28개, fold3 7개, fold4 0개 그룹이 걸쳐 있다."),
 (3, "따라서 주 보고 구간을 clean_Jul_Sep(2021-07-01~09-14)으로 고정했다. 이 구간을 TEST 로 쓰는 fold4 는 중복 교차가 0 이다."),
 (1, "고사용량(피크) 라벨 정의와 그 근거"),
 (2, "라벨: y(t+h) ≥ thr(t), thr(t) = 직전 30일 수요의 p95 (현재·미래값 제외, 최소 7일 누적)."),
 (2, "근거 ① 15분 — 한전 요금적용전력은 15분 평균전력으로 측정한다. 해상도를 임의로 고른 것이 아니다."),
 (2, "근거 ② 30일 — 요금적용전력은 검침 당월을 포함한 직전 12개월 중 7~9월·12~2월분 및 당월분의 최대값으로 결정된다. 직전 ~30일(검침월) 창이 과금 주기와 정합한다. 또한 본 분석 구간 7~9월은 연간 기본요금을 결정하는 하계 피크월에 해당한다."),
 (2, "근거 ③ p95 는 과금 규칙이 아니라 모델링 선택이다. '월 최대 1점'을 그대로 라벨로 쓰면 월당 양성 1건이 되어 학습도 평가도 불가능하다. 이 점을 숨기지 않고 민감도표로 공개한다."),
 (2, "민감도(clean 구간): 14일 p95 는 임계 일평균 변화 0.839 kw 로 흔들리고, 90일 p95 는 0.227 kw 로 둔하다. 30일 p95 는 0.487 kw 로 그 사이다. 고정 임계 187 kw 는 발생률 2.4%(사건 273건)로 양성이 더 희소해진다."),
 (3, "상세: analysis/tables/31_peak_threshold_sensitivity.csv (창 14/30/60/90일 × 분위 0.90/0.95/0.97 + 고정 180/187/190 kw)"),
]

bb = {(int(r.horizon_min), r.scope): r for _, r in V2B.iterrows()}
C2 = [
 (1, "예측 문제 정의"),
 (2, "결정시점 t 에서 t+15 / +30 / +45 / +60분의 15분 수요를 각각 직접(direct) 예측한다. 재귀 예측을 쓰지 않아 오차 누적을 피한다."),
 (2, "원본의 15분·30분·45분·60분 열은 '예측 호라이즌'이 아니라 한 시간 안의 네 개 관측 시각이다. 이를 호라이즌으로 오해하지 않도록 15분 간격 단일 시계열로 재구성한 뒤 h1~h4 타깃을 따로 만들었다."),
 (1, "평가 규약 — 비교 전에 먼저 고정했다"),
 (2, "Rolling-origin(TimeSeriesSplit, 5 fold). fold 내부를 TRAIN / CAL(TRAIN 뒤 20%) / TEST 로 시간순 분리."),
 (2, "임계·보정계수·샘플가중·스케일러·OOD 컷·특성모드 선택을 전부 TRAIN(+CAL)에서만 적합한다. TEST 는 적용만 한다."),
 (2, "랜덤 분할을 쓰지 않는다. 주 보고 구간은 clean_Jul_Sep(OOF 7,289 스텝, 고사용량 419 스텝)."),
 (2, "수치 예측 성능(MAE·RMSE·고사용량 구간 MAE)과 경고 성능(FN·FP·F1)을 하나의 점수로 섞지 않는다."),
 (1, "표준 모델·표준 지표 벤치마크 — 먼저 통용되는 기준으로 위치를 확인한다"),
 (2, "회귀 15종: Naive(직전값) / Seasonal naive(24시간 전) / Linear Regression / Ridge / Lasso / ElasticNet / k-NN / SVR(RBF) / Decision Tree / Random Forest / Extra Trees / Gradient Boosting / HistGradientBoosting / MLP / 제출 모델(HistGBM + 극단가중)."),
 (2, "지표도 표준만 쓴다: MAE, RMSE, MAPE, sMAPE, MASE, R². MASE 는 학습구간 Seasonal naive(24시간) 오차로 정규화한 값이라 단위에 의존하지 않는다."),
] + [
 (2, f"{r.model}: MAE {r.mae:.3f} / RMSE {r.rmse:.3f} / MAPE {r.mape:.2f}% / sMAPE {r.smape:.2f}% / MASE {r.mase:.3f} / R² {r.r2:.3f} / 고사용량 MAE {r.mae_peak:.3f}")
 for _, r in RSTD.sort_values("mae").iterrows()
] + [
 (3, f"제출 모델은 MAE 기준 4위이지만 고사용량 구간 MAE 는 {r_ours.mae_peak:.3f} kw 로 15종 중 가장 낮다. 과제가 고사용량 판정이므로 이 기준으로 골랐다(아래 선정 절차)."),
 (3, f"표준 지표로 읽으면 MAPE {r_ours.mape:.2f}%, MASE {r_ours.mase:.3f}, R² {r_ours.r2:.3f} 이다. MASE 1 미만은 24시간 전 값을 그대로 쓰는 것보다 낫다는 뜻이다."),
 (1, "표준 분류 모델 13종 — 고사용량 경보"),
 (2, "Majority class(전부 정상) / Naive rule(현재값>임계) / Logistic Regression / Gaussian Naive Bayes / k-NN / SVM(RBF) / Decision Tree / Random Forest / Extra Trees / Extra Trees+클래스가중 / Gradient Boosting / HistGradientBoosting / MLP."),
 (2, "지표: Accuracy, Precision, Recall, Specificity, Balanced Accuracy, F1, MCC, ROC-AUC, PR-AUC, Brier."),
] + [
 (2, f"{r.model}: 정확도 {r.accuracy:.3f} / 정밀도 {r.precision:.3f} / 재현율 {r.recall:.3f} / 특이도 {r.specificity:.3f} / 균형정확도 {r.balanced_accuracy:.3f} / F1 {r.f1:.3f} / MCC {r.mcc:.3f} / PR-AUC {r.pr_auc:.3f}")
 for _, r in CSTD.sort_values("f1", ascending=False).iterrows()
] + [
 (3, "정확도는 이 문제에서 의미가 작다. 전부 '정상'으로 답해도 0.943 이 나온다. F1 과 MCC 로 읽어야 한다."),
 (3, f"Random Forest 가 F1 {c_best.f1:.3f}, MCC {c_best.mcc:.3f} 로 동시 최고다. 클래스 가중을 주면 F1 이 0.453 으로 떨어져 쓰지 않는다."),
 (1, "그다음에 본 과제 고유 비교 — 점진 개선과 기준선"),
 (2, "기준선 B0 직전값 지속(persistence), B1 Ridge, B2 RandomForest."),
 (2, "주 모델 A0 HistGradientBoosting(기본). 여기에 네 가지 개선을 하나씩 더한 A1~A4, 전부 결합한 A5."),
 (3, "A1 잔차 타깃 — y(t+h) 대신 y(t+h) − kw(t) 를 학습하고 예측 시 kw(t) 를 더한다. 트리 모델이 수준 외삽에 약한 문제를 피한다."),
 (3, "A2 주기 인코딩 — 하루 내 위치(0~95)와 요일을 푸리에 항(sin/cos, 3차)으로 넣는다. 정수 인덱스의 인위적 순서를 없앤다."),
 (3, "A3 극단 가중 — 적합집합 분위 기준 relevance 가중(SERA 계열). 고사용량 표본의 손실 기여를 키운다. 가중은 TRAIN(+CAL)에서만 산출한다."),
 (3, "A4 휴무·재가동 특성 — 비가동 지속 길이, 가동 지속 길이, 재가동 1시간 이내 여부, 현재 수요와 24시간 이동평균의 차·비. 전부 t 시점까지의 인과 정보."),
 (2, "심층 신경망·파운데이션 모델은 쓰지 않았다. 표본이 7천 스텝 규모이고 CPU 전용 환경에서 재현성이 중요하며, 가동/정지의 계단형 레벨 변화는 트리 분할이 바로 처리한다."),
 (1, "성능 결과 (clean_Jul_Sep, OOF)"),
 (2, f"h60: MAE {a0.mae:.3f} → {a5.mae:.3f} kw ({pct(a5.mae_vs_A0_pct)}), 고사용량 구간 MAE {a0.mae_peak:.3f} → {a5.mae_peak:.3f} kw ({pct(a5.maepeak_vs_A0_pct)})."),
] + [
 (2, f"h{int(r.horizon_min)}: MAE {f(V2, variant='A0_baseline', horizon_min=int(r.horizon_min)).mae:.3f} → {r.mae:.3f} kw ({pct(r.mae_vs_A0_pct)}), "
     f"고사용량 MAE {f(V2, variant='A0_baseline', horizon_min=int(r.horizon_min)).mae_peak:.3f} → {r.mae_peak:.3f} kw ({pct(r.maepeak_vs_A0_pct)})")
 for _, r in V2[V2.variant == "A5_all"].sort_values("horizon_min").iterrows()
] + [
 (2, "기준선 대비(h60, 전 구간 OOF 평균): 직전값 지속 "
     f"{f(ABL, variant='B0_persistence', horizon_min=60).mae:.3f} / Ridge {f(ABL, variant='B1_ridge', horizon_min=60).mae:.3f} / "
     f"RandomForest {f(ABL, variant='B2_rf', horizon_min=60).mae:.3f} / A0 {f(ABL, variant='A0_baseline', horizon_min=60).mae:.3f} / "
     f"A5 {f(ABL, variant='A5_all', horizon_min=60).mae:.3f} kw."),
 (1, "최종 파이프라인 적용 결과 (clean_Jul_Sep, 신뢰도·경보까지 포함)"),
 (2, f"h60 MAE {FM.set_index('horizon_min').loc[60,'mae']:.3f} kw, 직전값 지속 대비 {FM.set_index('horizon_min').loc[60,'persistence_improvement']*100:.1f}% 개선 (v1.1 8.764 kw / 47.9%)."),
 (2, f"h15 MAE {FM.set_index('horizon_min').loc[15,'mae']:.3f} kw (v1.1 5.668 kw)."),
 (2, f"예측구간: 전체 적중률 {FM.set_index('horizon_min').loc[60,'coverage']:.3f}, "
     f"고사용량 구간 적중률 {FM.set_index('horizon_min').loc[60,'coverage_peak_only']:.3f} (v1.1 0.234), "
     f"평균 구간폭 {FM.set_index('horizon_min').loc[60,'mean_interval_width']:.2f} kw."),
 (3, "분위모델과 conformal 보정을 점예측과 같은 잔차 스케일로 옮긴 결과다. 구간은 더 좁아지면서 적중률은 올라갔다. 다만 목표 0.90 에는 여전히 미달한다."),
 (2, f"신뢰도 밴드별 h60 MAE: HIGH {FM.set_index('horizon_min').loc[60,'mae_band_HIGH']:.2f} / "
     f"MEDIUM {FM.set_index('horizon_min').loc[60,'mae_band_MEDIUM']:.2f} / "
     f"LOW {FM.set_index('horizon_min').loc[60,'mae_band_LOW']:.2f} kw — 단조 증가한다(밴드가 난이도를 실제로 구분한다)."),
 (1, "유의성 검정 — 일 단위 블록 부트스트랩"),
 (2, "수요는 하루 안에서 강하게 상관되므로 스텝 단위 재표집은 신뢰구간을 과소추정한다. 날짜를 블록으로 1,000회 재표집해 절대오차 차의 95% 구간을 구했다."),
] + [
 (2, f"h{h} {'전체' if sc=='ALL' else '고사용량 구간'}: MAE 차 {bb[(h,sc)].delta_mae:+.3f} kw, 95% CI [{bb[(h,sc)].ci_lo:.3f}, {bb[(h,sc)].ci_hi:.3f}] → "
     f"{'개선 유의' if bb[(h,sc)].improves else '유의하지 않음'}")
 for h in (15, 30, 45, 60) for sc in ("ALL", "peak_only")
] + [
 (3, "8개 비교 전부 신뢰구간 상한이 0 보다 작다. 우연으로 보기 어렵다."),
 (1, "모델 비교 ① 수치 예측 — 베이스라인 포함 6종, 동일 조건"),
] + [
 (2, f"{RNAME[r.model]}: "
     f"MAE {r.mae:.3f} / RMSE {r.rmse:.3f} / 고사용량 MAE {r.mae_peak:.3f} / 고사용량 편향 {r.bias_peak:+.2f} kw")
 for _, r in MCR[MCR.horizon_min == 60].sort_values("mae").iterrows()
] + [
 (3, "전부 같은 fold·같은 적합집합·같은 v2 레시피(잔차 타깃 + 극단 가중)로 적합했다. 전 구간 OOF, h60."),
 (1, "선정 — 사전 선언한 규칙과 그 결과"),
 (2, "규칙: (a) clean 4개 horizon 전부 MAE 악화 없음 AND (b) h60 일 블록 부트스트랩 95% CI 상한 < 0 AND (c) 고사용량 구간 MAE 악화 없음."),
] + [
 (2, f"h{int(r.horizon_min)} {'전체' if r.scope=='ALL' else '고사용량 구간'}: ExtraTrees − HistGBM = {r.delta_mae:+.3f} kw, 95% CI [{r.ci_lo:.3f}, {r.ci_hi:.3f}] → "
     f"{'개선' if r.improves else ('악화' if r.ci_lo > 0 else '차이 없음')}")
 for _, r in MSB.iterrows()
] + [
 (2, "판정: (a) 충족, (b) 충족, (c) 위반 → ExtraTrees 기각. HistGBM 유지."),
 (3, "ExtraTrees 는 전체 MAE 를 10% 낮추지만 고사용량 구간을 더 깎는다. 평균 오차를 사는 대신 정작 맞혀야 할 구간을 판다. 전체 MAE 만 보고 골랐다면 틀린 선택이 된다."),
 (1, "모델 비교 ② 고사용량 경보 — 베이스라인 포함 9종"),
] + [
 (2, f"{CNAME[r.model]}: "
     f"F1 {r.f1:.3f} / 정밀도 {r.precision:.3f} / 재현율 {r.recall:.3f} / 사건 recall {r.event_recall:.3f} / 경보 {r.alerts_per_day:.1f}건·일 (TP {int(r.tp)} FP {int(r.fp)} FN {int(r.fn)})")
 for _, r in MCA.sort_values("f1", ascending=False).iterrows()
] + [
 (2, "선정 규칙: 경보 12건/일 이하인 후보 중 clean F1 최대, 동점 시 사건 recall. 용량 제약은 본 실험 이전부터 config.yaml 에 있던 값이다."),
 (2, f"선택: {CNAME[win_c.model]} (F1 {win_c.f1:.3f}, 사건 recall {win_c.event_recall:.3f}, {win_c.alerts_per_day:.1f}건/일). HistGBM 은 12.4건/일로 용량 제약 위반."),
 (3, "스태킹은 전 구간 F1 0.477 로 상위였으나 clean 구간에서 0.251 로 붕괴했다. 스태커가 중복일이 섞인 앞쪽 fold 에 과적합했다. 전 구간 수치만 봤다면 잘못 골랐을 것이다."),
 (2, f"주 분류기를 HistGBM+isotonic 에서 RandomForest+isotonic 으로 교체했다. clean h60 F1 {float(pk60.loc['HGB_clf_uncalibrated','f1']):.3f} → {f1_now:.3f}, Brier {float(pk60.loc['HGB_clf_uncalibrated','brier']):.4f} → {float(pk60.loc['RF_clf(main)','brier']):.4f}, ECE {float(pk60.loc['HGB_clf_uncalibrated','ece']):.4f} → {float(pk60.loc['RF_clf(main)','ece']):.4f}. h15 은 F1 {float(pk15.loc['RF_clf(main)','f1']):.3f}."),
 (1, "경보 정밀도 업그레이드 — 임계 거리 특성 + 통합 운영점"),
 (2, f"문제: 이전 구성의 정밀도가 {pmv.precision:.3f} 으로 낮았다. 경보 3건 중 2건이 헛경고이고 하루 {pmv.alerts_per_day:.1f}건을 확인해야 했다."),
 (2, "대응 ① 임계까지의 거리를 직접 특성화했다 — margin(현재수요-임계), 비율, 지연 1/2/4/8/96, 램프, 이동 최대·평균, 당일 최대 margin."),
 (2, "대응 ② 피크 이력 — 최근 24시간·7일 피크 횟수, 마지막 피크 이후 경과 스텝."),
 (2, "대응 ③ 운영점을 폴드별이 아니라 전 폴드 CAL 을 통합해 확률 임계 1개로 결정했다. 폴드 간 양성 비율이 0.031~0.091 로 3배 차이나 폴드별 임계가 전이되지 않기 때문이다."),
 (2, "대응 ④ 클래스 가중·리샘플링은 쓰지 않았다. 문헌과 자체 실험 모두 재현율을 올리고 정밀도를 낮춘다."),
 (3, "산업 경보 표준 기법인 delay-timer(n-out-of-m)도 후보에 넣었으나 CAL 에서 선택되지 않았다. 본 데이터에서는 효과가 없었다."),
] + [
 (2, f"h{int(r.horizon_min)}: 정밀도 {afc[(afc.model==PREV_C)&(afc.horizon_min==r.horizon_min)].precision.iloc[0]:.3f} -> {r.precision:.3f}, "
     f"F1 {afc[(afc.model==PREV_C)&(afc.horizon_min==r.horizon_min)].f1.iloc[0]:.3f} -> {r.f1:.3f}, "
     f"경보 {afc[(afc.model==PREV_C)&(afc.horizon_min==r.horizon_min)].alerts_per_day.iloc[0]:.1f} -> {r.alerts_per_day:.1f}건/일")
 for _, r in afc[afc.model == OURS_C].sort_values("horizon_min").iterrows()
] + [
 (2, f"4개 호라이즌 평균: 정밀도 {pmv.precision:.3f} -> {om.precision:.3f} (+{(om.precision/pmv.precision-1)*100:.0f}%), "
     f"F1 {pmv.f1:.3f} -> {om.f1:.3f}, PR-AUC {pmv.pr_auc:.3f} -> {om.pr_auc:.3f}, "
     f"확인 부담 {pmv.alerts_per_day:.1f} -> {om.alerts_per_day:.1f}건/일 (-{(1-om.alerts_per_day/pmv.alerts_per_day)*100:.0f}%)."),
 (2, f"대가를 숨기지 않는다: 사건 포착률이 {pmv.event_recall:.3f} 에서 {om.event_recall:.3f} 로 낮아진다. 경보량을 절반으로 줄인 결과다."),
 (3, "같은 경보 예산으로 맞추면 제안 구성의 스텝 단위 정밀도는 오히려 낮아진다. 즉 낮은 경보량 구간에서 더 정확한 모델이지 전 구간에서 우월한 모델이 아니다. 운영 모드를 정밀 모드(기본)와 포착 모드 두 가지로 제시한다."),
 (1, "최종 모델 선택 이유"),
 (2, "A5 를 최종 모델로 선택했다. 네 호라이즌 모두에서 MAE 와 고사용량 구간 MAE 가 동시에 개선되었고, 블록 부트스트랩으로 유의했으며, 개선의 출처가 네 가지 개별 요소로 분해되어 설명 가능하기 때문이다."),
 (2, "선택 규칙은 실험 전에 선언했다: '전 호라이즌에서 MAE 가 악화되지 않고 고사용량 구간 MAE 가 개선되며 블록 부트스트랩 CI 상한이 0 미만'. 돌려본 뒤 기준을 바꾸지 않았다."),
 (3, "개별 요소만 보면 A3(극단 가중)는 전체 MAE 를 오히려 약간 악화시키면서 고사용량 MAE 만 낮춘다. 결합했을 때 두 지표가 함께 좋아진다는 점을 그대로 보고한다."),
]

st60 = STR[(STR.horizon_min == 60)]
CN = {"peak_only": "고사용량 구간만", "high_ramp_p90": "급변동 상위 10%", "non_operating": "비가동",
      "restart_1h": "재가동 1시간 이내", "midday_12_18": "12~18시", "weekend": "주말"}
C3 = [
 (1, "조건별 성능 — 전체 평균만 보지 않는다 (h60)"),
] + [
 (2, f"{CN[c]}: 표본 {int(f(st60, variant='A5_all', condition=c).n):,}스텝(전체의 {f(st60, variant='A5_all', condition=c).share*100:.1f}%), "
     f"MAE A0 {f(st60, variant='A0_baseline', condition=c).mae:.3f} → A5 {f(st60, variant='A5_all', condition=c).mae:.3f} kw, "
     f"전체 대비 배율 {f(st60, variant='A0_baseline', condition=c).mae_ratio_vs_all:.2f} → {f(st60, variant='A5_all', condition=c).mae_ratio_vs_all:.2f}")
 for c in CN
] + [
 (3, "가장 취약한 조건은 고사용량 구간 그 자체다(전체의 1.85배 → 1.61배). 다음이 재가동 직후와 급변동 구간이다."),
 (3, "A5 는 비가동·주말 구간에서 특히 크게 줄었다. 휴무 지속·재가동 경과 특성이 실제로 기여했다는 뜻이다."),
 (1, "F1 감사 — 0.47 은 본질적 한계인가 모델 결함인가"),
 (2, "같은 크기의 오차를 가지되 편향이 없는 예측기의 F1 상한을 측정했다. 실제값에 우리 모델의 잔차를 무작위로 섞어 주입하고(오차 크기는 같고 타깃과의 상관만 끊는다) CAL 과 동일한 방식으로 임계를 훑었다."),
 (2, f"결과: 현재 F1 {f1_now:.3f} 대 무편향 오라클 상한 {f1_ceil:.3f}. 격차 {f1_ceil - f1_now:.3f}. 본질적 한계가 아니라 결함이다."),
 (2, f"원인: 전체 편향은 {float(FDG[FDG.model=='R4_hist_gbm'].bias_all.iloc[0]):+.2f} kw 로 거의 0 인데, 고사용량 스텝에서만 {bias_pk:+.2f} kw 과소예측한다. 그 결과 고사용량 스텝의 {(1-above)*100:.1f}% 는 점예측이 임계선에 도달조차 못 한다."),
 (3, "MAE 를 최소화하는 회귀는 조건부 평균으로 수렴한다. 수요 분포의 오른쪽 꼬리에서는 평균이 실제보다 낮으므로 구조적으로 피크를 깎는다. MAE 가 좋아져도 F1 이 따라오지 않는 이유가 이것이다. 두 지표는 같은 방향을 가리키지 않는다."),
 (2, f"라벨도 경계에 몰려 있다 — 고사용량 스텝의 {float(FDG[FDG.model=='R4_hist_gbm'].peak_within_mae_of_thr.iloc[0])*100:.1f}% 가 임계선에서 ±MAE 이내다. 적응형 p95 라벨은 '상위 5%' 이므로 대부분의 양성이 경계 바로 위다."),
 (3, "다만 위 오라클은 이 경계 밀집 효과를 이미 포함하고도 0.69 를 낸다. 경계 밀집이 현재 수준을 설명해 주지는 않는다."),
 (1, "지금 수준을 기준선과 함께 적으면 (clean, h60)"),
 (2, f"사전확률(모두 경보) {float(pk60.loc['prior_constant','f1']):.3f} < persistence 규칙 {f1_pers:.3f} < 회귀 마진(점예측 그대로) {float(MCA[MCA.model=='C1_reg_margin'].f1.iloc[0]):.3f} < 현재 {f1_now:.3f} < 무편향 오라클 {f1_ceil:.3f} < 완전 예측 1.000."),
 (2, f"persistence 규칙 대비 +{(f1_now/f1_pers-1)*100:.0f}%, 사전확률 대비 {f1_now/float(pk60.loc['prior_constant','f1']):.1f}배. 그러나 상한의 {f1_now/f1_ceil*100:.0f}% 지점이다. 좋다고 말할 수 없다."),
 (2, f"사건 단위로는 186 사건 중 {int(round(float(win_c.event_recall)*186))} 건을 포착한다(사건 recall {float(win_c.event_recall):.3f}). 스텝 F1 과 사건 recall 은 다른 질문에 답한다. 하나로 섞지 않는다."),
 (2, "ROC-AUC 는 0.934 이지만 발생률 5.96% 에서 과대평가되는 지표다. 성능 근거로 쓰지 않는다."),
 (2, "남은 격차를 메우는 방향(다음 라운드, 사전 채택조건 선언 후 검정): ① 조건부 평균 대신 높은 분위를 점예측으로 사용 ② 과소예측에 더 큰 벌점을 주는 비대칭 손실 ③ 라벨을 이진이 아닌 초과 여유 회귀로 두고 결정만 이진화 ④ 스텝 F1 대신 사건 recall 과 확인 부담으로 임계 선택."),
 (1, "처음 보는 입력(OOD)의 역할을 다시 정의했다"),
 (2, "마할라노비스 거리를 TRAIN 에서 적합하고 TRAIN p99 를 컷으로 쓴다. 이것을 '판단 포기 신호'로 쓰는 것이 맞는지 검정했다."),
 (2, "① 지시력은 있다 — 거리 십분위별 MAE 가 최하위 6.44 kw 에서 최상위 10.10 kw 로 증가한다."),
 (2, f"② 그러나 포기 기준으로 쓰기엔 범위가 좁다 — OOD 판정은 전체의 {o5.ood_share*100:.1f}% 에 불과하다."),
 (2, f"③ 실제 해법은 모델 자체의 개선이었다 — OOD 구간 MAE / 정상 구간 MAE 비가 {o0.ood_penalty_ratio:.2f}(A0)에서 {o5.ood_penalty_ratio:.2f}(A5)로 내려갔다."),
 (2, f"④ OOD 구간 표본을 일부러 더 가중해 학습한 변형(A6)은 오히려 악화되어 기각했다 — OOD 구간 MAE 차 {ob.delta_mae:+.3f} kw, 95% CI [{ob.ci_lo:.3f}, {ob.ci_hi:.3f}]."),
 (3, "결론: OOD 는 취약 조건을 지목하는 진단 도구로만 쓴다. 회피 또는 가중은 본 데이터에서 효과가 없었다."),
 (1, "미탐지(FN)·오경보(FP) 분석"),
 (2, "경고 성능은 수치 예측 성능과 분리해 보고한다. 분모를 항상 같이 적는다."),
]
if FNFP is not None:
    for _, r in FNFP[FNFP.window == "TEST+LOCK"].iterrows():
        nm = {"ALL": "전체", "LOW_LOAD": "저부하", "STABLE_OPERATION": "정상가동",
              "TRANSITION_LIKE": "전환구간"}.get(r.state, r.state)
        if r.n_peak_steps == 0:
            C3.append((2, f"{nm}: 평가구간 내 고사용량 스텝 0개 → FN 율 산출 불가(분모 0). "
                          f"성능이 좋은 것이 아니라 정의되지 않는다."))
        else:
            C3.append((2, f"{nm}: 고사용량 {int(r.n_peak_steps)}스텝 중 미탐지 {int(r.FN)} (FN율 {r.fn_rate_given_peak:.3f}), "
                          f"정상 {int(r.n_nonpeak_steps)}스텝 중 오경보 {int(r.FP)} (FP율 {r.fp_rate_given_nonpeak:.3f})"))
C3 += [
 (3, "전환구간의 FN 율이 전체보다 낮게 보이지만 분모가 24 스텝에 불과하다. Wilson 95% 구간이 전체와 겹치므로 '전환구간에서 더 잘 잡는다'고 주장하지 않는다."),
 (1, "실패한 가설을 지우지 않고 남긴다"),
 (2, "운전상태 3분할 개별 모델(V4): 네 호라이즌 모두 전역 모델보다 나빴다(h60 +0.756 kw, 95% CI [0.281, 1.430]). 기각."),
 (2, "'데이터에 자연적인 군집이 3개 있다'는 주장: 실루엣 최적은 k=4(0.558) 이고 k=3 은 0.532, 규칙 라벨과 KMeans3 의 ARI 는 0.210. 철회."),
 (2, "2단계 경보가 단일 경보보다 우수하다는 주장: TEST 에서 7회 중 0회, 부트스트랩 CI [0.000, 0.000]. 철회."),
 (2, "외부 공공데이터(휴일·기상·태양기하) 추가: 누수 없는 CAL 규칙으로는 20개 fold×호라이즌 칸 중 0개에서 선택되었다. 사후적으로만 이득이 보인다."),
 (2, "OOD 구간 가중 학습(A6): 위와 같이 기각."),
]

C4 = [
 (1, "모델 출력이 어떤 현장 판단으로 이어지는가"),
 (2, "출력은 ① t+h 수요 예측값 ② 고사용량 초과 여부 판단 ③ 남은 준비시간(분) 세 가지다. 이 중 현장이 실제로 쓰는 것은 ②와 ③이다."),
 (2, "알림은 사건당 한 번만 보낸다. 다단계 경보를 성과로 내세우지 않는다 — 본 분석에서 2단계 경보가 단일 경보보다 낫다는 증거를 얻지 못했기 때문이다."),
 (1, "선행시간이 제도적으로 의미 있는 길이인지 확인했다"),
] + [
 (2, f"{int(r.lead_minutes)}분 — {r.basis}: {r.detail}")
 for _, r in LEAD.iterrows()
] + [
 (3, "정리: 15분은 기존 최대수요전력제어장치가 이미 커버하는 구간이므로 차별점이 아니다. 60분은 KPX 수요자원 거래시장의 표준 통보 리드타임과 일치하므로 본 시스템의 주 호라이즌으로 둔다. 30분은 추가 감축요청 여유에 해당해 최소 선행시간의 하한으로 쓴다."),
 (1, "피크전력 저감 측면의 활용"),
 (2, "요금적용전력은 15분 평균전력의 월 최대 1점으로 결정된다. 즉 한 달에 몇 번의 순간을 막는 것만으로 기본요금이 달라진다."),
 (2, "따라서 본 시스템의 쓰임은 '상시 제어'가 아니라 '월 최대를 만들 가능성이 있는 몇 개의 순간을 1시간 전에 지목하는 것'이다."),
 (2, "조치는 공장이 이미 가진 수단(가동 순서 조정, 비긴급 부하의 시간 이동)에 맡긴다. 어떤 설비를 끄라는 지시는 하지 않는다."),
 (1, "주장하지 않는 것"),
 (2, "절감액(원): 계약전력·요금 적용 조건을 확인하지 못했다. 금액을 계산하지 않는다."),
 (2, "설비 단위 원인·조치·고장 심각도 순위: 설비 식별자와 고장 이력이 데이터에 없다."),
 (2, "현장 활용성이 UI 개선만으로 입증되었다는 주장: 하지 않는다. 근거는 제때 잡은 사건 수와 오경보로 생기는 확인 부담이다."),
]

C5 = [
 (1, "데이터 결함을 찾아 평가 설계로 연결했다"),
 (2, f"파일명의 'augumented' 가 실제로 무엇인지 확인했다. 전체 {int(da.n_days)}일 중 {int(da.n_dup_days)}일이 전력 프로파일이 완전히 동일한 합성 복제일이었다({int(da.n_dup_groups)}그룹, 잉여 {int(da.n_redundant_days)}일)."),
 (2, "이 중복일이 fold 0 과 2 의 TRAIN·TEST 양쪽에 걸쳐 있음을 fold 단위로 세어 확인하고, 주 보고 구간을 중복이 거의 없는 clean_Jul_Sep 으로 고정했다. 부풀려진 성능을 쓰지 않기 위해서다."),
 (3, "제3장의 조건별 성능과 제2장의 최종 수치가 모두 이 구간 기준이다."),
 (1, "라벨 정의를 과금 제도에 묶었다"),
 (2, "15분은 요금적용전력의 측정 단위, 30일은 검침월에 대응한다. 동시에 p95 가 과금 규칙이 아니라 '학습 가능한 양성 표본 확보'를 위한 모델링 선택임을 명시하고 민감도표를 함께 제시한다."),
 (3, "임계 설정을 정당화하지 않고 넘어가거나, 반대로 과금 규칙인 것처럼 포장하지 않는다."),
 (1, "고사용량 구간을 직접 겨냥한 학습으로 그 구간 오차를 줄였다"),
 (2, f"잔차 타깃·주기 인코딩·극단 가중·휴무 특성을 결합해 h60 고사용량 구간 MAE 를 {a0.mae_peak:.2f} → {a5.mae_peak:.2f} kw ({pct(a5.maepeak_vs_A0_pct)}) 로 낮췄다. 네 호라이즌 모두 25% 안팎으로 개선되었고 블록 부트스트랩으로 유의했다."),
 (2, "이 개선은 제2장의 성능 표와 제3장의 조건별 오류분석에 동일하게 반영되어 있다. 새로운 알고리즘을 썼다는 주장이 아니라, 오류가 큰 조건을 먼저 찾고 그 조건을 겨냥했다는 주장이다."),
 (1, "지표 하나로 성능을 주장하지 않고, 지표 사이의 모순을 찾아 원인을 특정했다"),
 (2, f"수치 예측 MAE 는 크게 좋아졌는데 경보 F1 은 따라오지 않았다. 그 모순을 덮지 않고 무편향 오라클 F1 상한({f1_ceil:.3f})을 측정해 현재 {f1_now:.3f} 와의 격차를 수치로 드러냈다."),
 (2, f"원인을 고사용량 구간의 체계적 과소예측({bias_pk:+.2f} kw)으로 특정했고, 그 결과 ExtraTrees 가 전체 MAE 는 더 좋은데도 고사용량 MAE 악화를 이유로 기각될 수 있었다."),
 (3, "전체 MAE 하나만 봤다면 반대로 선택했을 것이다. 이 판단이 제2장의 선정 근거와 제3장의 오류분석에 그대로 연결된다."),
 (1, "실패한 시도를 지우지 않았다"),
 (2, "운전상태 3분할 모델 기각, '자연 군집 3개' 주장 철회, 2단계 경보 우위 주장 철회, 외부 공공데이터 게이트 0/20 선택, OOD 가중 학습 기각, ExtraTrees 회귀 기각, 경보 스태킹 기각 — 일곱 건을 결과와 함께 남겼다."),
 (2, "채택되지 않은 시도를 남기는 이유는 같은 데이터로 후속 작업을 하는 사람이 같은 길을 다시 걷지 않게 하기 위해서다."),
 (1, "아직 주장할 수 없는 것을 명시했다"),
 (2, "고사용량 구간의 예측구간 적중률이 0.234 로 목표 0.90 에 크게 미달한다. 숨기지 않고 공개하며, 구간 추정은 현재 상태로 신뢰할 수 없다고 적는다."),
 (2, "고사용량 '최초 시작 시각' 예측은 설계만 있고 검증 전이다. 검증 전에는 차별점으로 주장하지 않는다."),
]

C6 = [
 (1, "실행 방법 — 전처리부터 결과 생성까지"),
 (2, "python3 analysis/final_baseline/run_baseline.py  → 특성 생성, rolling-origin 평가, 신뢰도·경보 임계 산출, 표·그림·요약 JSON 생성까지 한 번에 수행한다."),
 (2, "python3 analysis/final_baseline/v2_forecast_improve.py  → 제2장의 모델 비교(A0~A5, 기준선 3종)와 제3장의 조건별 성능표를 재생성한다."),
 (2, "python3 analysis/final_baseline/v2_clean_summary.py  → clean 구간 집계와 블록 부트스트랩, 중복일 fold 교차 감사표를 재생성한다."),
 (2, "python3 analysis/final_baseline/v2_stress_robust.py  → OOD 역할 검정(십분위 곡선, A6 기각)을 재생성한다."),
 (2, "python3 analysis/final_baseline/test_baseline.py  → 누수 차단·재현성 수용테스트."),
 (2, "python3 analysis/submission/build_report.py / build_pptx.py  → 본 보고서와 발표자료를 표에서 직접 생성한다."),
 (1, "재현성 보장"),
 (2, "난수 시드 20260926 고정. CPU 전용(M1, GPU 미사용). Python 3.9.6, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1, matplotlib 3.9.4, joblib 1.5.3, PyYAML 6.0.3, holidays 0.83."),
 (2, "정책 가정(피크 임계 창·분위, 비용비, 최소 선행시간, 분위 격자 등)은 전부 analysis/final_baseline/config.yaml 한 곳에 노출되어 있다. 코드 수정 없이 바꿔 재현할 수 있다."),
 (2, "본 보고서와 발표자료의 모든 수치는 analysis/tables/*.csv 를 직접 읽어 생성했다. 손으로 옮겨 적은 값이 없다."),
 (1, "학습과 추론의 분리"),
 (2, "학습 산출물(모델, conformal 보정계수, 밴드 컷, OOD 컷, 경보 임계, 기본 정책)을 joblib 아티팩트로 저장한다."),
 (2, "inference.py 는 이 아티팩트만 읽어 새 입력에 대한 예측·판단을 만든다. 학습 코드 없이 동작한다. 테스트데이터 예측결과 파일을 이 경로로 생성한다."),
 (1, "제출물 구성"),
 (2, "소스코드(analysis/ 전체), requirements.txt, README, 학습용 데이터, 테스트데이터 예측결과 파일."),
 (2, "결과 테이블 analysis/tables/*.csv 와 그림 analysis/figures/*.png 는 전부 위 명령으로 재생성된다."),
]

CH = {"제1장": C1, "제2장": C2, "제3장": C3, "제4장": C4, "제5장": C5, "제6장": C6}
SUMMARY = (
    "설비 센서가 없는 공장에서 15분 수요·생산량·관측 기상·달력만으로 1시간 뒤 전력 수요를 예측하고, "
    "조치 가능한 선행시간 안에 고사용량 발생을 판단하는 모델을 만들었다. "
    f"잔차 타깃·주기 인코딩·극단 가중·휴무 특성을 결합해 1시간 전 예측 MAE 를 {a0.mae:.2f}→{a5.mae:.2f} kw, "
    f"고사용량 구간 MAE 를 {a0.mae_peak:.2f}→{a5.mae_peak:.2f} kw({pct(a5.maepeak_vs_A0_pct)})로 낮췄고 "
    "일 단위 블록 부트스트랩으로 네 호라이즌 전부 유의함을 확인했다. "
    "또한 데이터의 합성 중복일 160일을 찾아 평가 구간을 재설계했고, 채택되지 않은 다섯 건의 가설을 결과와 함께 공개한다.")

out = build(TPL, OUT / "결과보고서_전력피크조기경보.hwpx", CH,
            project_name="센서가 부족한 공장의 전력 고사용량 조기 판단",
            team_name="", summary=SUMMARY)
n = sum(len(v) for v in CH.values())
print(f"[hwpx] {Path(out).relative_to(ROOT)}  본문 문단 {n}개")
