"""07-K. 피크 임계 정의 통일 — 08(전체구간 p95, 유병률 ~5%) vs 13(첫 폴드 p95, 11.5%).

두 숫자는 모순이 아니라 서로 다른 정의였다. 제출 전에 하나로 고정해야 하므로
후보 정의를 같은 OOF 예측(07C, h=60분) 위에서 나란히 평가한다.
평가축:
  1) 누수 여부 — 임계 산출에 평가구간 정보가 들어가는가.
  2) 유병률 안정성 — 폴드별 유병률 편차(수요 추세가 있으면 고정 임계는 드리프트한다).
  3) 정의 간 일치도 — Jaccard.
  4) 하류 정책 민감도 — 같은 점수로 비용최적 임계를 찾았을 때 정책이 질적으로 바뀌는가.
정의는 모두 t 시점까지의 관측만 쓰거나(적응형), train 구간만 쓴다(고정형).
'전체구간 p95' 만 예외적으로 누수 정의로 표시하고 비교 기준으로만 남긴다.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, save_table, fig_path, mpl, TAB
from sklearn.metrics import average_precision_score, roc_auc_score
plt = mpl()

H = 4
o = pd.read_csv(TAB / "07C_oof_predictions.csv", parse_dates=["ts15"], encoding="utf-8-sig")
q = build().sort_values("ts15").reset_index(drop=True)
o = o.merge(q[["ts15"]], on="ts15", how="left")
folds = sorted(o.fold.unique())
train_end = o.ts15.min()                                  # 07C 첫 폴드 학습구간의 끝 = OOF 시작 직전
first_tr = q[q.ts15 < train_end]                          # 07C 가 임계를 뽑은 그 구간

# 적응형 임계: t 시점까지의 과거만 사용 (shift(1) 로 현재값 제외)
kw = q.set_index("ts15").kw
roll30d = kw.shift(1).rolling(96 * 30, min_periods=96 * 7).quantile(.95)
expand = kw.shift(1).expanding(min_periods=96 * 7).quantile(.95)
month_max = kw.shift(1).groupby(q.set_index("ts15")["m"].values).cummax()
aux = pd.DataFrame({"ts15": kw.index, "thr_roll30d_p95": roll30d.values,
                    "thr_expanding_p95": expand.values, "thr_month_max_so_far": month_max.values})
o = o.merge(aux, on="ts15", how="left")

DEFS = {
    "p95_whole_period(08 방식)": dict(thr=float(np.quantile(q.kw, .95)), leak=True,
                                      note="전체 2021 구간 p95 — 평가구간 포함(누수)"),
    "p95_first_train_fold(13 방식)": dict(thr=float(np.quantile(first_tr.kw, .95)), leak=False,
                                          note="07C 첫 폴드 학습구간 p95 — 고정, leak-free"),
    "p99_first_train_fold": dict(thr=float(np.quantile(first_tr.kw, .99)), leak=False,
                                 note="같은 구간 p99 — 희소 피크"),
    "p95_trailing_30d": dict(col="thr_roll30d_p95", leak=False, note="직전 30일 p95 — 적응형"),
    "p95_expanding": dict(col="thr_expanding_p95", leak=False, note="누적 과거 p95 — 적응형"),
    "new_monthly_max": dict(col="thr_month_max_so_far", leak=False,
                            note="해당 월 지금까지의 최대 초과 = 요금 기준 신규 최대수요"),
}
lab = {}
rows = []
for name, spec in DEFS.items():
    t = np.full(len(o), spec["thr"]) if "thr" in spec else o[spec["col"]].values
    y = (o.y_true.values >= t).astype(float)
    y[np.isnan(t)] = np.nan
    lab[name] = y
    m = ~np.isnan(y)
    pv = pd.Series(y[m]).groupby(o.fold.values[m]).mean()
    cl = o.clean.values.astype(bool) & m
    rows.append(dict(definition=name, leaky=spec["leak"], note=spec["note"],
                     threshold_kw_mean=float(np.nanmean(t)), threshold_kw_min=float(np.nanmin(t)),
                     threshold_kw_max=float(np.nanmax(t)), n_labeled=int(m.sum()),
                     prevalence=float(np.nanmean(y[m])), prevalence_clean=float(y[cl].mean()),
                     prevalence_fold_min=float(pv.min()), prevalence_fold_max=float(pv.max()),
                     prevalence_fold_spread=float(pv.max() - pv.min()),
                     prevalence_by_fold=str({int(k): round(v, 3) for k, v in pv.items()})))
defs = pd.DataFrame(rows)
save_table(defs, "07K_peak_definitions")

# ---- 정의 간 Jaccard
names = list(DEFS)
J = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        m = ~np.isnan(lab[a]) & ~np.isnan(lab[b])
        ia, ib = lab[a][m] == 1, lab[b][m] == 1
        J.loc[a, b] = float((ia & ib).sum() / max((ia | ib).sum(), 1))
save_table(J.reset_index().rename(columns={"index": "definition"}), "07K_peak_definition_jaccard")

# ---- 하류 정책 민감도: 같은 OOF 점수, 정의만 바꿔 비용최적 임계를 다시 찾는다
RATIOS = [1, 5, 10, 20, 50]
QS = np.round(np.arange(.50, .999, .01), 3)
prows, srows = [], []
for wtag, msk in [("oof_all_2021", np.ones(len(o), bool)), ("oof_clean_Jul_Sep", o.clean.values.astype(bool))]:
    for name in names:
        y = lab[name]
        m = msk & ~np.isnan(y)
        yy, p = y[m].astype(int), o.pred.values[m]
        if yy.sum() < 10:
            continue
        srows.append(dict(window=wtag, definition=name, n=int(m.sum()), n_peak=int(yy.sum()),
                          prevalence=float(yy.mean()),
                          auc=float(roc_auc_score(yy, p)), ap=float(average_precision_score(yy, p)),
                          ap_over_prevalence=float(average_precision_score(yy, p) / yy.mean())))
        for r in RATIOS:
            best = None
            for qv in QS:
                a = p >= float(np.quantile(p, qv))
                tp, fp = int((a & (yy == 1)).sum()), int((a & (yy == 0)).sum())
                fn = int(yy.sum()) - tp
                c = (fp + r * fn) / (r * max(yy.sum(), 1))
                if best is None or c < best["expected_normalized_cost"]:
                    best = dict(window=wtag, definition=name, cost_ratio_FN_FP=r, quantile=qv,
                                threshold_score_kw=float(np.quantile(p, qv)),
                                expected_normalized_cost=c, precision=tp / max(tp + fp, 1),
                                recall=tp / max(int(yy.sum()), 1),
                                alerts_per_day=float(a.mean() * 96), missed=fn)
            prows.append(best)
save_table(pd.DataFrame(srows), "07K_definition_ranking_quality")
pol = pd.DataFrame(prows)
save_table(pol, "07K_definition_policy_sensitivity")

# ---- 권고 기준(사전 명시): (1) leak-free (2) 운용 가능한 유병률 2~15%
#      (3) 점수로 실제 선별 가능(AUC>=0.85, AP/유병률 >= 3)
#      (4) 구간 일관성 — 증강구간 포함(all_2021)과 원본구간(clean)의 유병률이 같아야 한다.
#          08 vs 13 의 5% / 11.5% 불일치가 정확히 이 축에서 발생했으므로 1순위 기준으로 둔다.
#      (5) 폴드 드리프트, (6) 비용비에 따른 정책 폭 — 동점 처리용
qual = pd.DataFrame(srows)
qual = qual[qual.window == "oof_clean_Jul_Sep"].set_index("definition")
spread_pol = pol[pol.window == "oof_clean_Jul_Sep"].groupby("definition")["quantile"].agg(["min", "max"])
defs["auc"] = [float(qual.loc[n, "auc"]) if n in qual.index else np.nan for n in defs.definition]
defs["ap"] = [float(qual.loc[n, "ap"]) if n in qual.index else np.nan for n in defs.definition]
defs["ap_lift_over_prevalence"] = [float(qual.loc[n, "ap_over_prevalence"]) if n in qual.index else np.nan
                                   for n in defs.definition]
defs["policy_quantile_spread"] = [float(spread_pol.loc[n, "max"] - spread_pol.loc[n, "min"])
                                  if n in spread_pol.index else np.nan for n in defs.definition]
defs["eligible"] = (~defs.leaky) & defs.prevalence.between(.02, .15) & \
                   (defs.auc >= .85) & (defs.ap_lift_over_prevalence >= 3)
cand = defs[defs.eligible].copy()
cand["window_prevalence_gap"] = (cand.prevalence - cand.prevalence_clean).abs()
# 사전 정한 사전식(lexicographic) 규칙: 구간 일관성이 1순위, 0.01 이내 동률만 폴드 드리프트로 가린다.
cand["score"] = (cand.window_prevalence_gap / 0.01).round(0) * 100 + cand.prevalence_fold_spread.rank()
defs["window_prevalence_gap"] = (defs.prevalence - defs.prevalence_clean).abs()
rec = cand.sort_values("score").iloc[0].definition
defs["recommended"] = defs.definition == rec
save_table(defs, "07K_peak_definitions")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for name in names:
    y = lab[name]
    m = ~np.isnan(y)
    pv = pd.Series(y[m]).groupby(o.fold.values[m]).mean()
    axes[0].plot(pv.index, pv.values, "o-", label=name)
axes[0].set_xlabel("OOF fold"); axes[0].set_ylabel("피크 유병률"); axes[0].legend(fontsize=6)
axes[0].set_title("정의별 유병률 드리프트")
d = pol[pol.window == "oof_clean_Jul_Sep"]
for name in d.definition.unique():
    s = d[d.definition == name].sort_values("cost_ratio_FN_FP")
    axes[1].plot(s.cost_ratio_FN_FP, s["quantile"], "o-", label=name)
axes[1].set_xscale("log"); axes[1].set_xlabel("C_FN / C_FP"); axes[1].set_ylabel("비용최적 임계 분위")
axes[1].set_title("정의별 최적 정책 (clean 구간)"); axes[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(fig_path("07K_peak_definition")); plt.close(fig)
print(defs[["definition", "leaky", "eligible", "threshold_kw_mean", "prevalence", "prevalence_clean",
            "window_prevalence_gap", "prevalence_fold_spread", "recommended"]].to_string(index=False))
print(pd.DataFrame(srows).to_string(index=False))
print("권고 정의:", rec)
