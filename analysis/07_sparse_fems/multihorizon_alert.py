"""07-L. 다중 horizon 경보 — 15/30/45/60분 중 언제 경고해야 하는가.

기존 경보 실험(07H)은 h60 단독이었다. 여기서는 h1~h4 OOF 예측을 모두 만들고
  (1) horizon 별 선별력(AP/AUC), (2) 이벤트 단위 포착률과 선행시간,
  (3) 단일/OR/2-of-4/에스컬레이션 정책을 '같은 경보 예산'에서 비교한다.
피크 라벨은 07K 권고 정의(직전 30일 p95, 적응형·leak-free)를 쓴다.
이벤트 = 연속된 피크 스텝(15분 간격 1칸 공백 허용)을 하나로 묶은 것.
선행시간 = 이벤트 시작 전에 경보가 발령된 가장 이른 시점까지의 분.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, save_table, fig_path, mpl, SEED
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score
plt = mpl()

F = _fe.LEVELS["L3_+production"]
q = build()
d = q.dropna(subset=F + [f"y_h{h}" for h in _fe.HORIZONS]).reset_index(drop=True)
# 적응형 피크 임계 (과거만): 직전 30일 p95
thr_series = d.kw.shift(1).rolling(96 * 30, min_periods=96 * 7).quantile(.95)
d["thr_adaptive"] = thr_series
d = d.dropna(subset=["thr_adaptive"]).reset_index(drop=True)

folds = list(TimeSeriesSplit(n_splits=5).split(d))
oof = {h: np.full(len(d), np.nan) for h in _fe.HORIZONS}
for itr, ite in folds:
    for h in _fe.HORIZONS:
        m = HistGradientBoostingRegressor(random_state=SEED).fit(d.iloc[itr][F].values,
                                                                 d.iloc[itr][f"y_h{h}"].values)
        oof[h][ite] = m.predict(d.iloc[ite][F].values)
o = d[~np.isnan(oof[1])].copy().reset_index(drop=True)
keep = ~np.isnan(oof[1])
for h in _fe.HORIZONS:
    o[f"pred_h{h}"] = oof[h][keep]
    # t 시점 결정으로 t+h 를 맞히는 문제. 라벨 임계는 t 시점에 알 수 있는 적응형 임계.
    o[f"peak_h{h}"] = (o[f"y_h{h}"] >= o.thr_adaptive).astype(int)
# 실제 시각 기준 피크 여부 (이벤트 정의용): kw(t) 가 그 시점 임계를 넘었는가
o["peak_now"] = (o.kw >= o.thr_adaptive).astype(int)

# ---- horizon 별 선별력
rows = []
for wtag, sub in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean])]:
    for h in _fe.HORIZONS:
        y, p = sub[f"peak_h{h}"].values, sub[f"pred_h{h}"].values
        if y.sum() < 10:
            continue
        rows.append(dict(window=wtag, horizon_min=h * 15, n=len(sub), n_peak=int(y.sum()),
                         prevalence=float(y.mean()), auc=float(roc_auc_score(y, p)),
                         ap=float(average_precision_score(y, p)),
                         ap_lift=float(average_precision_score(y, p) / y.mean()),
                         mae=float(np.abs(sub[f"y_h{h}"] - p).mean())))
save_table(pd.DataFrame(rows), "07L_horizon_screening_power")


def events(peak, ts, max_gap=1):
    """연속 피크 스텝 -> 이벤트 구간 인덱스 목록."""
    idx = np.flatnonzero(peak == 1)
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev <= max_gap + 1:
            prev = i
            continue
        out.append((start, prev)); start, prev = i, i
    out.append((start, prev))
    return out


POLICIES = {
    "h15_only": lambda A: A[1],
    "h30_only": lambda A: A[2],
    "h45_only": lambda A: A[3],
    "h60_only": lambda A: A[4],
    "any_horizon_OR": lambda A: A[1] | A[2] | A[3] | A[4],
    "at_least_2_of_4": lambda A: (A[1].astype(int) + A[2] + A[3] + A[4]) >= 2,
    "escalate_h60_then_h15": lambda A: A[4] & (A[1] | A[2]),
    "long_only_OR_h45_h60": lambda A: A[3] | A[4],
}
QS = np.round(np.arange(.50, .996, .01), 3)
crows = []
for wtag, sub in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean].reset_index(drop=True))]:
    sub = sub.reset_index(drop=True)
    ev = events(sub.peak_now.values, sub.ts15.values)
    if not len(ev):
        continue
    for qv in QS:
        A = {h: (sub[f"pred_h{h}"].values >= float(np.quantile(sub[f"pred_h{h}"], qv))) for h in _fe.HORIZONS}
        # 경보가 '언제 발령됐는지'를 실제 시각으로 옮긴다: h 경보는 t 에 나오고 t+h 를 가리킨다.
        for pname, fn in POLICIES.items():
            al = np.asarray(fn(A)).astype(bool)
            # 스텝 단위 지표: 각 경보가 가리키는 미래 시점이 실제 피크였는가 (가장 긴 horizon 기준 통합)
            tgt = np.zeros(len(sub), bool)
            for h in _fe.HORIZONS:
                if pname.startswith(f"h{h*15}") or pname in ("any_horizon_OR", "at_least_2_of_4",
                                                             "escalate_h60_then_h15", "long_only_OR_h45_h60"):
                    tgt |= sub[f"peak_h{h}"].values.astype(bool)
            tp = int((al & tgt).sum()); fp = int((al & ~tgt).sum())
            # 이벤트 단위: 이벤트 시작 전 최대 60분 안에 경보가 있었나 + 선행시간
            det, leads = 0, []
            for s, e in ev:
                w = al[max(0, s - 4):s + 1]
                if w.any():
                    det += 1
                    first = max(0, s - 4) + int(np.flatnonzero(w)[0])
                    leads.append((s - first) * 15)
            crows.append(dict(window=wtag, policy=pname, quantile=qv,
                              alert_frequency=float(al.mean()), alerts_per_day=float(al.mean() * 96),
                              step_precision=tp / max(tp + fp, 1),
                              n_events=len(ev), events_detected=det, event_recall=det / len(ev),
                              median_lead_min=float(np.median(leads)) if leads else np.nan,
                              mean_lead_min=float(np.mean(leads)) if leads else np.nan,
                              lead_ge_30min_share=float(np.mean([l >= 30 for l in leads])) if leads else np.nan))
cur = pd.DataFrame(crows)
save_table(cur, "07L_multihorizon_alert_curve")

# ---- 같은 경보 예산에서의 비교 (목표 alerts/day 에 가장 가까운 점)
brows = []
for (wtag, pname), g in cur.groupby(["window", "policy"]):
    for budget in [5, 10, 20]:
        r = g.iloc[(g.alerts_per_day - budget).abs().argsort().iloc[0]]
        brows.append(dict(window=wtag, policy=pname, target_alerts_per_day=budget, **{
            k: r[k] for k in ["quantile", "alerts_per_day", "step_precision", "event_recall",
                              "median_lead_min", "mean_lead_min", "lead_ge_30min_share", "n_events"]}))
bud = pd.DataFrame(brows)
save_table(bud, "07L_policy_at_matched_budget")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
c = cur[cur.window == "oof_clean_Jul_Sep"]
for pname in POLICIES:
    s = c[c.policy == pname].sort_values("alerts_per_day")
    axes[0].plot(s.alerts_per_day, s.event_recall, "-", label=pname)
    axes[1].plot(s.alerts_per_day, s.step_precision, "-", label=pname)
    axes[2].plot(s.alerts_per_day, s.mean_lead_min, "-", label=pname)
axes[0].set_ylabel("이벤트 포착률"); axes[1].set_ylabel("스텝 정밀도"); axes[2].set_ylabel("평균 선행시간(분)")
for ax in axes:
    ax.set_xlabel("일 경보 수"); ax.set_xlim(0, 40); ax.legend(fontsize=6)
fig.suptitle("07L 다중 horizon 경보 정책 — clean_Jul_Sep (OOF)", fontsize=9)
fig.tight_layout(); fig.savefig(fig_path("07L_multihorizon_alert")); plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False))
print(bud[bud.window == "oof_clean_Jul_Sep"].to_string(index=False))
