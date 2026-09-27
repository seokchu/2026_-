"""07-I. 정보 가치(VOI) / 점진적 계측 — 존재하는 정보군만 평가한다.

A. 비용 없는 한계가치: 정보군 추가에 따른 ΔMAE / Δ피크AP / Δ난조건MAE / Δcoverage
B. 비용 민감도: 가상의 정규화 취득비용(1,2,5,10) 하의 gain/cost. 실제 센서 가격 아님.
그리고 "최고 성능의 90/95/98% 를 유지하는 최소 정보집합"을 계산한다.
"""
import numpy as np, pandas as pd
from _fe import save_table, fig_path, mpl, TAB
plt = mpl()

ab = pd.read_csv(TAB / "07B_information_ablation.csv", encoding="utf-8-sig")
bl = pd.read_csv(TAB / "07C_hard_condition_by_level.csv", encoding="utf-8-sig")
bs = pd.read_csv(TAB / "07B_ablation_bootstrap.csv", encoding="utf-8-sig")
CHAIN = ["L0_power_only", "L1_+calendar", "L2_+weather", "L3_+production", "L4_+latent_context"]
GROUP = {"L1_+calendar": "calendar", "L2_+weather": "weather",
         "L3_+production": "production", "L4_+latent_context": "latent_context"}
COSTS = {"calendar": 1, "weather": 2, "production": 5, "latent_context": 2}   # 가상 정규화 비용

hard_mae = bl.groupby("level").mae.mean() if len(bl) else None
rows = []
for wtag in ab.window.unique():
    for h in sorted(ab.horizon_min.unique()):
        d = ab[(ab.window == wtag) & (ab.horizon_min == h)].set_index("level")
        chain = [c for c in CHAIN if c in d.index]
        for prev, cur in zip(chain, chain[1:]):
            g = GROUP[cur]
            b = bs[(bs.window == wtag) & (bs.horizon_min == h) & (bs.level == prev)]
            rows.append(dict(window=wtag, horizon_min=h, added_group=g, from_level=prev, to_level=cur,
                             mae_before=d.loc[prev, "mae"], mae_after=d.loc[cur, "mae"],
                             d_mae=d.loc[cur, "mae"] - d.loc[prev, "mae"],
                             d_mae_pct=(d.loc[cur, "mae"] - d.loc[prev, "mae"]) / d.loc[prev, "mae"],
                             d_peak_ap=d.loc[cur, "peak_ap"] - d.loc[prev, "peak_ap"],
                             d_mae_peak=d.loc[cur, "mae_peak"] - d.loc[prev, "mae_peak"],
                             d_coverage=d.loc[cur, "cqr_coverage"] - d.loc[prev, "cqr_coverage"],
                             d_hard_mae=(hard_mae.get(cur, np.nan) - hard_mae.get(prev, np.nan))
                             if hard_mae is not None else np.nan,
                             cost_proxy=COSTS[g],
                             voi_per_cost=-(d.loc[cur, "mae"] - d.loc[prev, "mae"]) / COSTS[g]))
voi = pd.DataFrame(rows)
save_table(voi, "07I_marginal_value_of_information")

# ---- 성능 유지율 & 최소 정보집합
ret_rows = []
for wtag in ab.window.unique():
    for h in sorted(ab.horizon_min.unique()):
        d = ab[(ab.window == wtag) & (ab.horizon_min == h)].set_index("level")
        best_mae, best_ap = d.mae.min(), d.peak_ap.max()
        base = d.loc["L0_power_only", "mae"]
        for l in d.index:
            # 예측 유지율: persistence 대비 개선분을 최고 수준 대비 몇 % 확보했는가
            gain = base - d.loc[l, "mae"]
            gain_best = base - best_mae
            ret_rows.append(dict(window=wtag, horizon_min=h, level=l,
                                 mae=d.loc[l, "mae"], mae_retained_pct=float(gain / gain_best)
                                 if gain_best > 0 else np.nan,
                                 peak_ap=d.loc[l, "peak_ap"],
                                 peak_ap_retained_pct=float(d.loc[l, "peak_ap"] / best_ap),
                                 coverage=d.loc[l, "cqr_coverage"],
                                 n_features=d.loc[l, "n_features"],
                                 cum_cost=sum(COSTS[GROUP[c]] for c in CHAIN[1:CHAIN.index(l) + 1])
                                 if l in CHAIN else np.nan))
ret = pd.DataFrame(ret_rows)
save_table(ret, "07I_performance_retention")

min_rows = []
for wtag in ret.window.unique():
    for h in sorted(ret.horizon_min.unique()):
        d = ret[(ret.window == wtag) & (ret.horizon_min == h)]
        d = d[d.level.isin(CHAIN)].sort_values("cum_cost")
        for target in (.90, .95, .98):
            ok = d[d.mae_retained_pct >= target]
            min_rows.append(dict(window=wtag, horizon_min=h, retention_target=target,
                                 minimum_level=ok.iloc[0].level if len(ok) else "none",
                                 cum_cost=ok.iloc[0].cum_cost if len(ok) else np.nan,
                                 achieved=ok.iloc[0].mae_retained_pct if len(ok) else np.nan))
save_table(pd.DataFrame(min_rows), "07I_minimum_information_set")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for wtag, ax in zip(["all_2021", "clean_Jul_Sep"], axes):
    d = voi[voi.window == wtag].groupby("added_group")[["d_mae", "voi_per_cost"]].mean()
    ax.bar(range(len(d)), -d.d_mae, color="#27b")
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.index, rotation=15, fontsize=7)
    ax.set_ylabel("MAE 감소량 (kw, 양수=개선)"); ax.set_title(f"{wtag} — 정보군 한계가치 (horizon 평균)")
    ax.axhline(0, color="k", lw=.8)
fig.tight_layout(); fig.savefig(fig_path("07I_value_of_information")); plt.close(fig)
print(voi.groupby(["window", "added_group"])[["d_mae", "d_mae_pct", "d_peak_ap", "voi_per_cost"]].mean().to_string())
print(pd.DataFrame(min_rows).to_string(index=False))
