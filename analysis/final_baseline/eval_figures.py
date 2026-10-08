"""37. 성능 평가 그래프 2종 — (1) 예측 성능 (표준) (2) 제안 구성의 특장점.

템플릿 색(#0B5BDB)과 무채색만 쓴다. 수치는 전부 analysis/tables 의 CSV 에서 읽는다.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from common import fig_path, mpl, TAB                                      # noqa: E402
plt = mpl()
from sklearn.metrics import precision_recall_curve                         # noqa: E402

BLUE, DARK, GREY, LIGHT, RED = "#0B5BDB", "#111111", "#8A9099", "#D9DEE5", "#C0392B"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#AAB0B8", "grid.color": "#E7EAEE",
                     "font.size": 10})


def L(n):
    return pd.read_csv(TAB / f"{n}.csv", encoding="utf-8-sig")


RSTD = L("34_standard_benchmark_regression_clean")
CSTD = L("34_standard_benchmark_alert_clean")
AF = L("36_alert_final_metrics")
AS_ = pd.read_csv(TAB / "36_alert_final_scores.csv", encoding="utf-8-sig")
FM = L("23_baseline_final_metrics").set_index("horizon_min")
FDG = L("33_f1_ceiling_diagnosis")
DUPF = L("31_duplicate_fold_crossing")
LEAD = L("31_leadtime_basis")
afc = AF[AF.window == "clean_Jul_Sep"]
OURS = "제안 (margin 특성 + RF)"
PREV = "이전 구성 (기본특성 + RF)"

# ─────────────────────────────────────────── 평가 1. 예측 성능 (표준 지표)
fig = plt.figure(figsize=(16, 9.2))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3, height_ratios=[1, 1],
                      left=.07, right=.985, top=.95, bottom=.09)

ax = fig.add_subplot(gs[0, :2])
r = RSTD.sort_values("mae", ascending=False)
r = r[r.model != "Seasonal naive (24h 전)"]
col = [BLUE if "제출" in m else GREY for m in r.model]
ax.barh(range(len(r)), r.mae, color=col, height=.62)
for i, (m, v, pk) in enumerate(zip(r.model, r.mae, r.mae_peak)):
    ax.text(v + .25, i, f"{v:.2f}", va="center", fontsize=9,
            color=BLUE if "제출" in m else DARK)
ax.set_yticks(range(len(r))); ax.set_yticklabels(r.model, fontsize=9)
ax.set_xlabel("MAE (kw) — 낮을수록 좋음"); ax.grid(axis="x")
ax.set_title("① 수치 예측 — 표준 회귀 모델 14종, 1시간 전 (clean 구간 OOF)", fontsize=12, loc="left")

ax = fig.add_subplot(gs[0, 2])
r2 = RSTD[RSTD.model.isin(["제출 모델" if False else "HistGBM + 극단가중 (제출 모델)",
                           "Extra Trees", "Random Forest", "HistGradientBoosting",
                           "MLP (64,32)", "Linear Regression", "Naive (직전값)"])]
r2 = r2.sort_values("mae_peak")
col = [BLUE if "제출" in m else GREY for m in r2.model]
ax.barh(range(len(r2)), r2.mae_peak, color=col, height=.6)
ax.set_yticks(range(len(r2)))
ax.set_yticklabels([m.replace(" + 극단가중 (제출 모델)", "\n+극단가중(제출)") for m in r2.model],
                   fontsize=8)
for i, v in enumerate(r2.mae_peak):
    ax.text(v + .3, i, f"{v:.1f}", va="center", fontsize=8.5)
ax.set_xlabel("고사용량 구간 MAE (kw)"); ax.grid(axis="x")
ax.set_title("② 고사용량 구간만 — 제출 모델이 최저", fontsize=11, loc="left")

ax = fig.add_subplot(gs[1, :2])
c = CSTD.sort_values("f1", ascending=False).head(8)
mets = ["precision", "recall", "f1", "mcc", "pr_auc"]
lbl = ["정밀도", "재현율", "F1", "MCC", "PR-AUC"]
x = np.arange(len(c)); w = .15
shades = ["#0B5BDB", "#4A82E4", "#8AAEEE", "#111111", "#8A9099"]
for j, (m, lb) in enumerate(zip(mets, lbl)):
    ax.bar(x + (j - 2) * w, c[m].values, w, label=lb, color=shades[j])
ax.set_xticks(x); ax.set_xticklabels(c.model, rotation=18, ha="right", fontsize=9)
ax.legend(ncol=5, fontsize=9, frameon=False, loc="upper right")
ax.set_ylim(0, 1.05); ax.grid(axis="y")
ax.set_title("③ 고사용량 경보 — 표준 분류 모델 상위 8종 (clean, 1시간 전)", fontsize=12, loc="left")

ax = fig.add_subplot(gs[1, 2])
for nm, c_, ls in [(OURS, BLUE, "-"), (PREV, DARK, "--"),
                   ("persistence 규칙 (베이스라인)", GREY, ":")]:
    g = AS_[(AS_.clean) & (AS_.horizon_min == 60) & (AS_.model == nm)]
    if not len(g): continue
    p, rc, _ = precision_recall_curve(g.label.values.astype(int), g.score.values)
    ax.plot(rc, p, ls, color=c_, lw=2, label=nm.split(" (")[0])
base = AS_[(AS_.clean) & (AS_.horizon_min == 60) & (AS_.model == OURS)].label.mean()
ax.axhline(base, color=RED, lw=1, ls="-.", label=f"무작위 {base:.3f}")
ax.set_xlabel("재현율"); ax.set_ylabel("정밀도"); ax.set_ylim(0, 1)
ax.legend(fontsize=8.5, frameon=False); ax.grid(True)
ax.set_title("④ 정밀도–재현율 곡선 (1시간 전)", fontsize=11, loc="left")

fig.savefig(fig_path("37_eval1_prediction"), dpi=150); plt.close(fig)

# ─────────────────────────────────────────── 평가 2. 제안 구성의 특장점
fig = plt.figure(figsize=(16, 9.2))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                      left=.055, right=.985, top=.95, bottom=.07)

ax = fig.add_subplot(gs[0, 0])
h = sorted(afc.horizon_min.unique())
o = [afc[(afc.model == OURS) & (afc.horizon_min == x)].precision.iloc[0] for x in h]
p_ = [afc[(afc.model == PREV) & (afc.horizon_min == x)].precision.iloc[0] for x in h]
x = np.arange(len(h))
ax.bar(x - .2, p_, .4, color=GREY, label="이전 구성")
ax.bar(x + .2, o, .4, color=BLUE, label="제안 구성")
for i, (a, b) in enumerate(zip(p_, o)):
    ax.text(i + .2, b + .012, f"{b:.3f}", ha="center", fontsize=9, color=BLUE, weight="bold")
    ax.text(i - .2, a + .012, f"{a:.3f}", ha="center", fontsize=9, color=DARK)
ax.set_xticks(x); ax.set_xticklabels([f"{v}분" for v in h])
ax.set_ylabel("정밀도"); ax.set_ylim(0, .75); ax.grid(axis="y")
ax.legend(fontsize=9, frameon=False)
ax.set_title("① 정밀도 — 전 호라이즌 상승", fontsize=11.5, loc="left")

ax = fig.add_subplot(gs[0, 1])
a_ = [afc[(afc.model == OURS) & (afc.horizon_min == x)].alerts_per_day.iloc[0] for x in h]
b_ = [afc[(afc.model == PREV) & (afc.horizon_min == x)].alerts_per_day.iloc[0] for x in h]
ax.bar(x - .2, b_, .4, color=GREY, label="이전 구성")
ax.bar(x + .2, a_, .4, color=BLUE, label="제안 구성")
for i, (u, v) in enumerate(zip(b_, a_)):
    ax.text(i + .2, v + .3, f"{v:.1f}", ha="center", fontsize=9, color=BLUE, weight="bold")
    ax.text(i - .2, u + .3, f"{u:.1f}", ha="center", fontsize=9, color=DARK)
ax.set_xticks(x); ax.set_xticklabels([f"{v}분" for v in h])
ax.set_ylabel("경보 건수 / 일"); ax.grid(axis="y")
ax.set_ylim(0, max(b_) * 1.35); ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=2)
ax.set_title("② 확인 부담 — 43% 감소", fontsize=11.5, loc="left")

ax = fig.add_subplot(gs[0, 2])
g = AS_[(AS_.clean) & (AS_.horizon_min == 60)]
for nm, c_, mk in [(OURS, BLUE, "o"), (PREV, DARK, "s")]:
    gg = g[g.model == nm]
    sc, y = gg.score.values, gg.label.values.astype(bool)
    xs, ys = [], []
    for rate in np.arange(.02, .22, .005):
        pr = sc >= np.quantile(sc, 1 - rate)
        tp = (pr & y).sum(); fp = (pr & ~y).sum()
        xs.append(rate * 96); ys.append(tp / max(tp + fp, 1))
    ax.plot(xs, ys, color=c_, lw=2, label=nm.split(" (")[0])
    row = afc[(afc.model == nm) & (afc.horizon_min == 60)].iloc[0]
    ax.scatter([row.alerts_per_day], [row.precision], color=c_, marker=mk, s=70, zorder=5,
               edgecolor="white", linewidth=1.2)
ax.set_xlabel("경보 건수 / 일"); ax.set_ylabel("정밀도"); ax.grid(True)
ax.legend(fontsize=8.5, frameon=False)
ax.set_title("③ 낮은 경보량 구간에서 더 정확 (점 = 채택 운영점)", fontsize=11.5, loc="left")

ax = fig.add_subplot(gs[1, 0])
cov = [FM.loc[x, "coverage"] for x in h]
covp = [FM.loc[x, "coverage_peak_only"] for x in h]
ax.plot(x, cov, "o-", color=DARK, lw=2, label="전체 구간")
ax.plot(x, covp, "s-", color=BLUE, lw=2, label="고사용량 구간")
ax.axhline(.90, color=RED, ls="--", lw=1.2, label="목표 0.90")
ax.set_xticks(x); ax.set_xticklabels([f"{v}분" for v in h])
ax.set_ylim(0, 1.0); ax.set_ylabel("예측구간 적중률"); ax.grid(True)
ax.legend(fontsize=8.5, frameon=False)
ax.set_title("④ 예측구간 — 불확실성을 수치로 제시", fontsize=11.5, loc="left")

ax = fig.add_subplot(gs[1, 1])
bands = ["HIGH", "MEDIUM", "LOW"]
vals = [FM.loc[60, f"mae_band_{b}"] for b in bands]
ax.bar(bands, vals, color=[BLUE, "#6E9BE8", LIGHT], width=.55)
for i, v in enumerate(vals):
    ax.text(i, v + .25, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
ax.set_ylabel("MAE (kw)"); ax.grid(axis="y")
ax.set_title("⑤ 신뢰도 등급이 실제 오차 순서와 일치", fontsize=11.5, loc="left")

ax = fig.add_subplot(gs[1, 2])
f1_now = float(afc[(afc.model == OURS) & (afc.horizon_min == 15)].f1.iloc[0])
f1_pers = float(afc[(afc.model == "persistence 규칙 (베이스라인)") & (afc.horizon_min == 60)].f1.iloc[0])
f1_h60 = float(afc[(afc.model == OURS) & (afc.horizon_min == 60)].f1.iloc[0])
ceil = float(FDG[FDG.model == "R4_hist_gbm"].oracle_f1_same_error.iloc[0])
names = ["persistence\n규칙", "제안\n(1시간 전)", "제안\n(15분 전)", "편향 제거 시\n상한", "완전 예측"]
vals = [f1_pers, f1_h60, f1_now, ceil, 1.0]
cols = [GREY, BLUE, BLUE, LIGHT, LIGHT]
ax.bar(names, vals, color=cols, width=.6)
for i, v in enumerate(vals):
    ax.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=9.5, weight="bold")
ax.set_ylim(0, 1.12); ax.set_ylabel("F1"); ax.grid(axis="y")
ax.tick_params(labelsize=9)
ax.set_title("⑥ 현재 위치와 남은 여지", fontsize=11.5, loc="left")

fig.savefig(fig_path("37_eval2_distinctive"), dpi=150); plt.close(fig)
print("[fig] 37_eval1_prediction.png, 37_eval2_distinctive.png")
