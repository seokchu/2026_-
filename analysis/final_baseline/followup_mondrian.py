"""후속 R. 피크구간 조건부(Mondrian) conformal 을 Baseline v1 프로토콜에서 재평가.

문제: Baseline v1 의 전체 coverage 0.859, 피크구간 coverage 0.119 (보고서 23 §H).
질문: 그룹별 보정분위를 쓰면 피크구간 coverage 가 개선되는가, 아니면 구간폭만 넓어지는가?
그룹 정의는 결정시점 정보만 사용한다(미래 관측 금지):
  G1 regime     : LOW_LOAD / STABLE_OPERATION / TRANSITION_LIKE (ramp 컷은 TRAIN p90)
  G2 peak_now   : 현재 수요가 직전 30일 p95 를 넘었는가 (현재값만 사용)
  G3 level3     : 현재 수요의 TRAIN 3분위 구간
보정분위는 각 폴드 CAL 에서 그룹별로 산출한다. h60 만 본다.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, fig_path, mpl                  # noqa: E402
import features as FT, models as MD, pipeline as PP           # noqa: E402
from sklearn.model_selection import TimeSeriesSplit           # noqa: E402
plt = mpl()

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
SEED, rc = cfg["seed"], cfg["reliability"]
LO, HI = rc["quantile_levels"]
ALPHA = rc["conformal_alpha"]
H = 4

q, mode = FT.build(cfg["external"]["use_external_data"], cfg["peak"]["window_days"],
                   cfg["peak"]["quantile"], cfg["peak"]["min_days"])
d = FT.common_rows(q, mode)
F = FT.CORE
folds = list(TimeSeriesSplit(n_splits=cfg["data"]["n_splits"]).split(d))
rows = []
for fi, (itr, ite) in enumerate(folds):
    cut = int(len(itr) * (1 - cfg["data"]["cal_fraction"]))
    tr, ca = itr[:cut], itr[cut:]
    TR, CA, TE = d.iloc[tr], d.iloc[ca], d.iloc[ite]
    ycol = f"y_h{H}"
    ramp_cut = float(np.quantile(TR.kw_ramp4.abs(), cfg["regime"]["ramp_quantile"]))
    m_lo = MD.quantile_model(LO, SEED).fit(TR[F].values, TR[ycol].values)
    m_hi = MD.quantile_model(HI, SEED).fit(TR[F].values, TR[ycol].values)
    c_lo, c_hi = m_lo.predict(CA[F].values), m_hi.predict(CA[F].values)
    E = np.maximum(c_lo - CA[ycol].values, CA[ycol].values - c_hi)
    t_lo, t_hi = m_lo.predict(TE[F].values), m_hi.predict(TE[F].values)
    lvl_cuts = np.quantile(TR.kw, [1 / 3, 2 / 3])

    def groups(df):
        return {"G1_regime": PP.regime_rule(df, ramp_cut),
                "G2_peak_now": np.where(df.kw.values >= df.thr_adaptive.values, "peak_now", "normal"),
                "G3_level3": np.digitize(df.kw.values, lvl_cuts).astype(str)}

    gca, gte = groups(CA), groups(TE)
    y = TE[ycol].values
    peak_te = y >= TE.thr_adaptive.values          # 평가용 라벨(보정에는 쓰지 않는다)
    qlev = min(1.0, (1 - ALPHA) * (1 + 1 / len(E)))
    Qg = float(np.quantile(E, qlev))
    variants = {"global_CQR": (t_lo - Qg, t_hi + Qg)}
    for gname in gca:
        lo_v, hi_v = t_lo.copy(), t_hi.copy()
        for g in np.unique(gca[gname]):
            m = gca[gname] == g
            if m.sum() < 50:                        # 표본 부족 그룹은 전역 분위로 대체
                Qk = Qg
            else:
                Qk = float(np.quantile(E[m], min(1.0, (1 - ALPHA) * (1 + 1 / m.sum()))))
            t = gte[gname] == g
            lo_v[t], hi_v[t] = t_lo[t] - Qk, t_hi[t] + Qk
        variants[f"mondrian_{gname}"] = (lo_v, hi_v)
    cl = TE.clean.values
    for vname, (lo_v, hi_v) in variants.items():
        cov = (y >= lo_v) & (y <= hi_v)
        w = hi_v - lo_v
        reg = gte["G1_regime"]
        pc = peak_te & cl
        rows.append(dict(fold=fi, variant=vname, n=len(TE), clean_share=float(TE.clean.mean()),
                         coverage_clean=float(cov[cl].mean()) if cl.any() else np.nan,
                         coverage_peak_clean=float(cov[pc].mean()) if pc.any() else np.nan,
                         n_peak_clean=int(pc.sum()),
                         coverage=float(cov.mean()), mean_width=float(w.mean()),
                         coverage_peak=float(cov[peak_te].mean()) if peak_te.any() else np.nan,
                         width_peak=float(w[peak_te].mean()) if peak_te.any() else np.nan,
                         coverage_peak_now=float(cov[gte["G2_peak_now"] == "peak_now"].mean()),
                         coverage_transition=float(cov[reg == "TRANSITION_LIKE"].mean()),
                         coverage_low_load=float(cov[reg == "LOW_LOAD"].mean()),
                         peak_prevalence=float(peak_te.mean())))
res = pd.DataFrame(rows)
save_table(res, "24_mondrian_by_fold")
agg = res.groupby("variant").agg(
    n_folds=("fold", "size"), coverage=("coverage", "mean"), mean_width=("mean_width", "mean"),
    coverage_peak=("coverage_peak", "mean"), width_peak=("width_peak", "mean"),
    coverage_clean=("coverage_clean", "mean"), coverage_peak_clean=("coverage_peak_clean", "mean"),
    coverage_peak_now=("coverage_peak_now", "mean"),
    coverage_transition=("coverage_transition", "mean"),
    coverage_low_load=("coverage_low_load", "mean")).reset_index()
g = agg[agg.variant == "global_CQR"].iloc[0]
agg["d_coverage_peak_vs_global"] = agg.coverage_peak - g.coverage_peak
agg["d_coverage_vs_global"] = agg.coverage - g.coverage
agg["d_coverage_peak_clean_vs_global"] = agg.coverage_peak_clean - g.coverage_peak_clean
agg["rel_width_vs_global"] = agg.mean_width / g.mean_width - 1
# 사전 판정 규칙: 피크 coverage +0.05 이상 개선 & 전체 coverage 하락 0.02 이내 & 구간폭 +20% 이내
agg["adopt"] = ((agg.d_coverage_peak_vs_global >= .05) & (agg.d_coverage_vs_global >= -.02) &
                (agg.rel_width_vs_global <= .20) & (agg.variant != "global_CQR"))
save_table(agg, "24_mondrian_summary")
print(agg.round(4).to_string(index=False))
print("채택 후보:", list(agg[agg.adopt].variant) or "없음")

fig, ax = plt.subplots(figsize=(7, 3.6))
x = np.arange(len(agg))
ax.bar(x - .2, agg.coverage, .4, label="전체 coverage")
ax.bar(x + .2, agg.coverage_peak, .4, label="피크구간 coverage")
ax.axhline(.9, ls=":", color="#c33", lw=.8)
ax.set_xticks(x); ax.set_xticklabels(agg.variant, rotation=20, ha="right", fontsize=6)
ax.legend(fontsize=7); ax.set_title("조건부(Mondrian) conformal — 피크구간 coverage 개선 여부 (h60)")
fig.tight_layout(); fig.savefig(fig_path("24_mondrian_peak_coverage")); plt.close(fig)
