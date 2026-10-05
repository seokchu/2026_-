"""후속 T. '설비 가동상태 채널을 1~2개 추가하면 정보부족 플래그가 줄어드는가' — 대리(proxy) 측정.

KAMP 데이터에는 설비 단위 채널이 없다. 따라서 **대리 실험**으로만 답한다:
  A_no_operating  : 집계전력 + 달력 + 기상관측 (조업/생산 채널 없음)
  B_plus_operating: A + is_operating (조업 여부 1채널)
  C_plus_production: B + 생산량·prod_lag4·prod_diff (생산 정보까지)
해석 한계: 생산량/조업여부는 ERP 집계이며 설비 단위 센서가 아니다. 따라서 결과는
'조업·생산 문맥 채널 1~2개의 가치' 로만 읽는다. 특정 센서를 지목하지 않는다.

측정: h60 MAE / LOW+OOD 비중 / coverage(전체·피크) / information_gap_flag 비중.
정보부족 플래그 정의(보고서 23 §K 와 동일 축):
  불확실(LOW/OOD) 이고, 그 원인을 현재 보유 변수로 지목할 수 없는 경우.
  설명가능 조건은 각 정보수준이 실제로 가진 변수만으로 구성한다(없는 변수는 쓰지 않는다).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, fig_path, mpl                  # noqa: E402
import features as FT, models as MD, _fe                      # noqa: E402
from sklearn.model_selection import TimeSeriesSplit           # noqa: E402
plt = mpl()

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
SEED, rc = cfg["seed"], cfg["reliability"]
LO, HI = rc["quantile_levels"]
H = 4
BASE = _fe.L0 + _fe.CAL + _fe.WEA
LEVELS = {"A_no_operating": BASE,
          "B_plus_operating": BASE + ["is_operating"],
          "C_plus_production": BASE + ["is_operating", "생산량", "prod_lag4", "prod_diff"]}

q, mode = FT.build(cfg["external"]["use_external_data"], cfg["peak"]["window_days"],
                   cfg["peak"]["quantile"], cfg["peak"]["min_days"])
d = FT.common_rows(q, mode)          # 공통 행집합 유지 — 수준 간 동일 행 비교
folds = list(TimeSeriesSplit(n_splits=cfg["data"]["n_splits"]).split(d))
rows, store = [], {}
for lname, F in LEVELS.items():
    rec = []
    for fi, (itr, ite) in enumerate(folds):
        cut = int(len(itr) * (1 - cfg["data"]["cal_fraction"]))
        tr, ca = itr[:cut], itr[cut:]
        TR, CA, TE = d.iloc[tr], d.iloc[ca], d.iloc[ite]
        ycol = f"y_h{H}"
        pt = MD.point_model(cfg["forecast"]["point_model"], SEED).fit(
            d.iloc[np.r_[tr, ca]][F].values, d.iloc[np.r_[tr, ca]][ycol].values)
        lo, hi, cal_w, _ = MD.fit_cqr(TR[F].values, TR[ycol].values, CA[F].values,
                                      CA[ycol].values, TE[F].values, LO, HI,
                                      rc["conformal_alpha"], SEED)
        cv, ood_cut = MD.ood_scorer(TR[F].values, rc["ood_quantile"])
        maha = cv.mahalanobis(TE[F].values)
        band = MD.bands(hi - lo, cal_w, rc["band_width_quantiles"], maha, ood_cut)
        ramp_cut = float(np.quantile(TR.kw_ramp4.abs(), cfg["regime"]["ramp_quantile"]))
        t = TE[["ts15", "clean", "kw", "thr_adaptive", "kw_ramp4", "is_operating",
                "is_operating_lag4", ycol]].copy()
        t["pred"], t["lo"], t["hi"], t["band"] = pt.predict(TE[F].values), lo, hi, band
        t["abs_err"] = (t[ycol] - t.pred).abs()
        t["covered"] = ((t[ycol] >= t.lo) & (t[ycol] <= t.hi)).astype(int)
        t["is_peak"] = (t[ycol] >= t.thr_adaptive).astype(int)
        t["unsure"] = t.band.isin(cfg["diagnosis"]["uncertain_bands"])
        # 설명가능 조건: 해당 정보수준이 실제로 가진 변수만 사용
        expl = (t.kw_ramp4.abs() >= ramp_cut).values
        if "is_operating" in F:
            sw = t.is_operating.values != t.is_operating_lag4.values
            expl = expl | sw | (t.is_operating.values == 0)
        t["explained"] = expl
        t["info_gap"] = (t.unsure & ~t.explained).astype(int)
        rec.append(t)
    o = pd.concat(rec, ignore_index=True)
    store[lname] = o
    for wtag, s in [("all_2021", o), ("clean_Jul_Sep", o[o.clean == True])]:
        pk = s[s.is_peak == 1]
        rows.append(dict(window=wtag, level=lname, n=len(s), horizon_min=60,
                         mae=float(s.abs_err.mean()),
                         low_or_ood_share=float(s.unsure.mean()),
                         coverage=float(s.covered.mean()),
                         coverage_peak_only=float(pk.covered.mean()) if len(pk) else np.nan,
                         mean_interval_width=float((s.hi - s.lo).mean()),
                         information_gap_rate=float(s.info_gap.mean()),
                         explained_share=float(s.explained.mean()),
                         gap_mae=float(s.abs_err[s.info_gap == 1].mean()) if s.info_gap.any() else np.nan))
res = pd.DataFrame(rows)
b = res[res.level == "A_no_operating"].set_index("window")
for c in ["mae", "low_or_ood_share", "information_gap_rate", "coverage_peak_only"]:
    res[f"d_{c}_vs_A"] = [r[c] - b.loc[r.window, c] for _, r in res.iterrows()]
res["rel_gap_reduction_vs_A"] = [
    (b.loc[r.window, "information_gap_rate"] - r.information_gap_rate) /
    b.loc[r.window, "information_gap_rate"] for _, r in res.iterrows()]
save_table(res, "24_added_channel_value")
# 동일 행 대응 bootstrap (MAE 차)
boot = []
for lname in ["B_plus_operating", "C_plus_production"]:
    for wtag in ["all_2021", "clean_Jul_Sep"]:
        a = store["A_no_operating"]; x = store[lname]
        m = (a.clean == True).values if wtag == "clean_Jul_Sep" else np.ones(len(a), bool)
        boot.append(dict(window=wtag, level=lname, vs="A_no_operating",
                         **_fe.paired_boot(x.abs_err.values[m], a.abs_err.values[m])))
save_table(pd.DataFrame(boot), "24_added_channel_bootstrap")
print(res[res.window == "clean_Jul_Sep"].round(4).to_string(index=False))
print(pd.DataFrame(boot).round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
c = res[res.window == "clean_Jul_Sep"]
axes[0].bar(c.level, c.information_gap_rate, color="#27b")
axes[0].set_ylabel("정보부족 플래그 비중"); axes[0].set_title("조업·생산 채널 추가 효과")
axes[1].bar(c.level, c.mae, color="#c33"); axes[1].set_ylabel("h60 MAE (kw)")
axes[1].set_title("평균 오차")
for ax in axes:
    ax.set_xticklabels(c.level, rotation=15, ha="right", fontsize=6)
fig.tight_layout(); fig.savefig(fig_path("24_added_channel_value")); plt.close(fig)
