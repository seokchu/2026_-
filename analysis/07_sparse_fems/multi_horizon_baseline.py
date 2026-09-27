"""07-A. 다중 horizon (t+15/30/45/60분) direct 예측 기준선.

Persistence / Ridge / RandomForest / HGB. 시간순 분할, train+cal 로 학습, test 로 평가.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, window, split, peak_threshold, metrics, paired_boot, save_table, fig_path, mpl, SEED
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
plt = mpl()

MODELS = {
    "persistence": None,
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                  n_jobs=-1, random_state=SEED),
    "hgb": lambda: HistGradientBoostingRegressor(random_state=SEED),
}
F = _fe.LEVELS["L3_+production"]
rows, err_store = [], {}
for wtag in ["all_2021", "clean_Jul_Sep"]:
    sub = window(build(), wtag)
    for h in _fe.HORIZONS:
        tr, ca, te = split(sub, F, h)
        fit = pd.concat([tr, ca])
        y = te[f"y_h{h}"].values
        thr = peak_threshold(fit)
        rq = float(np.quantile(np.abs(fit.kw_ramp4), .9))
        for mname, mk in MODELS.items():
            p = te.kw.values if mk is None else mk().fit(fit[F].values, fit[f"y_h{h}"].values).predict(te[F].values)
            m = metrics(te, y, p, thr, rq)
            rows.append(dict(window=wtag, horizon_min=h * 15, model=mname, n_train=len(fit),
                             peak_threshold_kw=thr, **m))
            err_store[(wtag, h, mname)] = np.abs(y - p)
res = pd.DataFrame(rows)
base = res[res.model == "persistence"].set_index(["window", "horizon_min"]).mae
res["mae_vs_persistence_pct"] = [(r.mae - base[(r.window, r.horizon_min)]) / base[(r.window, r.horizon_min)]
                                 for r in res.itertuples()]
save_table(res, "07A_multihorizon_baseline")

boot = []
for (wtag, h, mname), e in err_store.items():
    if mname == "persistence":
        continue
    boot.append(dict(window=wtag, horizon_min=h * 15, model=mname,
                     vs="persistence", **paired_boot(e, err_store[(wtag, h, "persistence")])))
    if mname != "hgb":
        boot.append(dict(window=wtag, horizon_min=h * 15, model="hgb", vs=mname,
                         **paired_boot(err_store[(wtag, h, "hgb")], e)))
save_table(pd.DataFrame(boot), "07A_multihorizon_bootstrap")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, wtag in zip(axes, ["all_2021", "clean_Jul_Sep"]):
    for mname in MODELS:
        d = res[(res.window == wtag) & (res.model == mname)]
        ax.plot(d.horizon_min, d.mae, "o-", label=mname)
    ax.set_xlabel("예측 horizon (분)"); ax.set_ylabel("MAE (kw)")
    ax.set_title(f"{wtag} — horizon 별 MAE"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(fig_path("07A_multihorizon_mae")); plt.close(fig)
print(res[["window", "horizon_min", "model", "mae", "mae_vs_persistence_pct", "mae_operating",
           "mae_idle", "mae_peak", "mae_high_ramp"]].to_string(index=False))
