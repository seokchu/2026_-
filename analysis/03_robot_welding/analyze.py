"""03. Robot welding: schema, label granularity, injected-block forensics, derived physics."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from common import DATA, save_table, fig_path, mpl
plt = mpl()

r = pd.read_excel(DATA["weld_xlsx"], sheet_name="Raw data")
r.columns = ["idx", "machine", "item", "date", "th1", "th2", "force", "current", "voltage", "wtime"]
r = r.sort_values(["date", "idx"]).reset_index(drop=True)
res = pd.read_excel(DATA["weld_xlsx"], sheet_name="result")
P = ["force", "current", "voltage", "wtime"]

# ---- 1. label granularity
lab = dict(raw_rows=len(r), raw_days=int(r.date.nunique()), machines=int(r.machine.nunique()),
           items=int(r["item"].nunique()), thickness_combos=int(r[["th1", "th2"]].drop_duplicates().shape[0]),
           result_rows=len(res), result_days=int(res["working time"].nunique()),
           defect_types=int(res["defect type"].nunique()),
           total_defects=int(res.defect.sum()),
           row_level_labels_available=False,
           raw_days_without_label=";".join(sorted(set(r.date.dt.strftime("%Y-%m-%d")) - set(res["working time"].dt.strftime("%Y-%m-%d")))),
           unique_process_combos=int(r[P].drop_duplicates().shape[0]),
           duplicate_row_fraction=float(r[P].duplicated().mean()))
save_table(pd.DataFrame([lab]), "03_weld_label_granularity")

# ---- 2. per-day process stats + injected block detection
g = r.groupby("date").agg(n=("idx", "size"), **{f"{c}_{s}": (c, s) for c in P for s in ("mean", "std", "min", "max")})
g["n_force_gt5"] = r.groupby("date").force.apply(lambda s: int((s > 5).sum()))
dl = res.groupby("working time").defect.sum().rename("defects_total")
dt = res.pivot_table(index="working time", columns="defect type", values="defect", aggfunc="sum")
dt.columns = [f"defect_type_{c}" for c in dt.columns]
day = g.join(dl).join(dt)
save_table(day.reset_index(), "03_weld_daily_process_vs_defect")

# byte-identical injected block check
B = [d for d in r.date.unique() if (r[r.date == d].force > 5).sum() > 0]
blocks = {str(pd.Timestamp(d).date()): r[(r.date == d) & (r.force > 5)][P].reset_index(drop=True) for d in B}
ks = list(blocks)
chk = [dict(day_a=ks[0], day_b=b, n_rows=len(blocks[b]), byte_identical=bool(blocks[ks[0]].equals(blocks[b])))
       for b in ks[1:]]
save_table(pd.DataFrame(chk), "03_weld_injected_block_identity")

# ---- 3. two-population separation (which regime a day belongs to)
r["regime"] = np.where(r.date.isin(B), "B_with_injected_block", "A_clean")
r["is_injected"] = r.force > 5
pop = r.groupby(["regime", "is_injected"])[P].agg(["mean", "std", "min", "max", "size"])
save_table(pop.reset_index(), "03_weld_regime_population_stats", index=False)

# ---- 4. derived physics: does V*I, I^2*t add information beyond the raw four?
r["VI"] = r.voltage * r.current
r["VIt"] = r.VI * r.wtime
r["I2t"] = r.current ** 2 * r.wtime
r["force_per_mm"] = r.force / (r.th1 + r.th2)
der = r[P + ["VI", "VIt", "I2t", "force_per_mm"]].corr(method="spearman")
save_table(der.reset_index(), "03_weld_derived_spearman", index=False)
# R^2 of each derived var on the raw four (redundancy test)
from sklearn.linear_model import LinearRegression
red = []
for c in ["VI", "VIt", "I2t", "force_per_mm"]:
    lr = LinearRegression().fit(r[P], r[c])
    red.append(dict(derived=c, r2_from_raw_linear=lr.score(r[P], r[c]),
                    note="1.0 = 원변수 선형결합으로 완전 복원 → 정보 추가 없음"))
save_table(pd.DataFrame(red), "03_weld_derived_redundancy")

# ---- 5. relationship stability: rolling corr(V, I) within day
roll = []
for d, grp in r.sort_values(["date", "idx"]).groupby("date"):
    c = grp.voltage.rolling(100).corr(grp.current)
    roll.append(dict(date=str(pd.Timestamp(d).date()), regime=grp.regime.iloc[0], n=len(grp),
                     roll_corr_VI_mean=float(c.mean()), roll_corr_VI_std=float(c.std()),
                     roll_corr_VI_min=float(c.min()), roll_corr_VI_max=float(c.max()),
                     spearman_idx_force=float(stats.spearmanr(grp.idx, grp.force).statistic),
                     spearman_idx_current=float(stats.spearmanr(grp.idx, grp.current).statistic)))
save_table(pd.DataFrame(roll), "03_weld_relationship_stability")

# ---- 6. daily defect count vs daily process stats (n=8) with p-values
sub = day.dropna(subset=["defects_total"])
corr = []
for c in [c for c in g.columns if sub[c].nunique() > 1]:
    pr, pp = stats.pearsonr(sub[c], sub.defects_total)
    sr, sp = stats.spearmanr(sub[c], sub.defects_total)
    corr.append(dict(feature=c, n_days=len(sub), pearson=pr, pearson_p=pp, spearman=sr, spearman_p=sp,
                     significant_at_0p05=bool(min(pp, sp) < .05)))
save_table(pd.DataFrame(corr).sort_values("spearman", key=abs, ascending=False), "03_weld_daily_defect_corr")

# ---- figures
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].hist(r.force, bins=120, color="steelblue"); ax[0].set_yscale("log")
ax[0].set_title("weld force(bar): 주 분포 2.3 부근 + 5~10.5 별도 덩어리")
for d, grp in r.groupby("date"):
    ax[1].plot(grp.idx.values, grp.force.values, lw=.4, label=str(pd.Timestamp(d).date()))
ax[1].set_title("일자별 force vs 생산순번 (동일 243행 블록이 4일에 삽입)")
ax[1].legend(fontsize=5, ncol=2)
ax[2].scatter(r.current, r.voltage, s=2, c=(r.force > 5).map({True: "crimson", False: "steelblue"}), alpha=.3)
ax[2].set_xlabel("current(kA)"); ax[2].set_ylabel("voltage(v)")
ax[2].set_title("전류-전압 관계: 삽입블록(빨강)은 별도 궤적")
fig.tight_layout(); fig.savefig(fig_path("03_weld_overview")); plt.close(fig)
print("done")
