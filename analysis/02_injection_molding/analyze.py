"""02. Injection molding: structure, label forensics, regimes, drift, baseline, error map."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, average_precision_score
from common import load_mold, save_table, fig_path, mpl, SEED

plt = mpl()
OUT = {}

# ---------------------------------------------------------------- 1. pair structure
pair_rows = []
labeled = {}
for k in ["mold_lab_cn7", "mold_lab_rg3"]:
    d = load_mold(k)
    F = [c for c in d.columns if c not in ("idx", "PassOrFail")]
    d["pair"] = d.groupby(F, sort=False).ngroup()
    labeled[k] = (d, F)
    gl = d.groupby("pair").PassOrFail.agg(["size", "sum", "nunique"])
    conf = gl[gl["nunique"] > 1]
    consec = d.groupby("pair")["idx"].agg(lambda s: bool((np.diff(np.sort(s)) == 1).all()) if len(s) > 1 else True)
    pair_rows.append(dict(dataset=k, rows=len(d), pairs=int(d.pair.nunique()),
                          pair_size_2_frac=float((gl["size"] == 2).mean()),
                          pairs_consecutive_idx_frac=float(consec.mean()),
                          n_fail_rows=int(d.PassOrFail.sum()),
                          pairs_with_conflicting_labels=int(len(conf)),
                          fail_rows_inside_conflicting_pairs=int(conf["sum"].sum()),
                          pure_fail_pairs=int(((gl["sum"] == gl["size"]) & (gl["sum"] > 0)).sum()),
                          irreducible_error_frac_of_fails=float(conf["sum"].sum() / max(d.PassOrFail.sum(), 1))))
save_table(pd.DataFrame(pair_rows), "02_mold_duplicate_pair_forensics")

# ---------------------------------------------------------------- 2. fail position / burst
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
burst = []
for i, k in enumerate(["mold_lab_cn7", "mold_lab_rg3"]):
    d, F = labeled[k]
    y = d.PassOrFail.values
    ax[i].vlines(d.idx[y == 1], 0, 1, color="crimson", lw=1)
    ax[i].set_title(f"{k}: 불량 발생 위치 (idx=shot order), n_fail={y.sum()}")
    ax[i].set_yticks([])
    q = [float(y[j * len(y) // 4:(j + 1) * len(y) // 4].mean()) for j in range(4)]
    pos = d.loc[y == 1, "idx"].values
    burst.append(dict(dataset=k, n_fail=int(y.sum()), fail_rate=float(y.mean()),
                      q1=q[0], q2=q[1], q3=q[2], q4=q[3],
                      fail_idx_min=int(pos.min()), fail_idx_max=int(pos.max()),
                      fail_span_frac=float((pos.max() - pos.min()) / (d.idx.max() - d.idx.min())),
                      median_gap_between_fails=float(np.median(np.diff(pos))),
                      max_consecutive_fail_run=int(max((len(list(g)) for _, g in __import__("itertools").groupby(y) if _ == 1), default=0)))) 
fig.tight_layout(); fig.savefig(fig_path("02_mold_fail_positions")); plt.close(fig)
save_table(pd.DataFrame(burst), "02_mold_fail_burst_structure")

# ---------------------------------------------------------------- 3. univariate effect size, pair-deduplicated
eff = []
for k in ["mold_lab_cn7", "mold_lab_rg3"]:
    d, F = labeled[k]
    dd = d.groupby("pair", as_index=False).first()          # one row per physical shot
    y = dd.PassOrFail.values
    for c in F:
        if dd[c].nunique() < 2:
            eff.append(dict(dataset=k, var=c, auc=np.nan, cliff=np.nan, p=np.nan, note="constant")); continue
        a, b = dd.loc[y == 1, c], dd.loc[y == 0, c]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))
        # partial: does the effect survive controlling for shot position?
        r_pos = stats.spearmanr(dd[c], dd.idx).statistic
        eff.append(dict(dataset=k, var=c, auc=auc, cliff=2 * auc - 1, p=p,
                        spearman_with_shot_order=r_pos, note=""))
eff = pd.DataFrame(eff)
save_table(eff.sort_values(["dataset", "auc"], key=lambda s: (s - .5).abs() if s.name == "auc" else s,
                          ascending=[True, False]), "02_mold_univariate_effect")

# ---------------------------------------------------------------- 4. confound test: position vs variable
conf_rows = []
d, F = labeled["mold_lab_cn7"]
dd = d.groupby("pair", as_index=False).first()
y = dd.PassOrFail.values
early = dd.idx < dd.loc[y == 1, "idx"].max()      # region containing the fail burst
for c in ["Mold_Temperature_3", "Mold_Temperature_4", "Plasticizing_Position", "Max_Back_Pressure"]:
    full = roc_auc_score(y, dd[c])
    sub = dd[early]
    ys = sub.PassOrFail.values
    within = roc_auc_score(ys, sub[c]) if ys.sum() and (ys == 0).sum() else np.nan
    conf_rows.append(dict(var=c, auc_full=full, auc_within_burst_region=within,
                          n_within=int(len(sub)), n_fail_within=int(ys.sum())))
save_table(pd.DataFrame(conf_rows), "02_mold_temporal_confound_test")

# ---------------------------------------------------------------- 5. unlabeled regime structure
reg = []
for k in ["mold_unlab_cn7", "mold_unlab_rg3"]:
    d = load_mold(k)
    F = [c for c in d.columns if c != "idx"]
    kurt = d[F].kurt()
    key = kurt.idxmin()
    cl = (d[key] > d[key].median()).astype(int)
    sgn = (d[F] > d[F].median()).astype(int).corrwith(cl).abs()
    blocks = (cl.diff().fillna(0) != 0).cumsum().nunique()
    reg.append(dict(dataset=k, n=len(d), split_var=key, min_kurtosis=float(kurt.min()),
                    vars_with_kurt_below_neg1=int((kurt < -1).sum()),
                    cluster0=int((cl == 0).sum()), cluster1=int((cl == 1).sum()),
                    vars_abs_sign_corr_gt_0p95=int((sgn > .95).sum()), n_vars=len(F),
                    interleaved_blocks_in_row_order=int(blocks),
                    idx_min=int(d.idx.min()), idx_max=int(d.idx.max()),
                    max_idx_gap=int(np.diff(d.idx.values).max()),
                    median_unique_values_per_var=float(d[F].nunique().median())))
save_table(pd.DataFrame(reg), "02_mold_unlabeled_regime_structure")

fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
for r, k in enumerate(["mold_unlab_cn7", "mold_unlab_rg3"]):
    d = load_mold(k)
    for c, ax in zip(["Cushion_Position", "Mold_Temperature_3", "Barrel_Temperature_5"], axes[r]):
        ax.hist(d[c], bins=80, color="steelblue")
        ax.set_title(f"{k.split('_')[-1]} {c}", fontsize=8)
fig.suptitle("unlabeled 사출 데이터: 거의 모든 변수가 2점 분포(두 운전 레짐) — 라벨 데이터엔 없는 구조")
fig.tight_layout(); fig.savefig(fig_path("02_mold_unlabeled_bimodality")); plt.close(fig)

# labeled vs unlabeled distribution distance (KS on z-scores)
ks = []
for prod in ["cn7", "rg3"]:
    l = load_mold(f"mold_lab_{prod}"); u = load_mold(f"mold_unlab_{prod}")
    for c in [c for c in u.columns if c != "idx"]:
        s, p = stats.ks_2samp(l[c], u[c])
        ks.append(dict(product=prod, var=c, ks_stat=s, p=p))
save_table(pd.DataFrame(ks).sort_values("ks_stat", ascending=False), "02_mold_labeled_vs_unlabeled_ks")

# ---------------------------------------------------------------- 6. drift along shot order
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
for r, k in enumerate(["mold_lab_cn7", "mold_lab_rg3"]):
    d, F = labeled[k]
    dd = d.groupby("pair", as_index=False).first().sort_values("idx")
    for c, ax in zip(["Mold_Temperature_3", "Max_Back_Pressure"], axes[r]):
        ax.plot(dd.idx, dd[c], lw=.4, color="gray")
        ax.plot(dd.idx, dd[c].rolling(51, center=True).mean(), color="navy", lw=1.5)
        f = dd[dd.PassOrFail == 1]
        ax.scatter(f.idx, f[c], color="crimson", s=12, zorder=5)
        ax.set_title(f"{k} {c} (빨강=불량)", fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("02_mold_drift_vs_shot_order")); plt.close(fig)

# ---------------------------------------------------------------- 7. baseline models, pair-grouped CV
def cv_eval(X, y, groups, model, n_splits=5):
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups):
        m = model().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof

base = []
oof_store = {}
for k in ["mold_lab_cn7", "mold_lab_rg3"]:
    d, F = labeled[k]
    use = [c for c in F if d[c].nunique() > 1]
    X, y, g = d[use].values, d.PassOrFail.values, d.pair.values
    for nm, mk in [("logreg", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
                   ("tree_d3", lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED)),
                   ("rf", lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                         class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))]:
        oof = cv_eval(X, y, g, mk)
        base.append(dict(dataset=k, model=nm, cv="StratifiedGroupKFold(pair)",
                         auc=roc_auc_score(y, oof), ap=average_precision_score(y, oof),
                         prevalence=float(y.mean())))
        oof_store[(k, nm)] = oof
    # leaky comparison: ignore the pair grouping
    from sklearn.model_selection import StratifiedKFold
    oofl = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        oofl[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
                                          random_state=SEED, n_jobs=-1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    base.append(dict(dataset=k, model="rf_LEAKY_random_split", cv="StratifiedKFold (pair leak)",
                     auc=roc_auc_score(y, oofl), ap=average_precision_score(y, oofl), prevalence=float(y.mean())))
    # temporal split
    cut = int(len(d) * .7)
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
                               random_state=SEED, n_jobs=-1).fit(X[:cut], y[:cut])
    pt = m.predict_proba(X[cut:])[:, 1]
    base.append(dict(dataset=k, model="rf_temporal_70_30", cv="time-ordered holdout",
                     auc=roc_auc_score(y[cut:], pt) if len(set(y[cut:])) > 1 else np.nan,
                     ap=average_precision_score(y[cut:], pt) if len(set(y[cut:])) > 1 else np.nan,
                     prevalence=float(y[cut:].mean())))
save_table(pd.DataFrame(base), "02_mold_baseline_predictability")

# ---------------------------------------------------------------- 8. error forensics
err = []
for k in ["mold_lab_cn7", "mold_lab_rg3"]:
    d, F = labeled[k]
    oof = oof_store[(k, "rf")]
    y = d.PassOrFail.values
    thr = np.quantile(oof, 1 - y.mean())
    pred = (oof >= thr).astype(int)
    d2 = d.assign(oof=oof, pred=pred,
                  cell=np.where(y & pred, "TP", np.where(y & ~pred.astype(bool), "FN",
                        np.where(~y.astype(bool) & pred.astype(bool), "FP", "TN"))))
    save_table(d2[["idx", "pair", "PassOrFail", "oof", "pred", "cell"]], f"02_mold_oof_{k[-3:]}")
    grp = d2.groupby("cell").agg(n=("idx", "size"), median_idx=("idx", "median"),
                                 median_score=("oof", "median"))
    grp["dataset"] = k
    err.append(grp.reset_index())
    # is the FN set inside conflicting pairs?
    gl = d.groupby("pair").PassOrFail.nunique()
    d2["in_conflicting_pair"] = d2.pair.map(gl) > 1
    print(k, "FN in conflicting pair:", d2.query("cell=='FN'").in_conflicting_pair.mean(),
          "TP in conflicting pair:", d2.query("cell=='TP'").in_conflicting_pair.mean())
save_table(pd.concat(err), "02_mold_error_cells")

# ---------------------------------------------------------------- 9. rule mining (failure pockets)
pockets = []
for k in ["mold_lab_cn7", "mold_lab_rg3"]:
    d, F = labeled[k]
    dd = d.groupby("pair", as_index=False).first()
    use = [c for c in F if dd[c].nunique() > 1]
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, class_weight="balanced",
                              random_state=SEED).fit(dd[use], dd.PassOrFail)
    leaf = t.apply(dd[use])
    for lv in np.unique(leaf):
        m = leaf == lv
        pockets.append(dict(dataset=k, leaf=int(lv), n=int(m.sum()), share_of_rows=float(m.mean()),
                            fail_rate=float(dd.PassOrFail[m].mean()),
                            share_of_all_fails=float(dd.PassOrFail[m].sum() / dd.PassOrFail.sum()),
                            lift=float(dd.PassOrFail[m].mean() / dd.PassOrFail.mean())))
    (open(f"02_injection_molding/tree_{k[-3:]}.txt", "w")
     .write(export_text(t, feature_names=use, decimals=3)))
save_table(pd.DataFrame(pockets).sort_values(["dataset", "lift"], ascending=[True, False]), "02_mold_failure_pockets")
print("done")

# ---------------------------------------------------------------- 10. pairwise interaction search
import itertools
from sklearn.metrics import roc_auc_score as _auc
inter = []
for k in ["mold_lab_cn7", "mold_lab_rg3"]:
    d, F = labeled[k]
    dd = d.groupby("pair", as_index=False).first()
    use = [c for c in F if dd[c].nunique() > 2]
    y = dd.PassOrFail.values
    if y.sum() < 2: continue
    solo = {c: abs(_auc(y, dd[c]) - .5) for c in use}
    for a, b in itertools.combinations(use, 2):
        for tag, v in [("prod", dd[a] * dd[b]), ("diff", dd[a] - dd[b]),
                       ("ratio", dd[a] / dd[b].replace(0, np.nan))]:
            v = v.replace([np.inf, -np.inf], np.nan).fillna(0)
            if v.nunique() < 3: continue
            g = abs(_auc(y, v) - .5)
            inter.append(dict(dataset=k, a=a, b=b, kind=tag, auc_gain_over_best_solo=g - max(solo[a], solo[b]),
                              auc_interaction=.5 + g, auc_solo_a=.5 + solo[a], auc_solo_b=.5 + solo[b]))
inter = pd.DataFrame(inter)
# permutation null for the max gain
perm = []
for k in inter.dataset.unique():
    d, F = labeled[k]
    dd = d.groupby("pair", as_index=False).first()
    use = [c for c in F if dd[c].nunique() > 2]
    y = dd.PassOrFail.values
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(200):
        ys = rng.permutation(y)
        a, b = rng.choice(use, 2, replace=False)
        v = dd[a] * dd[b]
        null.append(abs(_auc(ys, v) - .5))
    obs = inter.query("dataset==@k").auc_gain_over_best_solo.max()
    perm.append(dict(dataset=k, max_observed_gain=float(obs), null_p95=float(np.quantile(null, .95)),
                     null_max=float(np.max(null)), n_perm=200,
                     exceeds_null_p95=bool(obs > np.quantile(null, .95))))
save_table(pd.DataFrame(perm), "02_mold_interaction_permutation_test")
save_table(inter.sort_values(["dataset", "auc_gain_over_best_solo"], ascending=[True, False])
           .groupby("dataset").head(15), "02_mold_top_interactions")
print("interaction search done", len(inter))
