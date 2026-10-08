"""P0-3. 운영상 피크 초과의 **보정된 확률** + 분류 베이스라인 (심사표 '이상 확률' 대응).

정의: p_h(t) = P( y_{t+h} > thr_t | X_t ),  thr_t = 직전 30일 p95 (기존 정의 불변).
명명 가드: 이것은 **운영상 피크 초과 확률**이다. 설비 고장 확률·품질 불량 확률·
          KEPCO 계약전력 초과 확률이 아니다. 라벨은 전력 수요 초과 사건뿐이다.

프로토콜(평가와 동일): 폴드별 TRAIN(분류기 학습) / CAL(isotonic 보정 + 운영임계 선택) / TEST(적용).
TEST 라벨은 어떤 적합·선택에도 쓰지 않는다.
베이스라인: persistence 규칙 / 회귀점수 임계 / HGB 분류기 / ExtraTrees 분류기 / 사전확률.
주 모델은 RandomForest(표준 벤치마크 34 에서 F1·MCC 최고). isotonic 보정도 함께 산출해
개선 여부를 표에 남긴다 — 본 데이터에서는 RF 가 이미 충분히 보정되어 isotonic 이 Brier 를 악화시킨다.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, fig_path, mpl                              # noqa: E402
import features as FT, models as MD                                       # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,  # noqa: E402
                              ExtraTreesClassifier)
from sklearn.isotonic import IsotonicRegression                           # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                       # noqa: E402
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, brier_score_loss)  # noqa: E402
plt = mpl()

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
SEED = cfg["seed"]
H = [1, 2, 3, 4]
GRID = np.round(np.arange(.005, .995, .005), 4)   # 운영임계 후보(확률). CAL 에서만 고른다


def ece(y, p, bins=10):
    """Expected calibration error — 동일폭 구간."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def rel_curve(y, p, tag, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        out.append(dict(score=tag, bin=b, bin_lo=edges[b], bin_hi=edges[b + 1],
                        n=int(m.sum()), mean_pred=float(p[m].mean()),
                        observed_rate=float(y[m].mean())))
    return out


q, mode = FT.build(cfg["external"]["use_external_data"], cfg["peak"]["window_days"],
                   cfg["peak"]["quantile"], cfg["peak"]["min_days"])
d = FT.common_rows(q, mode)
F = FT.CORE
folds = list(TimeSeriesSplit(n_splits=cfg["data"]["n_splits"]).split(d))
rows, rel_rows, store = [], [], []

for h in H:
    ycol = f"y_h{h}"
    lab = (d[ycol].values >= d.thr_adaptive.values).astype(int)
    for fi, (itr, ite) in enumerate(folds):
        cut = int(len(itr) * (1 - cfg["data"]["cal_fraction"]))
        tr, ca, te = itr[:cut], itr[cut:], ite
        Xtr, Xca, Xte = d.iloc[tr][F].values, d.iloc[ca][F].values, d.iloc[te][F].values
        ytr, yca, yte = lab[tr], lab[ca], lab[te]
        if ytr.sum() < 20 or yca.sum() < 5:
            continue
        # --- 주 모델: RandomForest + CAL isotonic 보정
        #     선정 근거: 34_standard_benchmark_alert_clean.csv — 표준 분류모델 12종 중
        #     F1·MCC 최고(F1 0.512 / MCC 0.510). 클래스 가중은 오히려 악화되어 쓰지 않는다.
        et = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                    random_state=SEED).fit(Xtr, ytr)
        p_ca_raw, p_te_raw = et.predict_proba(Xca)[:, 1], et.predict_proba(Xte)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca_raw, yca)
        p_te_cal = iso.predict(p_te_raw)
        p_ca_cal = iso.predict(p_ca_raw)
        # --- 베이스라인
        clf = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
        p_ca_hgb, p_te_hgb = clf.predict_proba(Xca)[:, 1], clf.predict_proba(Xte)[:, 1]
        rf = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                  random_state=SEED).fit(Xtr, ytr)
        p_te_rf = rf.predict_proba(Xte)[:, 1]
        reg = MD.point_model("hgb", SEED).fit(Xtr, d.iloc[tr][ycol].values)
        s_te_reg = reg.predict(Xte) - d.iloc[te].thr_adaptive.values      # 회귀 여유 점수
        s_ca_reg = reg.predict(Xca) - d.iloc[ca].thr_adaptive.values
        s_te_pers = d.iloc[te].kw.values - d.iloc[te].thr_adaptive.values  # persistence 규칙
        s_ca_pers = d.iloc[ca].kw.values - d.iloc[ca].thr_adaptive.values
        prior = float(ytr.mean())

        cands = {
            "RF_clf(main)": (p_ca_raw, p_te_raw, True),
            "RF_clf_isotonic": (p_ca_cal, p_te_cal, True),
            "HGB_clf_uncalibrated": (p_ca_hgb, p_te_hgb, True),
            "ET_clf_uncalibrated": (rf.predict_proba(Xca)[:, 1], p_te_rf, True),
            "regression_margin_score": (s_ca_reg, s_te_reg, False),
            "persistence_rule_margin": (s_ca_pers, s_te_pers, False),
            "prior_constant": (np.full(len(yca), prior), np.full(len(yte), prior), True)}
        clean_te = d.iloc[te].clean.values
        for name, (sc_ca, sc_te, is_prob) in cands.items():
            # 운영임계는 CAL 에서 F1 최대화로 선택 (TEST 미사용)
            if is_prob:
                cand_t = GRID
            else:
                cand_t = np.quantile(sc_ca, np.round(np.arange(.50, .999, .01), 3))
            f1s = [f1_score(yca, (sc_ca >= t).astype(int), zero_division=0) for t in cand_t]
            t_op = float(cand_t[int(np.argmax(f1s))])
            pred = (sc_te >= t_op).astype(int)
            for wtag, m in [("all_2021", np.ones(len(yte), bool)), ("clean_Jul_Sep", clean_te)]:
                if m.sum() < 50 or yte[m].sum() < 3:
                    continue
                yy, pp, sc = yte[m], pred[m], sc_te[m]
                rec = dict(window=wtag, horizon_min=h * 15, fold=fi, score=name, n=int(m.sum()),
                           prevalence=float(yy.mean()), operating_threshold=t_op,
                           pr_auc=float(average_precision_score(yy, sc)),
                           roc_auc=float(roc_auc_score(yy, sc)) if 0 < yy.mean() < 1 else np.nan,
                           f1=float(f1_score(yy, pp, zero_division=0)),
                           precision=float(precision_score(yy, pp, zero_division=0)),
                           recall=float(recall_score(yy, pp, zero_division=0)),
                           tp=int(((pp == 1) & (yy == 1)).sum()), fp=int(((pp == 1) & (yy == 0)).sum()),
                           fn=int(((pp == 0) & (yy == 1)).sum()), tn=int(((pp == 0) & (yy == 0)).sum()))
                if is_prob:
                    rec.update(brier=float(brier_score_loss(yy, np.clip(sc, 0, 1))),
                               ece=ece(yy, np.clip(sc, 0, 1)))
                else:
                    rec.update(brier=np.nan, ece=np.nan)
                rows.append(rec)
            if is_prob:
                rel_rows += [dict(horizon_min=h * 15, fold=fi, **r)
                             for r in rel_curve(yte[clean_te], np.clip(sc_te[clean_te], 0, 1), name)]
        store.append(pd.DataFrame(dict(ts15=d.iloc[te].ts15.values, fold=fi, horizon_min=h * 15,
                                       clean=clean_te, label=yte, p_cal=p_te_cal,
                                       p_raw=p_te_raw, thr_adaptive=d.iloc[te].thr_adaptive.values)))

res = pd.DataFrame(rows)
save_table(res, "28_peak_probability_by_fold")
agg = (res.groupby(["window", "horizon_min", "score"])
       .agg(n_folds=("fold", "nunique"), n=("n", "sum"), prevalence=("prevalence", "mean"),
            pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"), f1=("f1", "mean"),
            precision=("precision", "mean"), recall=("recall", "mean"),
            brier=("brier", "mean"), ece=("ece", "mean"),
            tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), tn=("tn", "sum"),
            operating_threshold=("operating_threshold", "mean")).reset_index())
save_table(agg, "28_peak_probability_summary")
save_table(pd.DataFrame(rel_rows), "28_peak_probability_reliability_curve")
pr = pd.concat(store, ignore_index=True)
save_table(pr, "28_peak_probability_rows")        # 행 단위 — gitignore

cl = agg[agg.window == "clean_Jul_Sep"]
print(cl[["horizon_min", "score", "prevalence", "pr_auc", "f1", "precision", "recall",
          "brier", "ece"]].round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
rc = pd.DataFrame(rel_rows)
for name, mk in [("RF_clf(main)", "o-"), ("RF_clf_isotonic", "s--")]:
    s = rc[(rc.score == name) & (rc.horizon_min == 60)].groupby("bin").agg(
        mean_pred=("mean_pred", "mean"), observed_rate=("observed_rate", "mean")).reset_index()
    axes[0].plot(s.mean_pred, s.observed_rate, mk, label=name, ms=4)
axes[0].plot([0, 1], [0, 1], ":", color="k", lw=.8)
axes[0].set_xlabel("예측 확률"); axes[0].set_ylabel("실제 발생률")
axes[0].set_title("신뢰도 곡선 (h60, 원본구간)"); axes[0].legend(fontsize=6)
x = np.arange(len(cl.score.unique()))
for i, hm in enumerate([15, 60]):
    s = cl[cl.horizon_min == hm].set_index("score").reindex(
        ["RF_clf(main)", "RF_clf_isotonic", "HGB_clf_uncalibrated", "ET_clf_uncalibrated",
         "regression_margin_score", "persistence_rule_margin", "prior_constant"])
    axes[1].bar(x + (i - .5) * .4, s.f1.values, .4, label=f"h{hm}")
axes[1].set_xticks(x); axes[1].set_xticklabels(
    ["RF(main)", "RF+iso", "HGB", "ET", "회귀여유", "persistence", "사전확률"],
    rotation=20, ha="right", fontsize=6)
axes[1].set_ylabel("F1"); axes[1].legend(fontsize=7); axes[1].set_title("분류 베이스라인 비교 (원본구간)")
fig.suptitle("FIGURE 7. 운영상 피크 초과 확률 — 보정과 분류 성능 (고장·과금 확률 아님)", fontsize=9)
fig.tight_layout(); fig.savefig(fig_path("28_peak_probability")); plt.close(fig)
