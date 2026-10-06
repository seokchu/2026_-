"""30-C. 필수 시각화 — 운전상태 시간흐름 / 2차원 투영 / 상태별 FN·FP(분모 포함) / 전구간 시계열.

주의 표기(그림 캡션에 포함):
  - 2차원 투영(PCA)은 **시각화용**이다. 그림에서 세 무리가 떨어져 보이는 것이 개별 모델의 성능 향상을
    뜻하지 않는다.
  - 규칙으로 붙인 운전상태(rule)와 k=3 을 미리 정한 군집(KMeans3)은 **다른 것**이다. 패널을 분리한다.
  - FN/FP 는 **경고 성능**, MAE 는 **수치 예측 성능**이다. 섞어 읽지 않는다.
  - 모든 패널에 평가 기간을 명시한다. 학습 구간은 포함하지 않는다.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, fig_path, mpl, TAB                        # noqa: E402
import features as FT                                                    # noqa: E402
from sklearn.decomposition import PCA                                    # noqa: E402
from sklearn.cluster import KMeans                                       # noqa: E402
from sklearn.preprocessing import StandardScaler                         # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                      # noqa: E402
plt = mpl()

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
PRE = cfg["output"]["table_prefix"]
SEED = cfg["seed"]
STAGES = {"early_h60": (4, 60), "confirmA_h45": (3, 45), "confirmB_h15": (1, 15)}
LOCK_DAYS = TEST_DAYS = 14
COL = {"LOW_LOAD": "#8c8c8c", "STABLE_OPERATION": "#2b7bba", "TRANSITION_LIKE": "#d6453d"}

o = pd.read_csv(TAB / f"{PRE}_oof_rows.csv", parse_dates=["ts15"], encoding="utf-8-sig").sort_values("ts15")
thr = pd.read_csv(TAB / f"{PRE}_cal_alert_thresholds.csv", encoding="utf-8-sig")
TM = {(int(r.fold), int(r.horizon_min), float(r.q)): float(r.threshold_kw) for r in thr.itertuples()}
pol = pd.read_csv(TAB / "28_sequential_operating_points.csv", encoding="utf-8-sig")
r0 = cfg["policy"]["default_cost_ratio"]
P = pol[(pol.cost_ratio_FN_FP == r0) &
        (pol.scope == f"lead_constrained_ge{cfg['policy']['minimum_lead_minutes']}min") &
        (pol.applied_to == "TEST")]
P = P.iloc[0] if len(P) else pol[(pol.cost_ratio_FN_FP == r0) & (pol.scope == "all_policies") &
                                (pol.applied_to == "TEST")].iloc[0]
print("그림에 사용할 고정 정책:", P.policy, P.early_q, P.confirm_stage, P.confirm_q, P.gate)

# ---- 목표시각 테이블 (sequential_policy.py 와 동일 규칙)
base = o[["ts15", "fold", "clean", "kw", "thr_adaptive", "peak_now", "regime", "pred_h4",
          "band_h4"]].rename(columns={"ts15": "T", "peak_now": "label"})
for name, (hh, back) in STAGES.items():
    dec = o[["ts15", "fold", f"pred_h{hh}"]].copy()
    dec["T"] = dec.ts15 + pd.Timedelta(minutes=back)
    dec = dec.rename(columns={"ts15": f"t_{name}", "fold": f"fold_{name}",
                              f"pred_h{hh}": f"score_{name}"})
    base = base.merge(dec, on="T", how="left")
tt = base.dropna(subset=["score_early_h60", "score_confirmA_h45", "score_confirmB_h15"]).reset_index(drop=True)
cl = tt[tt.clean].reset_index(drop=True)
end = cl["T"].max(); lock_start = end - pd.Timedelta(days=LOCK_DAYS)
test_start = lock_start - pd.Timedelta(days=TEST_DAYS)
EVAL = cl[cl["T"] >= test_start].reset_index(drop=True)          # TEST + LOCK = 평가 전용 30일
EVAL["split"] = np.where(EVAL["T"] < lock_start, "TEST", "LOCK")


def mask_stage(sub, stage, qv):
    hh, _ = STAGES[stage]
    t = np.array([TM[(int(f), hh * 15, float(qv))] for f in sub[f"fold_{stage}"].values])
    return sub[f"score_{stage}"].values >= t


early = mask_stage(EVAL, "early_h60", P.early_q)
cstage = str(P.confirm_stage) if isinstance(P.confirm_stage, str) and P.confirm_stage else ""
conf = mask_stage(EVAL, cstage, P.confirm_q) if cstage else np.ones(len(EVAL), bool)
unsure = EVAL.band_h4.isin(cfg["diagnosis"]["uncertain_bands"]).values
alert = early & (conf | ~unsure) if P.gate == "confirm_only_if_unsure" else early & conf
lab = EVAL.label.values == 1
EVAL["alert"] = alert; EVAL["FN"] = (~alert) & lab; EVAL["FP"] = alert & ~lab; EVAL["TP"] = alert & lab

# ---- 상태별 FN/FP (분모 동시 표기)
rows = []
for wtag in ["TEST", "LOCK", "TEST+LOCK"]:
    s = EVAL if wtag == "TEST+LOCK" else EVAL[EVAL.split == wtag]
    for st in ["ALL"] + list(COL):
        m = np.ones(len(s), bool) if st == "ALL" else (s.regime.values == st)
        y = (s.label.values == 1)[m]; a = s.alert.values[m]
        rows.append(dict(window=wtag, state=st, n_steps=int(m.sum()),
                         n_peak_steps=int(y.sum()), n_nonpeak_steps=int((~y).sum()),
                         TP=int((a & y).sum()), FP=int((a & ~y).sum()),
                         FN=int((~a & y).sum()), TN=int((~a & ~y).sum()),
                         fn_rate_given_peak=float((~a & y).sum() / max(y.sum(), 1)),
                         fp_rate_given_nonpeak=float((a & ~y).sum() / max((~y).sum(), 1)),
                         note="경고 성능 지표 — 수치 예측 오차(MAE)와 다른 축"))
fnfp = pd.DataFrame(rows)
save_table(fnfp, "30_fnfp_by_state")
print(fnfp.round(4).to_string(index=False))

# ---- FIGURE 10: 시간 흐름 위 운전상태
fig, axes = plt.subplots(2, 1, figsize=(12, 5.4))
seg = EVAL[(EVAL["T"] >= test_start) & (EVAL["T"] < test_start + pd.Timedelta(days=5))]
ax = axes[0]
ax.plot(seg["T"], seg.kw, color="#222", lw=.9, label="실측 수요")
for st, c in COL.items():
    m = seg.regime.values == st
    ax.scatter(seg["T"][m], seg.kw[m], s=7, color=c, label=st, zorder=3)
ax.plot(seg["T"], seg.thr_adaptive, "--", color="#c33", lw=.8, label="피크 임계(직전30일 p95)")
ax.legend(fontsize=6, ncol=5); ax.set_ylabel("kw")
ax.set_title("(a) 평가구간 중 5일 — 규칙 기반 운전상태 표시")
ax = axes[1]
for st, c in COL.items():
    m = EVAL.regime.values == st
    ax.scatter(EVAL["T"][m], np.full(m.sum(), 1), s=3, color=c, marker="|")
ax.axvline(lock_start, color="k", lw=.8, ls=":")
ax.text(lock_start, 1.02, " LOCK 시작", fontsize=6)
ax.set_yticks([]); ax.set_title("(b) 평가 전용 30일 전체 — 상태 띠 (TEST 15일 | LOCK 15일)")
fig.suptitle("FIGURE 10. 운전상태의 시간 분포 (평가구간 전용, 학습구간 제외)", fontsize=9)
fig.tight_layout(); fig.savefig(fig_path("30_regime_timeline")); plt.close(fig)

# ---- FIGURE 11: 2차원 투영 (시각화 전용) — 규칙 라벨 vs k=3 군집
q2, mode = FT.build(cfg["external"]["use_external_data"])
d2 = FT.common_rows(q2, mode)
folds = list(TimeSeriesSplit(n_splits=cfg["data"]["n_splits"]).split(d2))
pre = d2.iloc[folds[0][0]]
FC = ["kw", "kw_ramp4", "kw_std4", "kw_roll4"]
sc = StandardScaler().fit(pre[FC].values)
join = o[["ts15", "regime"]].rename(columns={"ts15": "ts15"})
dd = d2.merge(join, on="ts15", how="inner")
Z = sc.transform(dd[FC].values)
pc = PCA(2, random_state=SEED).fit(sc.transform(pre[FC].values))
XY = pc.transform(Z)
km3 = KMeans(3, n_init=10, random_state=SEED).fit(sc.transform(pre[FC].values))
kl = km3.predict(Z)
idx = np.random.default_rng(SEED).choice(len(dd), size=min(6000, len(dd)), replace=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for st, c in COL.items():
    m = (dd.regime.values == st)[idx]
    axes[0].scatter(XY[idx][m, 0], XY[idx][m, 1], s=3, color=c, alpha=.5, label=st)
axes[0].legend(fontsize=6); axes[0].set_title("(a) 규칙으로 붙인 운전상태")
for k in range(3):
    m = (kl == k)[idx]
    axes[1].scatter(XY[idx][m, 0], XY[idx][m, 1], s=3, alpha=.5, label=f"KMeans cluster {k}")
axes[1].legend(fontsize=6); axes[1].set_title("(b) k=3 을 **미리 정한** 군집")
for a in axes:
    a.set_xlabel(f"PC1 ({pc.explained_variance_ratio_[0]*100:.0f}%)")
    a.set_ylabel(f"PC2 ({pc.explained_variance_ratio_[1]*100:.0f}%)")
fig.suptitle("FIGURE 11. 2차원 투영은 **시각화용**이다. 무리가 떨어져 보이는 것이 개별 모델 성능 향상을 뜻하지 않는다.",
             fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("30_regime_projection")); plt.close(fig)

# ---- FIGURE 12: 상태별 FN/FP + 분모
f = fnfp[fnfp.window == "TEST+LOCK"].set_index("state")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
st_list = ["ALL"] + list(COL)
x = np.arange(len(st_list))
axes[0].bar(x - .2, [f.loc[s, "FN"] for s in st_list], .4, label="FN(놓침) 건수", color="#c33")
axes[0].bar(x + .2, [f.loc[s, "FP"] for s in st_list], .4, label="FP(헛경고) 건수", color="#27b")
axes[0].set_xticks(x); axes[0].set_xticklabels(st_list, rotation=15, ha="right", fontsize=6)
axes[0].legend(fontsize=6); axes[0].set_title("(a) 건수 — 분모가 다르므로 단독 해석 금지")
axes[1].bar(x - .2, [f.loc[s, "n_peak_steps"] for s in st_list], .4, label="실제 고사용량 스텝", color="#933")
axes[1].bar(x + .2, [f.loc[s, "n_nonpeak_steps"] for s in st_list], .4, label="정상 스텝", color="#357")
axes[1].set_yscale("log"); axes[1].set_xticks(x)
axes[1].set_xticklabels(st_list, rotation=15, ha="right", fontsize=6)
axes[1].legend(fontsize=6); axes[1].set_title("(b) 분모 — 상태별 표본 수(로그축)")
axes[2].bar(x - .2, [f.loc[s, "fn_rate_given_peak"] for s in st_list], .4, label="FN율(고사용량 중)", color="#c33")
axes[2].bar(x + .2, [f.loc[s, "fp_rate_given_nonpeak"] for s in st_list], .4, label="FP율(정상 중)", color="#27b")
axes[2].set_xticks(x); axes[2].set_xticklabels(st_list, rotation=15, ha="right", fontsize=6)
axes[2].legend(fontsize=6); axes[2].set_title("(c) 비율 — 분모 보정")
fig.suptitle("FIGURE 12. 상태별 경고 성능(FN/FP). 수치 예측 성능(MAE)과 다른 축이다. 평가기간 TEST+LOCK 30일",
             fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("30_fnfp_by_state")); plt.close(fig)

# ---- FIGURE 13: 전구간 시계열 + FN/FP 표시 + 확대 2패널
fig = plt.figure(figsize=(12, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
ax = fig.add_subplot(gs[0, :])
ax.plot(EVAL["T"], EVAL.kw, color="#222", lw=.7, label="실측 수요(목표시각)")
ax.plot(EVAL["T"], EVAL.score_early_h60, color="#27b", lw=.7, alpha=.8, label="예측(h60, 조기)")
ax.plot(EVAL["T"], EVAL.thr_adaptive, "--", color="#c33", lw=.8, label="고사용량 기준선")
ax.scatter(EVAL["T"][EVAL.FN], EVAL.kw[EVAL.FN], s=16, marker="x", color="#c33", label="FN 놓침", zorder=4)
ax.scatter(EVAL["T"][EVAL.FP], EVAL.kw[EVAL.FP], s=10, marker="^", color="#27b", label="FP 헛경고", zorder=4)
ax.axvline(lock_start, color="k", lw=.8, ls=":")
ax.legend(fontsize=6, ncol=5); ax.set_ylabel("kw")
ax.set_title("(a) 평가 전용 30일 개요 (TEST 15일 | LOCK 15일) — 경고 성능 표시")
fn_t = EVAL["T"][EVAL.FN]
fp_t = EVAL["T"][EVAL.FP]
for j, (tag, center) in enumerate([("대표 놓침(FN) 확대", fn_t.iloc[len(fn_t) // 2] if len(fn_t) else None),
                                   ("대표 헛경고(FP) 확대", fp_t.iloc[len(fp_t) // 2] if len(fp_t) else None)]):
    a = fig.add_subplot(gs[1, j])
    if center is None:
        a.set_title(f"({'bc'[j]}) {tag} — 해당 사례 없음", fontsize=8); continue
    w = EVAL[(EVAL["T"] >= center - pd.Timedelta(hours=6)) & (EVAL["T"] <= center + pd.Timedelta(hours=6))]
    a.plot(w["T"], w.kw, color="#222", lw=1, label="실측")
    a.plot(w["T"], w.score_early_h60, color="#27b", lw=1, label="예측 h60")
    a.plot(w["T"], w.thr_adaptive, "--", color="#c33", lw=.8, label="기준선")
    a.scatter(w["T"][w.FN], w.kw[w.FN], s=30, marker="x", color="#c33")
    a.scatter(w["T"][w.FP], w.kw[w.FP], s=22, marker="^", color="#27b")
    a.set_title(f"({'bc'[j]}) {tag} — {center:%m-%d %H:%M} 전후 6시간", fontsize=8)
    a.legend(fontsize=6); a.tick_params(labelsize=6)
fig.suptitle("FIGURE 13. 실측·예측·기준선·FN·FP (평가구간 전용). FN/FP=경고 성능, MAE=수치 예측 성능", fontsize=9)
fig.tight_layout(); fig.savefig(fig_path("30_timeseries_fn_fp")); plt.close(fig)
print("그림 4종 저장 완료")
