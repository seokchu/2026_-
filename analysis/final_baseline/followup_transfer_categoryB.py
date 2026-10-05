"""후속 U. Category B — 타 공장 공개데이터 전이 검증 (실제 수행).

데이터: Lee, Baek, Kim (2022) Scientific Data, 한국 제조공장 10개소 1분 전력 (2019-03~09),
        figshare DOI 10.6084/m9.figshare.14822256.v9, CC BY 4.0.
        캐시 `analysis/external_cache/categoryB/` (원본 재배포 금지 규정은 없으나 커밋하지 않는다).

질문: "다른 공장의 공개 전력데이터로 학습한 모델이 저계측 공장의 피크 예측을 대신할 수 있는가?"
설계:
  - 1분 데이터를 15분 평균으로 재표본(KAMP 15분 격자와 정합).
  - 전이 가능 특성만 사용: 집계전력 이력 + 달력(두 데이터셋 공통). 생산량·기상은 타 공장에 없다.
  - 수요 규모가 공장마다 다르므로 **공장별 robust 스케일**(train 중앙값/IQR)로 정규화해 학습·평가한다.
  - 평가는 각 대상 공장의 시간순 뒤쪽 20%.
  모델 3종:
    local   : 대상 공장 자체 과거로 학습 (상한 기준)
    transfer: 다른 9개 공장으로 학습, 대상 공장은 학습에 미포함 (leave-one-factory-out)
    persistence
  사전 판정: 전이 모델의 nMAE 가 local 대비 10% 이내면 '전이 유효'.
"""
import sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, fig_path, mpl                  # noqa: E402
import models as MD                                           # noqa: E402
plt = mpl()

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
SEED = cfg["seed"]
CACHE = ROOT / "analysis" / "external_cache" / "categoryB"
FDIR = CACHE / "Data_list" / "Factories"
H = 4                                                          # t+60분
FEATS = ["kw_z", "lag1", "lag2", "lag3", "lag4", "lag8", "lag96", "roll4", "roll96",
         "std4", "std96", "max96", "diff1", "ramp4", "tod", "day", "is_weekend", "m"]

if not FDIR.exists():
    z = CACHE / "Data_list.zip"
    assert z.exists(), f"캐시 없음: {z} (figshare 다운로드 필요)"
    zipfile.ZipFile(z).extractall(CACHE)


def load_factory(p):
    d = pd.read_csv(p)
    d.columns = ["ts", "kw"]
    d["ts"] = pd.to_datetime(d.ts, errors="coerce")
    d = d.dropna(subset=["ts"]).set_index("ts").sort_index()
    q = d.kw.resample("15min").mean().to_frame("kw").reset_index()
    q = q.dropna(subset=["kw"]).reset_index(drop=True)
    return q


def featurize(q):
    q = q.copy()
    for k in (1, 2, 3, 4, 8, 96):
        q[f"lag{k}"] = q.kw.shift(k)
    q["roll4"] = q.kw.rolling(4).mean(); q["roll96"] = q.kw.rolling(96).mean()
    q["std4"] = q.kw.rolling(4).std(); q["std96"] = q.kw.rolling(96).std()
    q["max96"] = q.kw.rolling(96).max()
    q["diff1"] = q.kw - q.lag1; q["ramp4"] = q.kw - q.lag4
    q["tod"] = q.ts.dt.hour * 4 + q.ts.dt.minute // 15
    q["day"] = q.ts.dt.dayofweek + 1
    q["is_weekend"] = q.day.isin([6, 7]).astype(int)
    q["m"] = q.ts.dt.month
    q["y"] = q.kw.shift(-H)
    # 적응형 피크 임계(직전 30일 p95, 과거만)
    q["thr"] = q.kw.shift(1).rolling(96 * 30, min_periods=96 * 7).quantile(.95)
    return q


facs = {}
for p in sorted(FDIR.glob("*.csv")):
    q = featurize(load_factory(p))
    q["factory"] = p.stem
    facs[p.stem] = q.dropna(subset=["lag96", "roll96", "y", "thr"]).reset_index(drop=True)
    print(p.stem, len(facs[p.stem]))

ZC = ["kw_z", "lag1", "lag2", "lag3", "lag4", "lag8", "lag96", "roll4", "roll96",
      "std4", "std96", "max96", "diff1", "ramp4"]


def scaled(q, cut):
    """공장별 robust 스케일 — 척도는 해당 공장의 train 구간에서만 산출."""
    tr = q.iloc[:cut]
    med, iqr = float(tr.kw.median()), float(np.subtract(*np.percentile(tr.kw, [75, 25])))
    iqr = iqr if iqr > 1e-6 else max(float(tr.kw.std()), 1e-6)
    s = q.copy()
    s["kw_z"] = (s.kw - med) / iqr
    for c in ["lag1", "lag2", "lag3", "lag4", "lag8", "lag96", "roll4", "roll96", "max96"]:
        s[c] = (s[c] - med) / iqr
    for c in ["std4", "std96", "diff1", "ramp4"]:
        s[c] = s[c] / iqr
    s["y_z"] = (s.y - med) / iqr
    s["scale_med"], s["scale_iqr"] = med, iqr
    return s


CUT = .8
prep = {k: scaled(q, int(len(q) * CUT)) for k, q in facs.items()}
rows = []
for tgt, q in prep.items():
    cut = int(len(q) * CUT)
    tr, te = q.iloc[:cut], q.iloc[cut:]
    med, iqr = float(q.scale_med.iloc[0]), float(q.scale_iqr.iloc[0])
    y_true = te.y.values
    pers = np.abs(y_true - te.kw.values).mean()
    peak = (y_true >= te.thr.values).astype(int)
    preds = {"persistence": te.kw.values}
    m = MD.point_model("hgb", SEED).fit(tr[ZC + ["tod", "day", "is_weekend", "m"]].values,
                                        tr.y_z.values)
    preds["local"] = m.predict(te[ZC + ["tod", "day", "is_weekend", "m"]].values) * iqr + med
    # leave-one-factory-out 전이: 대상 공장 데이터는 전혀 쓰지 않는다
    src = pd.concat([prep[k].iloc[:int(len(prep[k]) * CUT)] for k in prep if k != tgt],
                    ignore_index=True)
    mt = MD.point_model("hgb", SEED).fit(src[ZC + ["tod", "day", "is_weekend", "m"]].values,
                                         src.y_z.values)
    preds["transfer_leave_one_out"] = mt.predict(te[ZC + ["tod", "day", "is_weekend", "m"]].values) * iqr + med
    from sklearn.metrics import average_precision_score
    for name, p in preds.items():
        e = np.abs(y_true - p)
        rows.append(dict(target_factory=tgt, model=name, n_test=len(te),
                         mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                         nmae=float(e.mean() / np.abs(y_true).mean()),
                         persistence_improvement=float(1 - e.mean() / pers),
                         peak_prevalence=float(peak.mean()),
                         peak_ap=float(average_precision_score(peak, p)) if peak.sum() else np.nan,
                         mean_kw=float(np.abs(y_true).mean())))
res = pd.DataFrame(rows)
piv = res.pivot_table(index="target_factory", columns="model", values="nmae")
piv["transfer_vs_local_rel"] = piv.transfer_leave_one_out / piv.local - 1
piv["transfer_beats_persistence"] = piv.transfer_leave_one_out < piv.persistence
piv = piv.reset_index()
save_table(res, "24_categoryB_transfer_by_factory")
save_table(piv, "24_categoryB_transfer_summary")
ok = (piv.transfer_vs_local_rel <= .10).sum()
verdict = ("전이 유효(10% 이내) — %d/%d 공장" % (ok, len(piv)) if ok >= len(piv) * .5
           else "전이 비유효 — %d/%d 공장만 10%% 이내" % (ok, len(piv)))
save_table(pd.DataFrame([dict(
    verdict=verdict, n_factories=len(piv),
    n_within_10pct=int(ok),
    n_transfer_beats_persistence=int(piv.transfer_beats_persistence.sum()),
    median_transfer_vs_local_rel=float(piv.transfer_vs_local_rel.median()),
    rule="전이 nMAE 가 local 대비 +10% 이내인 공장이 과반이면 '전이 유효'",
    source="figshare 10.6084/m9.figshare.14822256.v9 (CC BY 4.0), 1분->15분 재표본",
    note="생산량·기상 없음. 집계전력+달력만 전이. KAMP 공장은 이 실험의 대상이 아니다(척도·기간 불일치)")]),
    "24_categoryB_transfer_verdict")
print(res.round(4).to_string(index=False))
print(piv.round(4).to_string(index=False))
print("판정:", verdict)

fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(piv))
ax.bar(x - .2, piv.local, .4, label="local 학습")
ax.bar(x + .2, piv.transfer_leave_one_out, .4, label="타 공장 전이")
ax.set_xticks(x); ax.set_xticklabels(piv.target_factory, rotation=20, ha="right", fontsize=7)
ax.set_ylabel("nMAE"); ax.legend(fontsize=7)
ax.set_title("Category B 전이 — leave-one-factory-out (h60, 15분 재표본)")
fig.tight_layout(); fig.savefig(fig_path("24_categoryB_transfer")); plt.close(fig)
