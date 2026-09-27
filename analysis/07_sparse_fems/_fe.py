"""07 공통 특성/분할/지표 모듈 — Sparse-FEMS 실험 전체가 이 파일만 쓴다.

핵심 설계 결정 (누수 규칙 대응):
  - 결정시점 t 에서 t+15/30/45/60 분을 direct 예측한다. 특성은 모두 t 시점까지의 정보.
  - '평균' 열은 같은 시간대 4개 분값의 평균이므로 타깃을 포함한다 → 전면 제외.
  - '공장인원'은 생산량과 spearman 0.994 로 중복 → 제외(06 보고서 근거).
  - '인건비'는 사실상 2값 상수 → 제외.
  - 생산량(t)은 '생산계획으로 사전 확정' 가정에서만 사용 가능 → L3 에 넣되,
    엄격 변형 L3s(생산량 1시간 지연만 사용) 를 같이 보고해 가정 민감도를 드러낸다.
  - 날씨는 t 시점 관측값만 사용(미래 예보 미사용).
  - 피크 임계·스케일러·클러스터·보정계수는 train 에서만 적합한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from common import load_power, power_long, save_table, fig_path, mpl, SEED, TAB  # noqa: F401

HORIZONS = [1, 2, 3, 4]          # 15/30/45/60 분
EXCLUDED = {"평균": "같은 시간 4개 분값의 평균 = 타깃 포함(누수)",
            "공장인원": "생산량과 spearman 0.994 중복",
            "인건비": "고유값 2개, 사실상 상수"}

L0 = ["kw", "kw_lag1", "kw_lag2", "kw_lag3", "kw_lag4", "kw_lag8", "kw_lag96",
      "kw_roll4", "kw_roll96", "kw_diff1", "kw_ramp4", "kw_std4", "kw_std96", "kw_max96"]
CAL = ["시간", "qi", "tod", "day", "m", "d", "is_weekend", "전기요금(계절)"]
WEA = ["기온", "습도", "풍속", "강수량"]
PRD = ["생산량", "prod_lag4", "prod_diff", "is_operating"]
PRD_STRICT = ["prod_lag4", "prod_diff", "is_operating_lag4"]

LEVELS = {"L0_power_only": L0,
          "L1_+calendar": L0 + CAL,
          "L2_+weather": L0 + CAL + WEA,
          "L3_+production": L0 + CAL + WEA + PRD,
          "L3s_+production_strict": L0 + CAL + WEA + PRD_STRICT}


def build(latent_cols=None):
    """15분 long 시계열 + 결정시점 특성 + 다중 horizon 타깃."""
    q = power_long(load_power()).sort_values("ts15").reset_index(drop=True)
    q["qi"] = q["q"].map({"15분": 0, "30분": 1, "45분": 2, "60분": 3})
    q["tod"] = q["시간"] * 4 + q["qi"]                     # 0..95 하루 내 위치
    q["is_weekend"] = q["day"].isin([6, 7]).astype(int)
    for k in (1, 2, 3, 4, 8, 96):
        q[f"kw_lag{k}"] = q.kw.shift(k)
    q["kw_roll4"] = q.kw.rolling(4).mean()
    q["kw_roll96"] = q.kw.rolling(96).mean()
    q["kw_std4"] = q.kw.rolling(4).std()
    q["kw_std96"] = q.kw.rolling(96).std()
    q["kw_max96"] = q.kw.rolling(96).max()
    q["kw_diff1"] = q.kw - q.kw_lag1
    q["kw_ramp4"] = q.kw - q.kw_lag4
    q["is_operating"] = (q["생산량"] > 0).astype(int)
    q["prod_lag4"] = q["생산량"].shift(4)
    q["prod_diff"] = q["생산량"] - q["prod_lag4"]
    q["is_operating_lag4"] = q["is_operating"].shift(4)
    for h in HORIZONS:
        q[f"y_h{h}"] = q.kw.shift(-h)
    q["clean"] = q["날짜"] >= 20210701
    if latent_cols is not None:
        lat = pd.read_csv(TAB / "07D_latent_state_features.csv", parse_dates=["ts15"],
                          encoding="utf-8-sig")
        q = q.merge(lat[["ts15"] + latent_cols], on="ts15", how="left")
    return q


def window(q, wtag):
    return q if wtag == "all_2021" else q[q.clean]


def split(sub, feats, h, ratios=(.6, .8)):
    """시간순 train / calibration / test. 결측 특성행은 제거(앞쪽 워밍업 구간)."""
    need = list(dict.fromkeys(feats + [f"y_h{h}"]))
    d = sub.dropna(subset=need).reset_index(drop=True)
    n = len(d)
    i1, i2 = int(n * ratios[0]), int(n * ratios[1])
    return d.iloc[:i1].copy(), d.iloc[i1:i2].copy(), d.iloc[i2:].copy()


def peak_threshold(tr, col="kw", qtl=.95):
    """피크 임계는 train 에서만 산출한다(규칙 3)."""
    return float(np.quantile(tr[col], qtl))


def metrics(te, y, p, thr, ramp_q=None):
    e = np.abs(y - p)
    oper = te.is_operating.values == 1
    peak = y >= thr
    ramp = np.abs(te.kw_ramp4.values)
    rq = float(np.quantile(ramp, .9)) if ramp_q is None else ramp_q
    hi = ramp >= rq
    scale = float(np.mean(np.abs(y)))
    return dict(n_test=len(y), mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                nmae=float(e.mean() / scale),
                mae_operating=float(e[oper].mean()) if oper.any() else np.nan,
                mae_idle=float(e[~oper].mean()) if (~oper).any() else np.nan,
                mae_peak=float(e[peak].mean()) if peak.any() else np.nan,
                mae_high_ramp=float(e[hi].mean()) if hi.any() else np.nan,
                err_p90=float(np.quantile(e, .9)), err_p99=float(np.quantile(e, .99)),
                err_max=float(e.max()), peak_prevalence=float(peak.mean()))


def paired_boot(e_new, e_base, n=2000, seed=SEED):
    """절대오차 차(new - base) 의 paired bootstrap CI. 음수면 new 가 개선."""
    d = np.asarray(e_new) - np.asarray(e_base)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    return dict(delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
                significant=bool(hi < 0 or lo > 0), improves=bool(hi < 0))
