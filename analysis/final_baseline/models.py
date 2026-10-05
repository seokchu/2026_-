"""Baseline v1 — Module B/C 모델. 점예측(HGB) + 분위예측 + conformal 보정(CQR).

깊은 신경망/파운데이션 모델은 쓰지 않는다(근거: 보고서 10·15 — 트리 모델이 이미 강하고
CPU 재현성이 좋다). 비교 기준선(persistence / ridge / rf)은 평가용으로만 유지한다.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

HGB_PARAMS = dict(random_state=None)        # 기본 하이퍼파라미터. 무차별 탐색 금지(CPU 제약)


def point_model(kind, seed):
    if kind == "hgb":
        return HistGradientBoostingRegressor(random_state=seed)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(1.0))
    if kind == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                     random_state=seed, n_jobs=-1)
    raise ValueError(kind)


def quantile_model(alpha, seed):
    return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, random_state=seed)


def fit_cqr(Xtr, ytr, Xcal, ycal, Xte, lo_q, hi_q, alpha, seed):
    """분위 모델은 TRAIN 에서, conformal 보정분위는 CAL 에서만 적합한다.

    반환: (test 하한, test 상한, CAL 구간폭) — CAL 구간폭이 밴드 컷의 근거가 된다.
    """
    m_lo = quantile_model(lo_q, seed).fit(Xtr, ytr)
    m_hi = quantile_model(hi_q, seed).fit(Xtr, ytr)
    c_lo, c_hi = m_lo.predict(Xcal), m_hi.predict(Xcal)
    E = np.maximum(c_lo - ycal, ycal - c_hi)
    Q = float(np.quantile(E, min(1.0, (1 - alpha) * (1 + 1 / len(E)))))
    cal_width = (c_hi + Q) - (c_lo - Q)
    return m_lo.predict(Xte) - Q, m_hi.predict(Xte) + Q, cal_width, (m_lo, m_hi, Q)


def ood_scorer(Xtr, q):
    """Mahalanobis 거리 + TRAIN 분위 컷. OOD 는 '큰 오차' 와 다른 개념이다."""
    cv = EmpiricalCovariance().fit(Xtr)
    cut = float(np.quantile(cv.mahalanobis(Xtr), q))
    return cv, cut


def bands(width, cal_width, band_q, maha, ood_cut):
    cuts = np.quantile(cal_width, band_q)
    return np.where(maha > ood_cut, "OOD",
                    np.where(width <= cuts[0], "HIGH",
                             np.where(width <= cuts[1], "MEDIUM", "LOW")))
