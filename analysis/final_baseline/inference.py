"""Baseline v1 — 배포 추론. 평가가 아니라 '다음 시점 운용' 전용.

artifacts/baseline_v1.joblib 을 읽어 결정시점 행 1개 이상에 대해 작업자 출력 필드를 만든다.
in-sample 성능을 평가 성능으로 보고하지 않는다(그 수치는 23_baseline_final_metrics.csv 가 유일).
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as FT                                   # noqa: E402

ART = HERE / "artifacts" / "baseline_v1.joblib"


def load(path=ART):
    return joblib.load(path)


def predict(rows, bundle=None, early_q=None, confirm_q=None):
    """rows: features.build() 산출 프레임의 결정시점 행들(피크 임계 포함)."""
    b = bundle or load()
    cfg = b["config"]
    early_q = early_q if early_q is not None else max(cfg["policy"]["threshold_grid"])
    confirm_q = confirm_q if confirm_q is not None else max(cfg["policy"]["threshold_grid"])
    out = pd.DataFrame({"timestamp": rows.ts15.values,
                        "current_peak_threshold": rows.thr_adaptive.values})
    for h in b["horizons"]:
        m = b["models"][h]
        X = rows[m["features"]].values
        p = m["point"].predict(X)
        lo = m["q_lo"].predict(X) - m["conformal_Q"]
        hi = m["q_hi"].predict(X) + m["conformal_Q"]
        w = hi - lo
        maha = m["ood"].mahalanobis(X)
        c0, c1 = m["band_cuts"]
        out[f"forecast_{h*15}m"] = p.round(2)
        out[f"interval_low_{h*15}m"] = lo.round(2)
        out[f"interval_high_{h*15}m"] = hi.round(2)
        out[f"reliability_{h*15}m"] = np.where(maha > m["ood_cut"], "OOD",
            np.where(w <= c0, "HIGH", np.where(w <= c1, "MEDIUM", "LOW")))
        out[f"peak_risk_score_{h*15}m"] = np.clip(
            (p - rows.thr_adaptive.values) / np.clip(w, 1e-6, None), -3, 3).round(3)
    he, hc = cfg["policy"]["early_horizon_min"] // 15, cfg["policy"]["confirm_horizon_min"] // 15
    out["early_peak_risk"] = (out[f"forecast_{he*15}m"] >=
                              b["models"][he]["alert_thresholds"][early_q]).astype(int)
    out["confirmed_peak_risk"] = (out.early_peak_risk.astype(bool) &
                                  (out[f"forecast_{hc*15}m"] >=
                                   b["models"][hc]["alert_thresholds"][confirm_q])).astype(int)
    sw = rows.is_operating.values != rows.is_operating_lag4.values
    out["operating_regime"] = np.where(sw | (np.abs(rows.kw_ramp4.values) >= b["ramp_cut"]),
                                       "TRANSITION_LIKE",
                                       np.where(rows.is_operating.values == 0, "LOW_LOAD",
                                                "STABLE_OPERATION"))
    out["ood_flag"] = (out[f"reliability_{he*15}m"] == "OOD").astype(int)
    unsure = out[f"reliability_{he*15}m"].isin(["LOW", "OOD"]).values
    explained = ((out.operating_regime == "TRANSITION_LIKE").values |
                 (np.abs(rows.kw_ramp4.values) >= b["ramp_cut"]) |
                 (rows.is_operating.values == 0) |
                 (out.early_peak_risk.values == 1) & (out.confirmed_peak_risk.values == 0))
    out["uncertainty_explained"] = explained.astype(int)
    out["information_gap_flag"] = (unsure & ~explained).astype(int)
    out["advisory_level"] = np.where(out.ood_flag == 1, "HUMAN_REVIEW",
        np.where((out.confirmed_peak_risk == 1) & ~unsure, "CONFIRMED_WARNING",
        np.where((out.confirmed_peak_risk == 1) & unsure, "LOW_CONFIDENCE_WARNING",
        np.where(out.information_gap_flag == 1, "INFORMATION_GAP",
        np.where(out.early_peak_risk == 1, "EARLY_WATCH", "NORMAL")))))
    return out


if __name__ == "__main__":
    q, mode = FT.build()
    d = FT.common_rows(q, mode)
    print(predict(d.tail(5)).T.to_string())
