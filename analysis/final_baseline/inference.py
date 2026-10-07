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
    dp = b.get("default_policy", {})
    early_q = early_q if early_q is not None else dp.get(
        "early_q", max(cfg["policy"]["threshold_grid"]))
    confirm_q = confirm_q if confirm_q is not None else dp.get(
        "confirm_q", max(cfg["policy"]["threshold_grid"]))
    gate = dp.get("gate", "none")
    out = pd.DataFrame({"timestamp": rows.ts15.values,
                        "current_peak_threshold": rows.thr_adaptive.values})
    for h in b["horizons"]:
        m = b["models"][h]
        X = rows[m["features"]].values
        p = m["point"].predict(X)
        if m.get("point_recipe", "v1") != "v1":
            p = p + rows.kw.values            # 잔차타깃 -> 수준값 복원
        raw_lo, raw_hi = m["q_lo"].predict(X), m["q_hi"].predict(X)
        spec = m.get("conformal_spec", {"mode": "global", "global_Q": m["conformal_Q"]})
        if spec.get("mode") == "level3":
            groups = np.digitize(rows.kw.values, np.asarray(spec["level_cuts"], dtype=float))
            qmap = spec.get("group_Q", {})
            qg = float(spec["global_Q"])
            qarr = np.asarray([float(qmap.get(int(g), qmap.get(str(int(g)), qg)))
                               for g in groups])
        else:
            qarr = np.full(len(rows), float(spec.get("global_Q", m["conformal_Q"])))
        lo = raw_lo - qarr
        hi = raw_hi + qarr
        if m.get("point_recipe", "v1") != "v1":
            lo, hi = lo + rows.kw.values, hi + rows.kw.values   # 잔차 -> 수준 복원
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
    raw_confirm = (out[f"forecast_{hc*15}m"] >=
                   b["models"][hc]["alert_thresholds"][confirm_q]).values
    early_mask = out.early_peak_risk.astype(bool).values
    early_rel = out[f"reliability_{he*15}m"]
    early_unsure = early_rel.isin(cfg["diagnosis"]["uncertain_bands"]).values
    if gate == "confirm_only_if_unsure":
        confirmed = early_mask & (raw_confirm | ~early_unsure)
    else:
        # none / advisory_if_unsure: 실제 경보는 단기 확인을 통과해야 한다.
        confirmed = early_mask & raw_confirm
    out["confirmed_peak_risk"] = confirmed.astype(int)
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
