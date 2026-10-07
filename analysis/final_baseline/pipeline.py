"""Baseline v1 — 시간순(rolling-origin) 평가 루프.

폴드마다 TRAIN / CAL / TEST 를 엄격히 시간순으로 분리한다.
TEST 로 적합하는 것은 하나도 없다:
  특성모드 선택 / 점모델 / 분위모델 / conformal 분위 / 밴드 컷 / OOD 컷 /
  레짐 ramp 컷 / 경보 임계(분위->값 변환) 전부 TRAIN 또는 CAL 에서만 산출한다.
피크 임계(직전 30일 p95)는 정의 자체가 과거만 보므로 전 구간에서 인과적으로 계산된다.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

import features as FT
import models as MD

H = [1, 2, 3, 4]                                  # 15/30/45/60 분


def regime_rule(df, ramp_cut):
    """Baseline v1 채택 레짐 탐지기 — 결정시점 정보만, 컷은 TRAIN p90."""
    sw = df.is_operating.values != df.is_operating_lag4.values
    return np.where(sw | (df.kw_ramp4.abs().values >= ramp_cut), "TRANSITION_LIKE",
                    np.where(df.is_operating.values == 0, "LOW_LOAD", "STABLE_OPERATION"))


def run_evaluation(d, mode, cfg):
    """OOF 행 테이블 + 모드선택 로그 + CAL 경보임계 + 레짐 비교를 만든다."""
    seed = cfg["seed"]
    RECIPE = cfg["forecast"].get("point_recipe", "v2_a5")
    rcfg, pcfg = cfg["reliability"], cfg["policy"]
    sets = FT.feature_sets(mode)
    rule = cfg["external"]["selection_rule"]
    folds = list(TimeSeriesSplit(n_splits=cfg["data"]["n_splits"]).split(d))
    sel_rows, thr_rows, regime_rows, out = [], [], [], []

    for fi, (itr, ite) in enumerate(folds):
        cut = int(len(itr) * (1 - cfg["data"]["cal_fraction"]))
        tr, ca, te = itr[:cut], itr[cut:], ite
        TR, CA, TE = d.iloc[tr], d.iloc[ca], d.iloc[te]
        ramp_cut = float(np.quantile(TR.kw_ramp4.abs(), cfg["regime"]["ramp_quantile"]))
        rec = TE[["ts15", "clean", "kw", "thr_adaptive", "is_operating", "생산량",
                  "kw_ramp4", "kw_std4"] + [f"y_h{h}" for h in H]].copy()
        rec["fold"] = fi
        rec["regime"] = regime_rule(TE, ramp_cut)
        rec["ramp_cut_train"] = ramp_cut

        # ---- 레짐 견고성(진단용): TRAIN 적합 GMM3 와의 일치도
        if "gmm3" in cfg["regime"]["compare_methods"]:
            FC = ["kw", "kw_ramp4", "kw_std4"]
            sc = StandardScaler().fit(TR[FC].values)
            gm = GaussianMixture(3, covariance_type="full",
                                 random_state=seed).fit(sc.transform(TR[FC].values))
            cl = gm.predict(sc.transform(TE[FC].values))
            prof = pd.DataFrame({"c": cl, "kw": TE.kw.values,
                                 "ar": TE.kw_ramp4.abs().values}).groupby("c").mean()
            st_c = int(prof.kw.idxmin()); tr_c = int(prof.drop(index=st_c).ar.idxmax())
            g = np.array(["STABLE_OPERATION"] * len(TE), dtype=object)
            g[cl == st_c] = "LOW_LOAD"; g[cl == tr_c] = "TRANSITION_LIKE"
            rec["regime_gmm3"] = g
            regime_rows.append(dict(fold=fi, n=len(TE),
                                    ari_rule_vs_gmm3=float(adjusted_rand_score(rec.regime, g)),
                                    transition_share_rule=float((rec.regime == "TRANSITION_LIKE").mean()),
                                    transition_share_gmm3=float((g == "TRANSITION_LIKE").mean())))

        for h in H:
            ycol = f"y_h{h}"
            ytr, yca = TR[ycol].values, CA[ycol].values
            cal_pred, metr = {}, {}
            for mname, F in sets.items():
                pc = MD.fit_predict_point(cfg["forecast"]["point_model"], seed,
                                          TR[F].values, ytr, TR.kw.values,
                                          CA[F].values, CA.kw.values, RECIPE)
                cal_pred[mname] = pc
                pk = CA[ycol].values >= CA.thr_adaptive.values
                metr[mname] = dict(cal_mae=float(np.abs(yca - pc).mean()),
                                   cal_mae_peak=float(np.abs(yca - pc)[pk].mean()) if pk.any() else np.nan)

            # ---- Module I: horizon 별 특성모드 게이트 (CAL 만 사용, 규칙 사전 명시)
            chosen = "CORE"
            gain = loss = wloss = np.nan
            if "PUBLIC" in metr:
                gain = (metr["CORE"]["cal_mae"] - metr["PUBLIC"]["cal_mae"]) / metr["CORE"]["cal_mae"]
                loss = (metr["PUBLIC"]["cal_mae_peak"] - metr["CORE"]["cal_mae_peak"]) / metr["CORE"]["cal_mae_peak"]
                # CAL 구간폭 비교 (분위모델은 TRAIN, 보정은 CAL 내부 분할로 근사)
                widths = {}
                for mname, F in sets.items():
                    k = int(len(tr) * .8)
                    _, _, cw, _ = MD.fit_cqr(d.iloc[tr[:k]][F].values, d.iloc[tr[:k]][ycol].values,
                                             d.iloc[tr[k:]][F].values, d.iloc[tr[k:]][ycol].values,
                                             CA[F].values[:1], *rcfg["quantile_levels"],
                                             rcfg["conformal_alpha"], seed)
                    widths[mname] = float(np.mean(cw))
                wloss = (widths["PUBLIC"] - widths["CORE"]) / widths["CORE"]
                if (gain >= rule["min_rel_mae_gain"] and loss <= rule["max_rel_peak_mae_loss"]
                        and wloss <= rule["max_rel_width_loss"]):
                    chosen = "PUBLIC"
            sel_rows.append(dict(fold=fi, horizon_min=h * 15, selected_mode=chosen,
                                 cal_mae_core=metr["CORE"]["cal_mae"],
                                 cal_mae_public=metr.get("PUBLIC", {}).get("cal_mae", np.nan),
                                 cal_rel_mae_gain=gain, cal_rel_peak_mae_loss=loss,
                                 cal_rel_width_loss=wloss,
                                 rule=f"gain>={rule['min_rel_mae_gain']} & peak_loss<="
                                      f"{rule['max_rel_peak_mae_loss']} & width_loss<="
                                      f"{rule['max_rel_width_loss']}"))

            # 선택 확정 후 TRAIN+CAL 로 재적합 (TEST 미사용).
            # 두 모드 모두 같은 적합집합으로 재적합해 CORE vs PUBLIC 비교를 공정하게 만든다.
            fit_idx = np.r_[tr, ca]
            FITD = d.iloc[fit_idx]
            for mname, Fm in sets.items():
                rec[f"pred_{mname}_h{h}"] = MD.fit_predict_point(
                    cfg["forecast"]["point_model"], seed, FITD[Fm].values,
                    FITD[ycol].values, FITD.kw.values, TE[Fm].values, TE.kw.values, RECIPE)
            F = sets[chosen]
            rec[f"pred_h{h}"] = rec[f"pred_{chosen}_h{h}"].values
            rec[f"mode_h{h}"] = chosen

            # ---- Module C: CQR 구간 + 신뢰도 밴드 + OOD
            # 분위모델/conformal 도 점예측과 같은 스케일(잔차)에서 적합한다.
            # 스케일이 다르면 구간이 개선된 점예측을 중심으로 놓이지 않는다.
            rbase_tr = TR.kw.values if RECIPE != "v1" else 0.0
            rbase_ca = CA.kw.values if RECIPE != "v1" else 0.0
            rbase_te = TE.kw.values if RECIPE != "v1" else 0.0
            if rcfg.get("conformal_groups") == "level3":
                # 조건부(Mondrian) 보정: 현재 수요의 TRAIN 3분위 구간별 보정분위 (후속 R 근거)
                lvl_cuts = np.quantile(TR.kw, [1 / 3, 2 / 3])
                gcal = np.digitize(CA.kw.values, lvl_cuts)
                gte_ = np.digitize(TE.kw.values, lvl_cuts)
                lo, hi, cal_width, _ = MD.fit_cqr_grouped(
                    TR[F].values, ytr - rbase_tr, CA[F].values, yca - rbase_ca,
                    TE[F].values, gcal, gte_,
                    *rcfg["quantile_levels"], rcfg["conformal_alpha"], seed)
            else:
                lo, hi, cal_width, _ = MD.fit_cqr(TR[F].values, ytr - rbase_tr, CA[F].values,
                                                  yca - rbase_ca, TE[F].values,
                                                  *rcfg["quantile_levels"],
                                                  rcfg["conformal_alpha"], seed)
            lo, hi = lo + rbase_te, hi + rbase_te       # 잔차 -> 수준 복원 (폭은 불변)
            cv, ood_cut = MD.ood_scorer(TR[F].values, rcfg["ood_quantile"])
            maha = cv.mahalanobis(TE[F].values)
            rec[f"int_lo_h{h}"], rec[f"int_hi_h{h}"] = lo, hi
            rec[f"int_width_h{h}"] = hi - lo
            rec[f"band_h{h}"] = MD.bands(hi - lo, cal_width, rcfg["band_width_quantiles"],
                                         maha, ood_cut)
            if h == 4:
                rec["ood_maha"], rec["ood_cut_train"] = maha, ood_cut
            # ---- 경보 임계: 분위 -> 값 변환을 CAL 예측분포에서만 수행
            pc_fin = MD.fit_predict_point(cfg["forecast"]["point_model"], seed,
                                          FITD[F].values, FITD[ycol].values, FITD.kw.values,
                                          CA[F].values, CA.kw.values, RECIPE)
            for qv in pcfg["threshold_grid"]:
                thr_rows.append(dict(fold=fi, horizon_min=h * 15, q=qv,
                                     threshold_kw=float(np.quantile(pc_fin, qv))))

            # ---- 기준선 (평가용)
            for b in cfg["forecast"]["eval_baselines"]:
                if b == cfg["forecast"]["point_model"]:
                    continue
                if b == "persistence":
                    rec[f"base_persistence_h{h}"] = TE.kw.values
                else:
                    mb = MD.point_model(b, seed).fit(d.iloc[fit_idx][FT.CORE].values,
                                                     d.iloc[fit_idx][ycol].values)
                    rec[f"base_{b}_h{h}"] = mb.predict(TE[FT.CORE].values)
        out.append(rec)

    o = pd.concat(out, ignore_index=True)
    o["peak_now"] = (o.kw >= o.thr_adaptive).astype(int)
    for h in H:
        o[f"peak_h{h}"] = (o[f"y_h{h}"] >= o.thr_adaptive).astype(int)
        o[f"abs_err_h{h}"] = (o[f"y_h{h}"] - o[f"pred_h{h}"]).abs()
        o[f"covered_h{h}"] = ((o[f"y_h{h}"] >= o[f"int_lo_h{h}"]) &
                              (o[f"y_h{h}"] <= o[f"int_hi_h{h}"])).astype(int)
        o[f"margin_h{h}"] = o[f"pred_h{h}"] - o.thr_adaptive
        # 피크위험 점수: 예측-임계 여유를 구간폭으로 정규화한 '점수'(확률 아님)
        o[f"peak_risk_score_h{h}"] = (o[f"margin_h{h}"] / o[f"int_width_h{h}"].clip(lower=1e-6)
                                      ).clip(-3, 3)
    o["ood_flag"] = (o.band_h4 == "OOD").astype(int)
    return o, pd.DataFrame(sel_rows), pd.DataFrame(thr_rows), pd.DataFrame(regime_rows)


def forecast_metrics(o, cfg):
    """horizon x 예측기 x 평가창 성능표."""
    rows = []
    preds = ["pred"] + [f"pred_{m}" for m in ("CORE", "PUBLIC") if f"pred_{m}_h4" in o] + \
            [f"base_{b}" for b in cfg["forecast"]["eval_baselines"]
             if b != cfg["forecast"]["point_model"]]
    for wtag, s in [("all_2021", o), ("clean_Jul_Sep", o[o.clean == True])]:
        s = s.reset_index(drop=True)
        for h in H:
            y = s[f"y_h{h}"].values
            pers = np.abs(y - s.kw.values).mean()
            peak = s[f"peak_h{h}"].values == 1
            ramp_hi = s.kw_ramp4.abs().values >= np.quantile(s.kw_ramp4.abs(), .9)
            for p in preds:
                c = f"{p}_h{h}"
                if c not in s:
                    continue
                e = np.abs(y - s[c].values)
                rows.append(dict(window=wtag, predictor=p.replace("pred_", "mode_")
                                 .replace("pred", "baseline_v1_selected"), horizon_min=h * 15,
                                 n=len(s), mae=float(e.mean()),
                                 rmse=float(np.sqrt((e ** 2).mean())),
                                 nmae=float(e.mean() / np.abs(y).mean()),
                                 persistence_improvement=float(1 - e.mean() / pers),
                                 mae_low_load=float(e[s.regime == "LOW_LOAD"].mean()),
                                 mae_stable=float(e[s.regime == "STABLE_OPERATION"].mean()),
                                 mae_transition=float(e[s.regime == "TRANSITION_LIKE"].mean()),
                                 mae_peak_condition=float(e[peak].mean()) if peak.any() else np.nan,
                                 mae_high_ramp=float(e[ramp_hi].mean())))
    return pd.DataFrame(rows)


def reliability_metrics(o):
    rows = []
    for wtag, s in [("all_2021", o), ("clean_Jul_Sep", o[o.clean == True])]:
        s = s.reset_index(drop=True)
        for h in H:
            for b in ["HIGH", "MEDIUM", "LOW", "OOD"]:
                t = s[s[f"band_h{h}"] == b]
                if not len(t):
                    continue
                rows.append(dict(window=wtag, horizon_min=h * 15, band=b, n=len(t),
                                 share=len(t) / len(s), mae=float(t[f"abs_err_h{h}"].mean()),
                                 coverage=float(t[f"covered_h{h}"].mean()),
                                 mean_interval_width=float(t[f"int_width_h{h}"].mean()),
                                 peak_prevalence=float(t[f"peak_h{h}"].mean()),
                                 transition_share=float((t.regime == "TRANSITION_LIKE").mean())))
            pk = s[s[f"peak_h{h}"] == 1]
            rows.append(dict(window=wtag, horizon_min=h * 15, band="ALL", n=len(s), share=1.0,
                             mae=float(s[f"abs_err_h{h}"].mean()),
                             coverage=float(s[f"covered_h{h}"].mean()),
                             mean_interval_width=float(s[f"int_width_h{h}"].mean()),
                             peak_prevalence=float(s[f"peak_h{h}"].mean()),
                             coverage_peak_only=float(pk[f"covered_h{h}"].mean()) if len(pk) else np.nan,
                             transition_share=float((s.regime == "TRANSITION_LIKE").mean())))
    return pd.DataFrame(rows)
