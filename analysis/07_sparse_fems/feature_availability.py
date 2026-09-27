"""07-0. 특성 가용성 감사 + 누수 실증.

산출: 07_feature_availability.csv (변수별 결정시점 가용성/안전성)
      07_leakage_checks.csv       (제외 결정의 수치 근거)
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, save_table

q = build()
rows = []

def add(v, group, avail, future_known, post_event, safe, note):
    rows.append(dict(variable=v, group=group, available_at_t=avail,
                     known_future_schedule=future_known, observed_only_after_event=post_event,
                     safe_for_h1_h4=safe, note=note))

for v in _fe.L0:
    add(v, "power_history", True, False, False, True, "t 시점까지의 집계 수요만 사용")
for v in _fe.CAL:
    add(v, "calendar", True, True, False, True, "달력/계절요금 — 사전 확정")
for v in _fe.WEA:
    add(v, "weather", True, False, False, True, "t 시점 관측값만 사용(예보 미사용)")
add("생산량", "production", True, "가정", False, "가정부",
    "ERP 시간 집계. 생산계획으로 사전 확정 가정에서만 안전 → L3s 로 민감도 확인")
add("prod_lag4", "production", True, False, False, True, "1시간 전 생산량 — 무조건 안전")
add("prod_diff", "production", True, "가정", False, "가정부", "생산량(t) - prod_lag4")
add("is_operating", "production", True, "가정", False, "가정부", "생산량(t)>0")
add("is_operating_lag4", "production", True, False, False, True, "1시간 전 조업 여부")
for v, why in _fe.EXCLUDED.items():
    add(v, "EXCLUDED", True, False, v == "평균", False, why)
av = pd.DataFrame(rows)
save_table(av, "07_feature_availability")

# ---- 누수 실증 1: '평균'은 같은 시간의 4개 분값 평균인가
p = _fe.load_power()
recon = p[["15분", "30분", "45분", "60분"]].mean(1)
chk = [dict(check="평균 == mean(15,30,45,60)",
            value=float((np.abs(recon - p["평균"]) <= 0.5).mean()),
            interpretation="1.0 이면 평균은 타깃 4개를 포함하는 파생열 → 특성으로 쓰면 직접 누수")]
# ---- 누수 실증 2: 공장인원-생산량 중복
chk.append(dict(check="spearman(생산량, 공장인원)",
                value=float(p[["생산량", "공장인원"]].corr(method="spearman").iloc[0, 1]),
                interpretation="0.99 이상이면 독립적 인원 정보로 볼 수 없다"))
chk.append(dict(check="인건비 nunique", value=float(p["인건비"].nunique()),
                interpretation="사실상 상수 → 정보 없음"))
# ---- 누수 실증 3: 랜덤 분할 vs 시간순 분할 (중복 프로파일 인식)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
F = _fe.LEVELS["L3_+production"]
d = q.dropna(subset=F + ["y_h4"]).reset_index(drop=True)
rng = np.random.default_rng(_fe.SEED)
idx = rng.permutation(len(d))
cut = int(len(d) * .8)
mk = lambda: HistGradientBoostingRegressor(random_state=_fe.SEED)
rtr, rte = d.iloc[idx[:cut]], d.iloc[idx[cut:]]
m = mk().fit(rtr[F].values, rtr.y_h4.values)
mae_rand = mean_absolute_error(rte.y_h4, m.predict(rte[F].values))
ttr, tte = d.iloc[:cut], d.iloc[cut:]
m = mk().fit(ttr[F].values, ttr.y_h4.values)
mae_temp = mean_absolute_error(tte.y_h4, m.predict(tte[F].values))
chk += [dict(check="h4 MAE random split", value=mae_rand,
             interpretation="복제 프로파일이 train/test 양쪽에 들어가 낙관적"),
        dict(check="h4 MAE temporal split", value=mae_temp,
             interpretation="이 값만 실제 기대 성능으로 보고한다")]
save_table(pd.DataFrame(chk), "07_leakage_checks")
print(pd.DataFrame(chk).to_string(index=False))
