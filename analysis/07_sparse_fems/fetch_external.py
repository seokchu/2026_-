"""07-N-0. 외부 공공데이터 수집·캐싱 (Category A 만).

원칙:
  - 예측시점에 얻을 수 있는 정보만 쓴다. 미래 예보는 재현 가능한 과거 예보 소스가 없어 제외한다.
  - 한국 공장이 현실적으로 조달 가능한 무료·무인증 소스만 쓴다.
  - 원본 캐시는 analysis/external_cache/ (gitignore). 커밋하는 것은 파생 특성표와 출처표뿐이다.
  - KAMP 공장의 위치는 문서화되어 있지 않다. 따라서 KAMP 기온과 국내 관측소 기온의
    상관으로 '가장 가까운 후보 관측소'를 추정하고, 그 추정 자체를 결과로 기록한다.

소스:
  1) 공휴일/근무일  : python `holidays` 패키지 (KR), MIT 라이선스, 오프라인 규칙 기반
  2) 시간별 기상    : NOAA NCEI Integrated Surface Database (ISD) global-hourly, 무인증·공개
  3) 태양기하       : 계산값(외부 다운로드 없음) — 일출/일몰/태양고도
  4) 요금 시간대구조: 한전 산업용 계절·시간대 구분(경/중간/최대부하)의 '구조'만 사용. 금액(KRW) 미사용.
"""
import sys, io, urllib.request, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_power, save_table, ANA

CACHE = ANA / "external_cache"
CACHE.mkdir(exist_ok=True)
YEAR = 2021
# NOAA ISD 국내 주요 관측소 (USAF 번호 + 99999)
STATIONS = {"47108099999": "SEOUL", "47112099999": "INCHEON", "47133099999": "DAEJEON",
            "47143099999": "DAEGU", "47152099999": "ULSAN", "47156099999": "GWANGJU",
            "47159099999": "BUSAN", "47105099999": "CHUNCHEON", "47131099999": "CHEONGJU",
            "47138099999": "POHANG"}
ISD_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{y}/{s}.csv"


def fetch(url, path):
    if path.exists() and path.stat().st_size > 1000:
        return path
    with urllib.request.urlopen(url, timeout=180) as r:
        path.write_bytes(r.read())
    return path


def parse_isd(path):
    """ISD 고정폭 필드 파싱. 결측 코드(9999...)는 NaN."""
    d = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({"ts": pd.to_datetime(d.DATE)})
    tmp = d.TMP.str.split(",", expand=True)
    out["ext_temp"] = pd.to_numeric(tmp[0], errors="coerce") / 10
    out.loc[tmp[0].astype(str).str.contains("9999"), "ext_temp"] = np.nan
    dew = d.DEW.str.split(",", expand=True)
    out["ext_dewpoint"] = pd.to_numeric(dew[0], errors="coerce") / 10
    out.loc[dew[0].astype(str).str.contains("9999"), "ext_dewpoint"] = np.nan
    slp = d.SLP.str.split(",", expand=True)
    out["ext_pressure"] = pd.to_numeric(slp[0], errors="coerce") / 10
    out.loc[out.ext_pressure > 1080, "ext_pressure"] = np.nan
    wnd = d.WND.str.split(",", expand=True)
    out["ext_wind"] = pd.to_numeric(wnd[3], errors="coerce") / 10
    out.loc[out.ext_wind > 90, "ext_wind"] = np.nan
    vis = d.VIS.str.split(",", expand=True)
    out["ext_visibility"] = pd.to_numeric(vis[0], errors="coerce")
    out.loc[out.ext_visibility >= 999999, "ext_visibility"] = np.nan
    cig = d.CIG.str.split(",", expand=True)
    out["ext_ceiling"] = pd.to_numeric(cig[0], errors="coerce")
    out.loc[out.ext_ceiling >= 99999, "ext_ceiling"] = np.nan
    out["lat"], out["lon"], out["name"] = d.LATITUDE.iloc[0], d.LONGITUDE.iloc[0], d.NAME.iloc[0]
    # UTC -> KST
    out["ts_kst"] = out.ts + pd.Timedelta(hours=9)
    out["ts_h"] = out.ts_kst.dt.floor("h")
    g = out.groupby("ts_h")[["ext_temp", "ext_dewpoint", "ext_pressure", "ext_wind",
                             "ext_visibility", "ext_ceiling"]].mean()
    return g, dict(name=out.name.iloc[0], lat=float(out.lat.iloc[0]), lon=float(out.lon.iloc[0]),
                   n_obs=len(out))


# ---- 1) KAMP 기온과 비교해 후보 관측소 추정
p = load_power()
kamp = p[["ts", "기온", "습도", "풍속", "강수량"]].copy()
kamp["ts_h"] = pd.to_datetime(kamp.ts).dt.floor("h")
rows, series = [], {}
for sid, label in STATIONS.items():
    try:
        f = fetch(ISD_URL.format(y=YEAR, s=sid), CACHE / f"isd_{sid}_{YEAR}.csv")
        g, meta = parse_isd(f)
    except Exception as e:                                   # 네트워크/포맷 실패는 기록만
        rows.append(dict(station_id=sid, label=label, status=f"FAILED: {type(e).__name__}"))
        continue
    series[sid] = g
    m = kamp.merge(g.reset_index(), on="ts_h", how="inner")
    ok = m.기온.notna() & m.ext_temp.notna()
    rows.append(dict(station_id=sid, label=label, status="OK", station_name=meta["name"],
                     lat=meta["lat"], lon=meta["lon"], n_matched=int(ok.sum()),
                     pearson_temp=float(np.corrcoef(m.기온[ok], m.ext_temp[ok])[0, 1]),
                     mae_temp=float(np.abs(m.기온[ok] - m.ext_temp[ok]).mean()),
                     bias_temp=float((m.ext_temp[ok] - m.기온[ok]).mean())))
match = pd.DataFrame(rows).sort_values("mae_temp")
save_table(match, "07N_external_station_match")
best = match[match.status == "OK"].iloc[0]
print("추정 최근접 관측소:", best.label, best.station_name, "MAE", round(best.mae_temp, 2))

# ---- 2) 공휴일/근무일 (holidays 패키지)
import holidays
kr = holidays.KR(years=[YEAR - 1, YEAR, YEAR + 1])
days = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
hol = pd.DataFrame({"date": days})
hol["is_holiday"] = [int(d.date() in kr) for d in days]
hol["holiday_name"] = [kr.get(d.date(), "") for d in days]
hol["is_weekend"] = [int(d.weekday() >= 5) for d in days]
hol["is_workday"] = ((hol.is_holiday == 0) & (hol.is_weekend == 0)).astype(int)
nonwork = hol.is_workday == 0
hol["is_bridge_day"] = 0
for i in range(1, len(hol) - 1):                            # 비근무일 사이에 낀 근무일
    if hol.is_workday.iloc[i] == 1 and nonwork.iloc[i - 1] and nonwork.iloc[i + 1]:
        hol.loc[i, "is_bridge_day"] = 1
nw_idx = np.flatnonzero(nonwork.values)
hol["days_to_next_nonworkday"] = [int(nw_idx[nw_idx >= i][0] - i) if (nw_idx >= i).any() else -1
                                  for i in range(len(hol))]
hol["days_since_prev_nonworkday"] = [int(i - nw_idx[nw_idx <= i][-1]) if (nw_idx <= i).any() else -1
                                     for i in range(len(hol))]
hol["nonwork_run_len"] = 0
r = 0
for i in range(len(hol) - 1, -1, -1):                        # 이후 연속 비근무일 길이
    r = r + 1 if nonwork.iloc[i] else 0
    hol.loc[i, "nonwork_run_len"] = r
save_table(hol, "07N_external_calendar_2021")

# ---- 3) 태양기하 (계산, 다운로드 없음) — 관측소 위경도 사용
lat, lon = float(best.lat), float(best.lon)
h15 = pd.date_range(f"{YEAR}-01-01 00:15", f"{YEAR}-12-31 23:45", freq="15min")
doy = h15.dayofyear.values
frac_h = h15.hour.values + h15.minute.values / 60
decl = np.radians(23.44) * np.sin(np.radians(360 / 365 * (doy - 81)))
B = np.radians(360 / 365 * (doy - 81))
eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)     # 분
tst = frac_h * 60 + 4 * (lon - 135) + eot                            # KST 표준자오선 135E
ha = np.radians(tst / 4 - 180)
la = np.radians(lat)
elev = np.degrees(np.arcsin(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)))
cos_omega = -np.tan(la) * np.tan(decl)
omega = np.degrees(np.arccos(np.clip(cos_omega, -1, 1)))
daylen = 2 * omega / 15
sol = pd.DataFrame({"ts15": h15, "solar_elevation": elev, "is_daylight": (elev > 0).astype(int),
                    "day_length_h": daylen})
# ---- 4) 요금 시간대 구조 (한전 산업용 계절·부하시간대 '구분'만; 금액 미사용)
mo, hr = h15.month.values, h15.hour.values
season = np.where(np.isin(mo, [6, 7, 8]), "summer",
                  np.where(np.isin(mo, [11, 12, 1, 2]), "winter", "spring_fall"))
load_zone = np.full(len(h15), "mid", dtype=object)
light = (hr < 9) | (hr >= 23)
load_zone[light] = "light"
peak_summer = np.isin(hr, [11, 12, 13, 14, 15, 16, 17]) & (season == "summer")
peak_sf = np.isin(hr, [11, 12, 13, 14, 15, 16, 17]) & (season == "spring_fall")
peak_winter = (np.isin(hr, [10, 11, 12, 17, 18, 19, 20, 21, 22])) & (season == "winter")
load_zone[peak_summer | peak_sf | peak_winter] = "peak"
sol["tariff_season"] = season
sol["tariff_load_zone"] = load_zone
save_table(sol, "07N_external_solar_tariff_2021")

# ---- 5) 선택 관측소 시간별 기상 캐시(파생만 저장)
g = series[best.station_id].reset_index().rename(columns={"ts_h": "ts"})
g = g.sort_values("ts")
g["ext_temp_diff1h"] = g.ext_temp.diff()
g["ext_temp_roll24h"] = g.ext_temp.rolling(24, min_periods=6).mean()
g["ext_dew_spread"] = g.ext_temp - g.ext_dewpoint
# 이슬점 기반 체감(습구근사, Stull 2011) — 추가 정보가 아니라 파생 변환임을 명시
g["ext_heat_index_proxy"] = g.ext_temp + 0.33 * (6.11 * np.exp(
    17.27 * g.ext_dewpoint / (237.3 + g.ext_dewpoint))) - 4.0
g["ext_pressure_diff3h"] = g.ext_pressure.diff(3)
# 전일 일최고/최저 (예측시점에 확정된 정보)
dd = g.set_index("ts").ext_temp.resample("D").agg(["max", "min"]).shift(1)
dd.columns = ["ext_prev_day_tmax", "ext_prev_day_tmin"]
g = g.merge(dd.reset_index().rename(columns={"ts": "date"}).assign(
    date=lambda x: x.date.dt.normalize()), left_on=g.ts.dt.normalize(), right_on="date", how="left")
g = g.drop(columns=["key_0", "date"], errors="ignore")
save_table(g, "07N_external_weather_hourly")

# 시간 결측은 '마지막 관측 유지(ffill)' 로 메운다. 예측시점에 실제로 가능한 처리이며
# 미래 관측을 쓰지 않는다. 최대 6시간까지만 유지하고 그 이상은 결측으로 남긴다.
EXTC = ["ext_temp", "ext_dewpoint", "ext_pressure", "ext_wind", "ext_visibility", "ext_ceiling",
        "ext_temp_diff1h", "ext_temp_roll24h", "ext_dew_spread", "ext_heat_index_proxy",
        "ext_pressure_diff3h", "ext_prev_day_tmax", "ext_prev_day_tmin"]
grid = pd.DataFrame({"ts": pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31 23:00", freq="h")})
gg = grid.merge(g[["ts"] + EXTC], on="ts", how="left")
gap_before = gg[EXTC].isna().mean().mean()
gg[EXTC] = gg[EXTC].ffill(limit=6)
save_table(pd.DataFrame([dict(hourly_rows_observed=int(g.ts.between(grid.ts.min(), grid.ts.max()).sum()),
                             hourly_grid_rows=len(grid),
                             missing_share_before_ffill=float(gap_before),
                             missing_share_after_ffill=float(gg[EXTC].isna().mean().mean()),
                             ffill_limit_hours=6,
                             note="ffill 은 과거 관측만 사용. 6시간 초과 공백은 결측 유지")]),
           "07N_external_weather_coverage")
save_table(gg, "07N_external_weather_hourly_filled")

# ---- 6) 출처·라이선스 문서표
doc = pd.DataFrame([
    dict(dataset="Korean public holidays", source="python holidays 0.83 (KR)",
         url="https://pypi.org/project/holidays/", retrieved="2026-09-28", license="MIT",
         time_range="2020-2022 규칙", variables="is_holiday, holiday_name, is_workday, is_bridge_day, "
         "days_to_next_nonworkday, nonwork_run_len",
         available_at_prediction_time="예 — 달력은 사전 확정",
         preprocessing="일자 단위 생성 후 15분 계열에 날짜 조인", join_key="date",
         missingness="없음", note="공식 관보 기준 규칙. 임시공휴일 반영 여부는 패키지 버전에 의존"),
    dict(dataset="NOAA NCEI ISD global-hourly", source="NOAA National Centers for Environmental Information",
         url=ISD_URL.format(y=YEAR, s=best.station_id), retrieved="2026-09-28",
         license="미국 정부 공개자료(제한 없음). 재배포 가능하나 본 저장소는 원본 미커밋",
         time_range="2021-01-01 ~ 2021-12-31 (UTC, KST 변환)",
         variables="ext_temp, ext_dewpoint, ext_pressure, ext_wind, ext_visibility, ext_ceiling 및 파생",
         available_at_prediction_time="예(관측값). 예보는 사용하지 않음",
         preprocessing="고정폭 필드 파싱, 결측코드 제거, 시간 평균, KST(+9) 변환",
         join_key="ts(시간)", missingness="표 07N_external_weather_hourly 의 결측률 참조",
         note=f"KAMP 위치 미문서화 → 기온 상관으로 {best.label} 추정(MAE {best.mae_temp:.2f}도)"),
    dict(dataset="Solar geometry", source="천문 계산(다운로드 없음)", url="-", retrieved="2026-09-28",
         license="해당 없음", time_range="2021", variables="solar_elevation, is_daylight, day_length_h",
         available_at_prediction_time="예 — 결정론적 계산",
         preprocessing="적위/시간각/균시차 근사식, 위경도는 추정 관측소 좌표",
         join_key="ts15", missingness="없음", note="근사식 오차 수 분 수준"),
    dict(dataset="KEPCO industrial tariff time structure",
         source="한국전력 산업용 계절/부하시간대 구분(공개 요금표 구조)",
         url="https://cyber.kepco.co.kr/ckepco/front/jsp/CY/E/E/CYEEHP00201.jsp",
         retrieved="2026-09-28", license="공개 요금표의 시간대 구분만 사용",
         time_range="2021 기준 구조", variables="tariff_season, tariff_load_zone(light/mid/peak)",
         available_at_prediction_time="예 — 사전 공표",
         preprocessing="월/시간 규칙으로 범주 생성. 금액(KRW) 미사용",
         join_key="ts15", missingness="없음",
         note="계약종별(갑/을, 고압 구분) 미확인 → 구분 구조만 사용하고 요금 절감액 주장 금지"),
])
save_table(doc, "07N_external_data_sources")
print(doc[["dataset", "license", "available_at_prediction_time"]].to_string(index=False))
