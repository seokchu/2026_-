"""Shared helpers for the K-AI manufacturing EDA. Seed / paths / IO in one place."""
from pathlib import Path
import numpy as np, pandas as pd

SEED = 20260926
ROOT = Path(__file__).resolve().parent.parent
ANA = ROOT / "analysis"
FIG = ANA / "figures"
TAB = ANA / "tables"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

DATA = {
    "mold_lab_cn7": ROOT / "1. 사출성형기 AI 데이터셋/moldset_labeled_cn7.csv",
    "mold_lab_rg3": ROOT / "1. 사출성형기 AI 데이터셋/moldset_labeled_rg3.csv",
    "mold_unlab_cn7": ROOT / "1. 사출성형기 AI 데이터셋/moldset_unlabeled_cn7.csv",
    "mold_unlab_rg3": ROOT / "1. 사출성형기 AI 데이터셋/moldset_unlabeled_rg3.csv",
    "weld_xlsx": ROOT / "2. 용접기 AI 데이터셋/Welding Data Set_01.xlsx",
    "weld_scaled": ROOT / "2. 용접기 AI 데이터셋/scaled_data.csv",
    "press_normal": ROOT / "3. 소성가공 예지보전 AI 데이터셋/press_data_normal.csv",
    "press_outlier": ROOT / "3. 소성가공 예지보전 AI 데이터셋/outlier_data.csv",
    "power": ROOT / "5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv",
}

def save_table(df, name, index=False):
    out = TAB / f"{name}.csv"
    df.to_csv(out, index=index, encoding="utf-8-sig")
    print(f"[table] {out.relative_to(ROOT)}  shape={df.shape}")
    return out

def fig_path(name):
    return FIG / f"{name}.png"

def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "font.family": "AppleGothic", "axes.unicode_minus": False})
    return plt

def load_mold(key):
    d = pd.read_csv(DATA[key])
    return d.rename(columns={d.columns[0]: "idx"})

def load_press():
    n = pd.read_csv(DATA["press_normal"], index_col=0, parse_dates=["TimeStamp"])
    o = pd.read_csv(DATA["press_outlier"], index_col=0, parse_dates=["TimeStamp"])
    n["source"], o["source"] = "normal", "outlier"
    return n, o

def segment(d, gap_s=0.15):
    """Burst segmentation: the press logger writes <=50-sample bursts split by gaps."""
    dt = d["TimeStamp"].diff().dt.total_seconds()
    return (dt.isna() | (dt > gap_s)).cumsum()

def load_power(fix_hour=True):
    p = pd.read_csv(DATA["power"], encoding="utf-8-sig")
    p["date"] = pd.to_datetime(p["날짜"], format="%Y%m%d")
    if fix_hour:
        # 2021-07-13 / 2021-07-15: the 시간 column was overwritten with power-like
        # values (24 rows each, still in row order). Rebuild hour from row order.
        bad = p.loc[p["시간"] > 23, "날짜"].unique()
        p["hour_corrupt"] = p["날짜"].isin(bad)
        p["시간"] = p.groupby("날짜").cumcount()
    p["ts"] = p["date"] + pd.to_timedelta(p["시간"], unit="h")
    return p.sort_values("ts").reset_index(drop=True)

def power_long(p):
    """Expand the 4 in-hour readings into a 15-min demand series."""
    q = p.melt(id_vars=["ts", "날짜", "시간", "생산량", "기온", "풍속", "습도", "강수량",
                        "전기요금(계절)", "day", "d", "m", "공장인원", "인건비", "hour_corrupt"],
               value_vars=["15분", "30분", "45분", "60분"], var_name="q", value_name="kw")
    off = {"15분": 15, "30분": 30, "45분": 45, "60분": 60}
    q["ts15"] = q["ts"] + pd.to_timedelta(q["q"].map(off), unit="m")
    return q.sort_values("ts15").reset_index(drop=True)
