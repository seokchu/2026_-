"""01. File inventory + data dictionary + cross-file structural audit."""
import sys, hashlib, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from common import DATA, ROOT, save_table, load_mold, load_press, load_power, segment

rows = []
for k, p in DATA.items():
    h = hashlib.md5(p.read_bytes()).hexdigest()[:12]
    rows.append({"key": k, "path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "md5_12": h})
inv = pd.DataFrame(rows)
save_table(inv, "01_file_inventory")

# ---------- data dictionary ----------
dd = []

def add(ds, col, meaning, unit, src, note):
    dd.append(dict(dataset=ds, column=col, meaning=meaning, unit=unit, evidence=src, note=note))

mold = load_mold("mold_lab_cn7")
for c in mold.columns:
    if c == "idx":
        add("injection_molding", c, "행 번호 = 원본 통합파일의 shot 순번(추정)", "-", "값이 단조증가, cn7 0~1210 / rg3 1211~2392 로 연속", "temporal proxy. 절대 시각 없음")
    elif c == "PassOrFail":
        add("injection_molding", c, "품질 판정 0=양품 1=불량", "binary", "값 {0,1}", "unlabeled 파일에는 없음")
    else:
        add("injection_molding", c, "사출 공정 변수 (컬럼명 기반, 문서 없음)", "z-score (무차원)", "파일별 mean=0 std=1", "원단위 불명 → unknown/needs verification")

wd = pd.read_excel(DATA["weld_xlsx"], sheet_name="data set")
for _, r in wd.iterrows():
    add("robot_welding", str(r["Data"]), str(r["항목 설명"]), str(r["수집 범위"]), "xlsx 'data set' 시트(공식 스키마)", "")
add("robot_welding", "defect", "해당 일자·불량유형의 불량 '개수'(행 단위 아님)", "count", "xlsx 'result' 시트", "행 단위 라벨 아님 → 지도학습 불가")
add("robot_welding", "defect type", "1=파임불량 2=용접부족 3=크랙발생", "code", "xlsx 'result' 시트 Unnamed:6", "")

for c, m, u, ev in [("TimeStamp", "샘플 시각 (ms)", "datetime", "0.1s 등간격 + 갭"),
                    ("AI0_Vibration", "진동 채널 0", "unknown (전압 추정)", "단위 문서 없음"),
                    ("AI1_Vibration", "진동 채널 1", "unknown", "단위 문서 없음"),
                    ("AI2_Current", "전류", "unknown (A 아님, ±270 범위)", "정상파일은 주기 1.8s 정현파"),
                    ("Equipment_state", "0=정상 1=이상", "binary", "파일 단위로 상수")]:
    add("hydraulic_press", c, m, u, ev, "")

pw = load_power()
pmean = {"날짜": ("측정 일자", "YYYYMMDD"), "시간": ("시(0-23)", "hour"),
         "15분": ("해당 시의 15분 시점 전력", "unknown(kW 추정)"), "30분": ("30분 시점 전력", "unknown"),
         "45분": ("45분 시점 전력", "unknown"), "60분": ("60분 시점 전력", "unknown"),
         "평균": ("4개 값의 평균(정수 반올림)", "unknown"), "생산량": ("시간당 생산량", "unknown(ea 추정)"),
         "기온": ("기온", "degC"), "풍속": ("풍속", "m/s"), "습도": ("상대습도", "%"),
         "강수량": ("강수량", "mm"), "전기요금(계절)": ("계절별 요금 단가", "KRW/kWh 추정"),
         "day": ("요일 코드 1-7", "code"), "d": ("일", "day"), "m": ("월", "month"),
         "공장인원": ("공장 인원(정규화 추정, float·max 48.4)", "unknown"),
         "인건비": ("인건비 계수 {1.0,1.5}", "unknown")}
for c, (m, u) in pmean.items():
    add("power", c, m, u, "컬럼명 + 값 분포", "문서 없음 → 단위 unknown" if "unknown" in u else "")
save_table(pd.DataFrame(dd), "01_data_dictionary")

# ---------- structural audit ----------
aud = []
for k in ["mold_lab_cn7", "mold_lab_rg3", "mold_unlab_cn7", "mold_unlab_rg3"]:
    d = load_mold(k)
    F = [c for c in d.columns if c not in ("idx", "PassOrFail")]
    g = d.groupby(F, sort=False).ngroup()
    gl = d.assign(g=g).groupby("g").size()
    aud.append(dict(dataset=k, n_rows=len(d), n_cols=d.shape[1],
                    idx_min=d.idx.min(), idx_max=d.idx.max(), idx_monotonic=bool(d.idx.is_monotonic_increasing),
                    dup_full_rows=int(d.duplicated(subset=F).sum()),
                    unique_feature_rows=int(g.nunique()),
                    modal_group_size=int(gl.mode().iloc[0]),
                    const_cols=";".join([c for c in F if d[c].nunique() == 1]),
                    n_fail=int(d.PassOrFail.sum()) if "PassOrFail" in d else -1))
n, o = load_press()
for nm, d in [("press_normal", n), ("press_outlier", o)]:
    s = segment(d)
    aud.append(dict(dataset=nm, n_rows=len(d), n_cols=d.shape[1], idx_min=0, idx_max=len(d) - 1,
                    idx_monotonic=True, dup_full_rows=int(d.duplicated().sum()),
                    unique_feature_rows=int(s.nunique()), modal_group_size=int(s.value_counts().max()),
                    const_cols="Equipment_state", n_fail=int(d.Equipment_state.sum())))
save_table(pd.DataFrame(aud), "01_structural_audit")

# ---------- press acquisition-grid forensics (leakage check) ----------
Q = 1.1920929  # 24-bit ADC LSB pattern found in outlier_data.csv
grid = []
for nm, d in [("press_normal", n), ("press_outlier", o)]:
    r = d.AI2_Current.values / Q
    dev = np.abs(r - np.round(r))
    dec = d[["AI0_Vibration", "AI1_Vibration"]].astype(str).map(lambda s: len(s.split(".")[-1]))
    grid.append(dict(file=nm, n=len(d), current_grid_mean_dev=dev.mean(), current_grid_max_dev=dev.max(),
                     on_grid=bool(dev.max() < 1e-3),
                     vib_decimals_mode=int(pd.concat([dec.AI0_Vibration, dec.AI1_Vibration]).mode().iloc[0]),
                     current_nuniq_ratio=d.AI2_Current.nunique() / len(d)))
save_table(pd.DataFrame(grid), "01_press_acquisition_grid_leak")
print(json.dumps({"seed_note": "audit is deterministic, no sampling"}, ensure_ascii=False))
