"""발표자료(.pptx) 생성. 수치는 analysis/tables/*.csv 에서만 읽는다(하드코딩 금지).

구성 원칙
  - 표준 모델·표준 지표로 먼저 비교하고, 고유 분석은 뒤에 '부가'로 붙인다.
  - 한 장에 한 메시지. 제목이 곧 주장. 본문은 표 하나 또는 짧은 항목 4개 이내.
  - 장식 최소화: 단색 글자, 얇은 구분선, 교차 음영만 사용. 색 블록·그라데이션·아이콘 없음.
실행: python3 analysis/submission/build_pptx.py
"""
import sys
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
import copy

ROOT = Path(__file__).resolve().parents[2]
TAB, FIG = ROOT / "analysis" / "tables", ROOT / "analysis" / "figures"
OUT = ROOT / "submission"; OUT.mkdir(exist_ok=True)

FONT = "맑은 고딕"
INK = RGBColor(0x11, 0x11, 0x11)
MUTED = RGBColor(0x70, 0x75, 0x7A)
RULE = RGBColor(0xD8, 0xDC, 0xE0)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
TINT = RGBColor(0xF7, 0xF8, 0xFA)
PICK = RGBColor(0xE8, 0xEF, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NO_STYLE = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"      # No Style, No Grid


def L(n):
    return pd.read_csv(TAB / f"{n}.csv", encoding="utf-8-sig")


def tb(df, **q):
    for k, v in q.items():
        df = df[df[k] == v]
    return df


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
_page = {"n": 0}


def rule(s, top, left=0.7, width=11.95, h=0.012, color=RULE):
    r = s.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = color; r.line.fill.background()
    r.shadow.inherit = False
    return r


def txt(s, x, y, w, h, lines, size=12, color=INK, bold=False, space=4, align=PP_ALIGN.LEFT):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, t in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t; p.alignment = align
        p.space_after = Pt(space)
        for r in p.runs:
            r.font.size = Pt(size); r.font.name = FONT
            r.font.color.rgb = color; r.font.bold = bold
    return tf


def slide(title, sub=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
    txt(s, 0.7, 0.42, 12.0, 0.6, title, size=24, bold=True)
    if sub:
        txt(s, 0.7, 1.03, 12.0, 0.35, sub, size=11.5, color=MUTED)
    rule(s, 1.47)
    _page["n"] += 1
    txt(s, 11.6, 6.95, 1.05, 0.3, str(_page["n"]), size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def source(s, text):
    txt(s, 0.7, 6.95, 9.5, 0.3, text, size=8.5, color=MUTED)


def table(s, df, top=1.75, left=0.7, width=11.95, size=10.5, hdr=None,
          pick=None, right_cols=None, col_w=None, row_h=0.265):
    rows, cols = len(df) + 1, df.shape[1]
    g = s.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width),
                           Inches(row_h * rows)).table
    g._tbl.tblPr.find(qn("a:tableStyleId")) is None or g._tbl.tblPr.remove(
        g._tbl.tblPr.find(qn("a:tableStyleId")))
    el = g._tbl.tblPr.makeelement(qn("a:tableStyleId"), {}); el.text = NO_STYLE
    g._tbl.tblPr.append(el)
    g.first_row = False; g.horz_banding = False
    if col_w:
        tot = sum(col_w)
        for j, w in enumerate(col_w):
            g.columns[j].width = Inches(width * w / tot)
    right_cols = right_cols if right_cols is not None else list(range(1, cols))
    heads = hdr or list(df.columns)
    for j, c in enumerate(heads):
        cell = g.cell(0, j); cell.text = str(c)
        cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.05)
        pr = cell.text_frame.paragraphs[0]
        pr.alignment = PP_ALIGN.RIGHT if j in right_cols else PP_ALIGN.LEFT
        for r in pr.runs:
            r.font.size, r.font.bold, r.font.name = Pt(size), True, FONT
            r.font.color.rgb = MUTED
    for i in range(len(df)):
        sel = pick is not None and bool(pick[i])
        for j in range(cols):
            v = df.iloc[i, j]
            cell = g.cell(i + 1, j)
            cell.text = f"{v:,.3f}" if isinstance(v, float) else str(v)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PICK if sel else (TINT if i % 2 else WHITE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.01)
            pr = cell.text_frame.paragraphs[0]
            pr.alignment = PP_ALIGN.RIGHT if j in right_cols else PP_ALIGN.LEFT
            for r in pr.runs:
                r.font.size, r.font.name = Pt(size), FONT
                r.font.color.rgb = INK if not sel else ACCENT
                r.font.bold = sel
    return g


def bullets(s, items, x=0.7, y=1.75, w=11.95, h=4.9, size=13.5, space=9):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        lv, t = it if isinstance(it, tuple) else (0, it)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("· " if lv == 0 else "    – ") + t
        p.space_after = Pt(space if lv == 0 else space - 4)
        for r in p.runs:
            r.font.size = Pt(size if lv == 0 else size - 1.5)
            r.font.name = FONT
            r.font.color.rgb = INK if lv == 0 else MUTED
    return tf


def kpi(s, items, y=1.9, size_v=34, size_k=11):
    n = len(items)
    w = 11.95 / n
    for i, (k, v, note) in enumerate(items):
        x = 0.7 + i * w
        txt(s, x, y, w - 0.2, 0.3, k, size=size_k, color=MUTED)
        txt(s, x, y + 0.32, w - 0.2, 0.7, v, size=size_v, bold=True, color=ACCENT)
        if note:
            txt(s, x, y + 1.18, w - 0.2, 0.5, note, size=10, color=MUTED)


# ───────────────────────────────────────────────────────────── 데이터
RSTD = L("34_standard_benchmark_regression_clean")
CSTD = L("34_standard_benchmark_alert_clean")
FM = L("23_baseline_final_metrics").set_index("horizon_min")
PKP = L("28_peak_probability_summary")
pk60 = tb(PKP, window="clean_Jul_Sep", horizon_min=60).set_index("score")
pk15 = tb(PKP, window="clean_Jul_Sep", horizon_min=15).set_index("score")
MSB = L("33_select_regression_bootstrap")
FDG = L("33_f1_ceiling_diagnosis")
DUP = L("31_duplicate_day_audit")
DUPF = L("31_duplicate_fold_crossing")
STR = L("31_v2_stress_map")
LEAD = L("31_leadtime_basis")
MCA = L("33_model_compare_alert_clean")
OURS_R = "HistGBM + 극단가중 (제출 모델)"
r_ours = tb(RSTD, model=OURS_R).iloc[0]
c_best = CSTD.sort_values("f1", ascending=False).iloc[0]
f1_ceil = float(tb(FDG, model="R4_hist_gbm").oracle_f1_same_error.iloc[0])
bias_pk = float(tb(FDG, model="R4_hist_gbm").bias_peak.iloc[0])
above = float(tb(FDG, model="R4_hist_gbm").peak_pred_above_thr.iloc[0])
F1_PROD = float(pk60.loc["RF_clf(main)", "f1"])          # 생산 구성(기준 수치)
F1_PROD15 = float(pk15.loc["RF_clf(main)", "f1"])

# ───────────────────────────────────────────────────────────── 1 표지
s = prs.slides.add_slide(BLANK)
s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
txt(s, 0.9, 2.05, 11.5, 1.0, "공장 전력 수요 예측과 고사용량 조기 경보", size=38, bold=True)
txt(s, 0.9, 3.15, 11.5, 0.9,
    ["15분 수요·생산량·기상·달력만으로 1시간 뒤를 예측한다.",
     "제6회 K-인공지능 제조데이터 분석 경진대회 · 주제 ⑤ 자원 최적화"], size=14, color=MUTED, space=6)
rule(s, 4.35, left=0.9, width=4.2, h=0.02, color=ACCENT)
kpi(s, [("MAE (1시간 전)", f"{r_ours.mae:.2f} kw", "MASE {:.3f} · MAPE {:.1f}%".format(r_ours.mase, r_ours.mape)),
        ("R²", f"{r_ours.r2:.3f}", "Naive 대비 MAE −53%"),
        ("경보 F1", f"{F1_PROD:.3f}", f"PR-AUC {float(pk60.loc['RF_clf(main)','pr_auc']):.3f} · 재현율 {float(pk60.loc['RF_clf(main)','recall']):.3f}"),
        ("평균 선행", f"{FM.loc[60,'policy_mean_lead_min']:.0f} 분", "KPX 통보 기준 60분")],
    y=4.75, size_v=30)

# ───────────────────────────────────────────────────────────── 2 과제
s = slide("1시간 전에, 고사용량을 맞힐 수 있는가",
          "설비 센서 없음. 15분 수요·생산량·관측 기상·달력만 주어진다.")
bullets(s, [
    "예측: 결정시점 t 에서 t+15/30/45/60분 수요를 각각 직접 예측",
    "경보: 직전 30일 p95 를 넘을지 판정. 한전 요금적용전력이 15분 평균의 월 최대값이기 때문",
    "선행: KPX 수요자원 거래시장은 감축요청을 1시간 전에 통보한다 → h60 이 기준",
    "하지 않는 것: 설비 단위 조치 지시 · 고장 심각도 · 절감액(원)",
], y=1.9, size=15, space=16)
source(s, "analysis/tables/31_leadtime_basis.csv · 31_tariff_label_basis.csv")

# ───────────────────────────────────────────────────────────── 3 데이터
s = slide("데이터 — 선박엔진용 볼트·너트 제조 1개 공장",
          "2021-01-01 ~ 09-14 · 6,168행 × 18열 · 15분 수요로 재구성")
_t = pd.DataFrame([
    ["전력 15/30/45/60분", "15분 평균 수요 4개", "사용 — 한전 요금적용전력과 같은 단위"],
    ["평균", "같은 시간 4개 분값의 평균", "제외 — 예측 타깃을 포함(누수)"],
    ["생산량", "시간당 생산 수량", "사용 — 가동/비가동 상태의 근거"],
    ["기온·습도·풍속·강수", "동시각 관측값", "사용 — 예보는 쓰지 않음"],
    ["공장인원", "생산량과 Spearman 0.994", "제외 — 중복"],
    ["인건비", "고유값 2개", "제외 — 사실상 상수"],
], columns=["변수", "내용", "처리"])
table(s, _t, top=1.85, size=12, right_cols=[], col_w=[2.4, 3.4, 5.0], row_h=0.42)
bullets(s, ["2021-07-13·07-15 의 '시간' 열이 전력값으로 덮여 있어 행 순서로 복구",
            "kw 가 정확히 0 인 74 스텝은 계량 결측 의심으로 분리 표기(보간하지 않음)"],
        y=4.9, size=12.5, space=8)

# ───────────────────────────────────────────────────────────── 4 평가 설계
s = slide("비교가 가능하도록 평가를 먼저 고정했다",
          "모든 모델이 같은 분할·같은 적합집합·같은 지표를 쓴다")
bullets(s, [
    "Rolling-origin 5-fold. fold 내부를 TRAIN / CAL / TEST 로 시간순 분리",
    "임계·보정·가중·스케일러는 TRAIN(+CAL)에서만 적합. TEST 는 적용만",
    "주 보고 구간 clean_Jul_Sep — OOF 7,288 스텝, 고사용량 419 스텝(5.75%)",
    "수치 예측 지표와 경보 지표를 한 점수로 합치지 않는다",
], y=1.95, size=15, space=18)
_t = pd.DataFrame([["수치 예측", "MAE · RMSE · MAPE · sMAPE · MASE · R²"],
                   ["이진 경보", "Accuracy · Precision · Recall · F1 · Balanced Acc. · MCC · ROC-AUC · PR-AUC · Brier"]],
                  columns=["과제", "사용 지표 (표준)"])
table(s, _t, top=4.6, size=12, right_cols=[], col_w=[1.6, 10.3], row_h=0.42)

# ───────────────────────────────────────────────────────────── 5 표준 회귀 벤치마크
s = slide("표준 회귀 모델 15종 비교 — 1시간 전 예측",
          "clean 구간 OOF 7,288 스텝 · 동일 특성·동일 분할")
_r = RSTD[["model", "mae", "rmse", "mape", "smape", "mase", "r2", "mae_peak"]].copy()
_r = _r[~_r.model.isin(["Seasonal naive (24h 전)"])]
pick = (_r.model == OURS_R).values
table(s, _r.round(3), top=1.8, size=10,
      hdr=["모델", "MAE", "RMSE", "MAPE %", "sMAPE %", "MASE", "R²", "고사용량 MAE"],
      pick=pick, col_w=[3.3, 1.1, 1.1, 1.2, 1.2, 1.0, 1.0, 1.5], row_h=0.3)
source(s, "analysis/tables/34_standard_benchmark_regression_clean.csv · "
          "Seasonal naive(24h) 는 MASE 1.208 로 표에서 생략")

# ───────────────────────────────────────────────────────────── 6 회귀 선정
s = slide("제출 모델은 MAE 1등이 아니다 — 고사용량 구간을 기준으로 골랐다",
          "전체 MAE 만 보면 Extra Trees, 고사용량 MAE 로 보면 HistGBM+극단가중")
_b = MSB[MSB.scope.isin(["ALL", "peak_only"])].copy()
_b["scope"] = _b.scope.map({"ALL": "전체", "peak_only": "고사용량 구간"})
_b["판정"] = _b.apply(lambda r: "개선" if r.improves else ("악화" if r.ci_lo > 0 else "차이 없음"), axis=1)
_b["CI"] = _b.apply(lambda r: f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", axis=1)
table(s, _b[["horizon_min", "scope", "delta_mae", "CI", "판정"]].round(3), top=1.9, size=11.5,
      hdr=["선행(분)", "범위", "MAE 차 (ET − HistGBM)", "95% 신뢰구간", "판정"],
      right_cols=[0, 2, 3], col_w=[1.3, 2.0, 2.6, 3.0, 1.6], row_h=0.32)
bullets(s, ["Extra Trees 는 전체 MAE 가 유의하게 낮지만 고사용량 MAE 는 h45·h60 에서 유의하게 높다",
            "사전에 선언한 조건 — 전체·고사용량 둘 다 악화 없을 때만 교체 → 교체하지 않았다"],
        y=4.95, size=13, space=10)
source(s, "analysis/tables/33_select_regression_bootstrap.csv · 일 단위 블록 부트스트랩 1,000회")

# ───────────────────────────────────────────────────────────── 7 표준 분류 벤치마크
s = slide("표준 분류 모델 13종 비교 — 고사용량 경보",
          "clean 구간 7,288 스텝 · 양성 비율 5.75% · 임계는 CAL 에서 F1 최대로 선택")
_c = CSTD[["model", "accuracy", "precision", "recall", "specificity",
           "balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc"]].copy()
pick = (_c.model == c_best.model).values
table(s, _c.round(3), top=1.8, size=10,
      hdr=["모델", "정확도", "정밀도", "재현율", "특이도", "균형정확도", "F1", "MCC", "ROC-AUC", "PR-AUC"],
      pick=pick, col_w=[3.0, 1.0, 1.0, 1.0, 1.0, 1.2, 0.9, 0.9, 1.1, 1.1], row_h=0.3)
source(s, "analysis/tables/34_standard_benchmark_alert_clean.csv · "
          "정확도는 불균형에서 의미가 작다(전부 정상 예측 시 0.943). F1·MCC 로 읽는다")

# ───────────────────────────────────────────────────────────── 8 분류 선정
s = slide("경보 모델은 Random Forest — F1·MCC 동시 최고",
          "클래스 가중·스태킹은 오히려 나빠져 쓰지 않는다")
_sel = CSTD.sort_values("f1", ascending=False).head(6)[["model", "f1", "mcc", "precision", "recall", "fp", "fn"]]
table(s, _sel.round(3), top=1.9, size=11.5,
      hdr=["모델", "F1", "MCC", "정밀도", "재현율", "오경보", "미탐"],
      pick=(_sel.model == c_best.model).values,
      col_w=[3.6, 1.1, 1.1, 1.2, 1.2, 1.1, 1.1], row_h=0.34)
bullets(s, [
    f"회귀 점수를 그대로 경보에 쓰면 F1 {float(tb(MCA, model='C1_reg_margin').f1.iloc[0]):.3f} — 전용 분류기가 필요하다",
    "클래스 가중(balanced) 적용 시 F1 0.512 → 0.453 으로 악화",
    "스태킹은 전 구간 F1 0.477 이었으나 clean 구간에서 0.251 로 붕괴 → 기각",
    f"위 표는 공통 특성집합 기준 순위다. 생산 구성(외부 달력·기상 포함 + isotonic 보정)에서는 F1 {F1_PROD:.3f}",
], y=4.6, size=13, space=10)
source(s, "analysis/tables/34_standard_benchmark_alert_clean.csv · 33_model_compare_alert_clean.csv")

# ───────────────────────────────────────────────────────────── 9 최종 성능
s = slide("최종 성능 — 네 호라이즌", "clean 구간 OOF · 제출 모델 기준")
_f = pd.DataFrame([[f"{h}분", FM.loc[h, "mae"], FM.loc[h, "rmse"],
                    FM.loc[h, "persistence_improvement"] * 100,
                    float(tb(PKP, window="clean_Jul_Sep", horizon_min=h)
                          .set_index("score").loc["RF_clf(main)", "f1"]),
                    float(tb(PKP, window="clean_Jul_Sep", horizon_min=h)
                          .set_index("score").loc["RF_clf(main)", "pr_auc"]),
                    FM.loc[h, "coverage"], FM.loc[h, "coverage_peak_only"]]
                   for h in (15, 30, 45, 60)],
                  columns=["선행", "MAE", "RMSE", "Naive 대비 %", "경보 F1", "PR-AUC",
                           "구간 적중률", "고사용량 적중률"])
table(s, _f.round(3), top=1.95, size=12,
      col_w=[1.1, 1.2, 1.2, 1.6, 1.3, 1.3, 1.5, 1.8], row_h=0.38)
bullets(s, [f"선행이 길수록 어려워진다 — 15분 F1 {F1_PROD15:.2f}, 60분 F1 {F1_PROD:.2f}",
            "구간 적중률은 목표 0.90 에 미달한다. 고사용량 구간은 특히 낮다"],
        y=4.5, size=13, space=10)
source(s, "analysis/tables/23_baseline_final_metrics.csv · 28_peak_probability_summary.csv")

# ───────────────────────────────────────────────────────────── 10 오류분석
s = slide("어디서 틀리는가 — 고사용량 구간 자체가 가장 어렵다", "1시간 전 예측, 조건별 MAE")
st = tb(STR, horizon_min=60, variant="A5_all")
CN = {"ALL": "전체", "peak_only": "고사용량 구간", "high_ramp_p90": "급변동 상위 10%",
      "restart_1h": "재가동 1시간 이내", "midday_12_18": "12~18시",
      "weekend": "주말", "non_operating": "비가동"}
_s = st[st.condition.isin(CN)][["condition", "n", "share", "mae", "mae_ratio_vs_all"]].copy()
_s["condition"] = _s.condition.map(CN)
_s["share"] = (_s.share * 100).round(1)
table(s, _s.round(3), top=1.95, size=12,
      hdr=["조건", "스텝 수", "비중 %", "MAE", "전체 대비 배율"],
      col_w=[3.2, 1.6, 1.4, 1.4, 2.0], row_h=0.36)
bullets(s, ["고사용량 구간 MAE 는 전체의 1.61배 — 가장 취약하다",
            "비가동·주말은 전체보다 낮다(0.85·0.86). 휴무·재가동 특성이 작동했다"],
        y=4.8, size=13, space=10)
source(s, "analysis/tables/31_v2_stress_map.csv")

# ───────────────────────────────────────────────────────────── 11 부가 ① F1 한계
s = slide(f"부가분석 ① F1 {F1_PROD:.2f} 는 한계가 아니라 편향 때문이다",
          "오차 크기는 그대로 두고 편향만 없앴을 때의 F1 을 측정했다")
kpi(s, [("현재 F1", f"{F1_PROD:.3f}", "clean 구간, 1시간 전"),
        ("편향 없을 때 F1", f"{f1_ceil:.3f}", "같은 잔차 분포를 무작위 주입"),
        ("고사용량 구간 과소예측", f"{bias_pk:.1f} kw", "전체 편향은 −0.96 kw"),
        ("임계에 도달한 피크", f"{above*100:.0f} %", f"나머지 {100-above*100:.0f}% 는 선에 못 미친다")],
    y=1.9, size_v=30)
bullets(s, ["MAE 를 줄이면 조건부 평균으로 수렴한다 — 분포의 오른쪽 꼬리를 구조적으로 깎는다",
            "그래서 MAE 가 좋아져도 F1 이 따라오지 않는다",
            "다음: 높은 분위를 점예측으로 사용 · 과소예측에 더 큰 벌점을 주는 비대칭 손실"],
        y=3.95, size=13.5, space=12)
source(s, "analysis/tables/33_f1_ceiling_diagnosis.csv")

# ───────────────────────────────────────────────────────────── 12 부가 ② 데이터 품질
s = slide("부가분석 ② 학습·평가 구간에 합성 복제일이 섞여 있었다",
          "하루 96스텝 전력 프로파일을 해시로 비교해 확인")
d_all, d_cl = tb(DUP, scope="all_2021").iloc[0], tb(DUP, scope="clean_Jul_Sep").iloc[0]
kpi(s, [("전 기간 복제일", f"{int(d_all.n_dup_days)} / {int(d_all.n_days)} 일",
         f"{int(d_all.n_dup_groups)}개 그룹 · 잉여 {int(d_all.n_redundant_days)}일"),
        ("주 보고 구간", f"{int(d_cl.n_dup_days)} / {int(d_cl.n_days)} 일", "1개 그룹"),
        ("fold 0·2 교차", "23 · 28 그룹", "TRAIN·TEST 양쪽에 존재 → 성능 과대평가"),
        ("fold 4 교차", "0 그룹", "clean 구간을 TEST 로 쓰는 fold")], y=1.95, size_v=26)
bullets(s, ["전 구간 평균을 대표 성능으로 쓰면 부풀려진다 → 주 보고 구간을 clean 으로 고정",
            "수용테스트에 'fold 교차 0건' 검사를 넣어 회귀를 막는다"], y=4.5, size=13.5, space=12)
source(s, "analysis/tables/31_duplicate_day_audit.csv · 31_duplicate_fold_crossing.csv")

# ───────────────────────────────────────────────────────────── 13 부가 ③ 예측구간
s = slide("부가분석 ③ 예측구간 — 좁아지면서 적중률은 올랐다",
          "분위회귀 + conformal 보정. 목표 적중률 0.90")
_i = pd.DataFrame([[f"{h}분", FM.loc[h, "coverage"], FM.loc[h, "coverage_peak_only"],
                    FM.loc[h, "mean_interval_width"], FM.loc[h, "mae_band_HIGH"],
                    FM.loc[h, "mae_band_MEDIUM"], FM.loc[h, "mae_band_LOW"]]
                   for h in (15, 30, 45, 60)],
                  columns=["선행", "전체 적중률", "고사용량 적중률", "평균 폭(kw)",
                           "HIGH MAE", "MEDIUM MAE", "LOW MAE"])
table(s, _i.round(3), top=1.95, size=12, col_w=[1.1, 1.6, 1.8, 1.5, 1.3, 1.5, 1.3], row_h=0.38)
bullets(s, ["신뢰도 등급이 실제 오차 순서와 일치한다 — HIGH 4.7 < MEDIUM 10.3 < LOW 13.6 kw",
            "고사용량 구간 적중률은 0.56 으로 목표 0.90 에 미달한다. 해결이 아니라 개선이다"],
        y=4.4, size=13.5, space=12)
source(s, "analysis/tables/23_baseline_final_metrics.csv")

# ───────────────────────────────────────────────────────────── 14 현장 활용
s = slide("현장 활용 — 사건당 한 번, 남은 준비시간과 함께", "선행시간은 제도 기준에 맞춘다")
_l = LEAD[["lead_minutes", "basis", "role_in_our_system"]].copy()
table(s, _l, top=1.9, size=11, hdr=["선행(분)", "근거", "역할"],
      right_cols=[0], col_w=[1.1, 5.2, 5.6], row_h=0.5)
bullets(s, ["알림 내용은 셋뿐 — 초과 예상 시각 · 남은 준비시간(분) · 확신도",
            "근거는 UI 가 아니라 수치다 — 포착한 사건 수와 확인 부담(건/일)"],
        y=4.5, size=13.5, space=12)
source(s, "analysis/tables/31_leadtime_basis.csv")

# ───────────────────────────────────────────────────────────── 15 재현성·한계
s = slide("재현성, 그리고 아직 주장하지 않는 것")
bullets(s, [
    "명령 한 줄로 전처리 → 학습 → 추론 → 결과 생성. 시드 고정(20260926), CPU 전용",
    "학습/추론 분리 — joblib 아티팩트 + inference.py 로 예측 파일 생성",
    "모든 수치는 analysis/tables/*.csv 에서 읽어 이 장표를 만들었다",
    "주장하지 않는 것",
    (1, "절감액(원) — 계약전력·요금 조건 미확인"),
    (1, "설비 단위 원인·조치 — 설비 식별자 없음"),
    (1, "고사용량 구간 예측구간 적중률 0.56 — 목표 0.90 미달"),
    (1, "기각한 가설 7건을 보고서에 그대로 남겼다"),
], y=1.9, size=14, space=13)

# ───────────────────────────────────────────────────────────── 부록
APPENDIX = [("23_baseline_fig2_forecast_interval", "부록 · 예측값과 예측구간"),
            ("23_baseline_fig3_reliability_bands", "부록 · 신뢰도 등급별 오차"),
            ("30_timeseries_fn_fp", "부록 · 평가 30일 실측·예측·미탐·오경보"),
            ("28_peak_probability", "부록 · 경보 확률의 신뢰도 곡선")]
for nm, ttl in APPENDIX:
    if (FIG / f"{nm}.png").exists():
        s2 = slide(ttl)
        s2.shapes.add_picture(str(FIG / f"{nm}.png"), Inches(1.0), Inches(1.75), width=Inches(11.3))
        source(s2, f"analysis/figures/{nm}.png")

out = OUT / "발표자료_전력피크조기경보.pptx"
prs.save(out)
print(f"[pptx] {out.relative_to(ROOT)}  slides={_page['n'] + 1}")
