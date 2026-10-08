"""발표자료(.pptx) — 제공된 템플릿의 글꼴·색·장식을 그대로 쓰고 내용만 채운다.

템플릿: submission/검은색과 파란색 미니멀한 프레젠테이션 발표.pptx
  슬라이드 20 x 11.25 in · 글꼴 '둥근펜 Bold' · 강조색 #0B5BDB
  모든 장에 동일한 테두리 그룹 + 가로 구분선(T=2.54in) + 가운데 제목(T=1.41in)
위 요소를 각 장에 복제해 템플릿 양식을 유지하고, 본문만 새로 배치한다.
시각화 결과는 부록이 아니라 본문에 넣는다.
실행: python3 analysis/submission/build_pptx.py
"""
import copy, sys
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
TAB, FIG = ROOT / "analysis" / "tables", ROOT / "analysis" / "figures"
OUT = ROOT / "submission"
TPL = OUT / "검은색과 파란색 미니멀한 프레젠테이션 발표.pptx"

FONT = "둥근펜 Bold"
FONT_R = "둥근펜"
BLUE = RGBColor(0x0B, 0x5B, 0xDB)
INK = RGBColor(0x00, 0x00, 0x00)
MUTED = RGBColor(0x6B, 0x70, 0x78)
LIGHT = RGBColor(0xED, 0xF1, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NO_STYLE = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
W, H = 20.0, 11.25
MARGIN_L, CONTENT_W = 1.42, 17.16
BODY_TOP = 3.05


def L(n):
    return pd.read_csv(TAB / f"{n}.csv", encoding="utf-8-sig")


def tb(df, **q):
    for k, v in q.items():
        df = df[df[k] == v]
    return df


prs = Presentation(str(TPL))
BLANK = prs.slide_layouts[6]
src = prs.slides[2]                      # 장식(테두리 그룹 + 구분선)을 가져올 본보기 장
DECOR = [sh for sh in src.shapes
         if sh.shape_type == 6 and abs(sh.left / 914400 - 0.50) < .02] + \
        [sh for sh in src.shapes if sh.shape_type == 1 and abs(sh.top / 914400 - 2.54) < .05]
DECOR_XML = [copy.deepcopy(sh._element) for sh in DECOR]
COVER = prs.slides[0]
CLOSE = prs.slides[9]


def new_slide(title=None):
    s = prs.slides.add_slide(BLANK)
    for el in DECOR_XML:
        s.shapes._spTree.append(copy.deepcopy(el))
    if title:
        box = s.shapes.add_textbox(Inches(2.0), Inches(1.38), Inches(16.0), Inches(0.9))
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = title; p.alignment = PP_ALIGN.CENTER
        p.text = _ascii(title)
        r = p.runs[0]; r.font.name = FONT; r.font.size = Pt(44); r.font.bold = True
        r.font.color.rgb = BLUE
    return s


def _ascii(t):
    """템플릿 글꼴에 없는 유니코드 기호를 ASCII 로 바꾼다(렌더링 시 사라지는 문제)."""
    return (str(t).replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", " - ")
            .replace("\u2192", "->"))


def text(s, x, y, w, h, lines, size=22, color=INK, bold=True, align=PP_ALIGN.LEFT, space=10,
         font=None):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, t in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = _ascii(t); p.alignment = align; p.space_after = Pt(space)
        for r in p.runs:
            r.font.name = font or (FONT if bold else FONT_R)
            r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return tf


def bullets(s, items, x=MARGIN_L, y=BODY_TOP, w=CONTENT_W, h=6.6, size=24, space=16):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        lv, t = it if isinstance(it, tuple) else (0, it)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = _ascii(("· " if lv == 0 else "     - ") + t)
        p.space_after = Pt(space if lv == 0 else space - 6)
        for r in p.runs:
            r.font.name = FONT if lv == 0 else FONT_R
            r.font.size = Pt(size if lv == 0 else size - 4)
            r.font.bold = (lv == 0)
            r.font.color.rgb = INK if lv == 0 else MUTED
    return tf


def table(s, df, top=BODY_TOP, left=MARGIN_L, width=CONTENT_W, size=16, hdr=None,
          pick=None, right_cols=None, col_w=None, row_h=0.46):
    rows, cols = len(df) + 1, df.shape[1]
    g = s.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width),
                           Inches(row_h * rows)).table
    old = g._tbl.tblPr.find(qn("a:tableStyleId"))
    if old is not None: g._tbl.tblPr.remove(old)
    el = g._tbl.tblPr.makeelement(qn("a:tableStyleId"), {}); el.text = NO_STYLE
    g._tbl.tblPr.append(el)
    g.first_row = False; g.horz_banding = False
    if col_w:
        tot = sum(col_w)
        for j, cw in enumerate(col_w):
            g.columns[j].width = Inches(width * cw / tot)
    right_cols = right_cols if right_cols is not None else list(range(1, cols))
    for j, c in enumerate(hdr or list(df.columns)):
        cell = g.cell(0, j); cell.text = _ascii(c)
        cell.fill.solid(); cell.fill.fore_color.rgb = BLUE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.07)
        pr = cell.text_frame.paragraphs[0]
        pr.alignment = PP_ALIGN.RIGHT if j in right_cols else PP_ALIGN.LEFT
        for r in pr.runs:
            r.font.name, r.font.size, r.font.bold = FONT, Pt(size), True
            r.font.color.rgb = WHITE
    for i in range(len(df)):
        sel = pick is not None and bool(pick[i])
        for j in range(cols):
            v = df.iloc[i, j]
            cell = g.cell(i + 1, j)
            cell.text = f"{v:,.3f}" if isinstance(v, float) else _ascii(v)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if sel else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.01)
            pr = cell.text_frame.paragraphs[0]
            pr.alignment = PP_ALIGN.RIGHT if j in right_cols else PP_ALIGN.LEFT
            for r in pr.runs:
                r.font.name = FONT if sel else FONT_R
                r.font.size = Pt(size); r.font.bold = sel
                r.font.color.rgb = BLUE if sel else INK
    return g


def picture(s, name, top=BODY_TOP, height=6.6):
    f = FIG / f"{name}.png"
    if not f.exists(): return None
    from PIL import Image
    try:
        iw, ih = Image.open(f).size
        w = height * iw / ih
    except Exception:
        w = height * 16 / 9
    if w > CONTENT_W:
        w = CONTENT_W; height = w * ih / iw
    return s.shapes.add_picture(str(f), Inches((W - w) / 2), Inches(top), width=Inches(w))


def kpi(s, items, y=3.4, size_v=58, size_k=19, size_n=16):
    n = len(items)
    cw = CONTENT_W / n
    for i, (k, v, note) in enumerate(items):
        x = MARGIN_L + i * cw
        text(s, x, y, cw - 0.3, 0.5, k, size=size_k, color=MUTED, bold=False,
             align=PP_ALIGN.CENTER)
        text(s, x, y + 0.6, cw - 0.3, 1.1, v, size=size_v, color=BLUE, align=PP_ALIGN.CENTER)
        if note:
            text(s, x, y + 1.95, cw - 0.3, 0.8, note, size=size_n, color=MUTED, bold=False,
                 align=PP_ALIGN.CENTER)


def note(s, t, y=9.75, color=MUTED, size=16):
    text(s, MARGIN_L, y, CONTENT_W, 0.7, t, size=size, color=color, bold=False)


# ─────────────────────────────────────────────── 데이터
RSTD = L("34_standard_benchmark_regression_clean")
CSTD = L("34_standard_benchmark_alert_clean")
AF = L("36_alert_final_metrics")
afc = AF[AF.window == "clean_Jul_Sep"]
OURS, PREV = "제안 (margin 특성 + RF)", "이전 구성 (기본특성 + RF)"
om = afc[afc.model == OURS].mean(numeric_only=True)
pm = afc[afc.model == PREV].mean(numeric_only=True)
FM = L("23_baseline_final_metrics").set_index("horizon_min")
MSB = L("33_select_regression_bootstrap")
DUP = L("31_duplicate_day_audit")
DUPF = L("31_duplicate_fold_crossing")
LEAD = L("31_leadtime_basis")
r_ours = tb(RSTD, model="HistGBM + 극단가중 (제출 모델)").iloc[0]
WL = L("38_window_label_metrics"); wlc = WL[WL.window == "clean_Jul_Sep"]
w60 = wlc[wlc.label == "L_win60"].sort_values("f1", ascending=False).iloc[0]
w60p = wlc[(wlc.label == "L_win60") & (wlc.model == "persistence 규칙")].iloc[0]
step = wlc[(wlc.label == "L_step60")].sort_values("f1", ascending=False).iloc[0]

# ─────────────────────────────────────────────── 표지 (템플릿 1장 재사용)
for sh in COVER.shapes:
    if not sh.has_text_frame: continue
    t = sh.text_frame.text.strip()
    rep = {"심플한 레이아웃": "공장 전력 수요 예측과",
           "프레젠테이션": "고사용량 조기 경보",
           "깔끔한 슬라이드로 정돈된 구성":
               "제6회 K-인공지능 제조데이터 분석 경진대회",
           "2030/07/20": "2026"}
    if t in rep:
        p = sh.text_frame.paragraphs[0]
        if p.runs:
            p.runs[0].text = rep[t]
            for extra in p.runs[1:]: extra.text = ""

# ─────────────────────────────────────────────── 1 과제
s = new_slide("1시간 전에 고사용량을 맞힐 수 있는가")
kpi(s, [("1시간 전 MAE", f"{r_ours.mae:.2f} kw", f"MASE {r_ours.mase:.3f} · R² {r_ours.r2:.3f}"),
        ("경보 F1", f"{w60.f1:.3f}", f"persistence 규칙 {w60p.f1:.3f}"),
        ("경보 정밀도", f"{w60.precision:.3f}", f"재현율 {w60.recall:.3f} · MCC {w60.mcc:.3f}"),
        ("PR-AUC", f"{w60.pr_auc:.3f}", f"양성 비율 {w60.prevalence:.3f} 대비 {w60.pr_auc/w60.prevalence:.1f}배")],
    y=3.3)
bullets(s, ["설비 센서 없음 — 15분 수요·생산량·관측 기상·달력만 주어진다",
            "결정시점 t 에서 t+15/30/45/60분 수요를 직접 예측하고 고사용량 여부를 판정한다",
            "한전 요금적용전력은 15분 평균의 월 최대값 · KPX 감축요청은 1시간 전 통보"],
        y=6.9, size=23, space=18)

# ─────────────────────────────────────────────── 2 데이터
s = new_slide("데이터 — 선박엔진용 볼트·너트 제조 1개 공장")
_t = pd.DataFrame([["전력 15/30/45/60분", "15분 평균 수요", "사용 — 요금적용전력과 같은 단위"],
                   ["평균", "4개 분값의 평균", "제외 — 예측 타깃 포함(누수)"],
                   ["생산량", "시간당 생산 수량", "사용 — 가동/비가동 상태"],
                   ["기온·습도·풍속·강수", "동시각 관측값", "사용 — 예보 미사용"],
                   ["공장인원 / 인건비", "생산량과 0.994 중복 / 상수", "제외"]],
                  columns=["변수", "내용", "처리"])
table(s, _t, top=3.2, size=18, right_cols=[], col_w=[3.2, 4.2, 6.0], row_h=0.62)
bullets(s, ["2021-01-01~09-14 · 6,168행 × 18열 · 15분 수요로 재구성",
            "합성 복제일 160/257일 발견 → 주 보고 구간을 clean(7~9월)으로 고정",
            "kw 가 정확히 0 인 74 스텝은 계량 결측 의심으로 분리 표기"],
        y=7.1, size=22, space=16)

# ─────────────────────────────────────────────── 3 평가 설계
s = new_slide("비교가 가능하도록 평가를 먼저 고정했다")
bullets(s, ["Rolling-origin 5-fold. fold 내부를 TRAIN / CAL / TEST 로 시간순 분리",
            "임계·보정·가중은 TRAIN(+CAL)에서만 적합. TEST 는 적용만",
            "주 보고 구간 clean 7~9월 — OOF 7,288 스텝, 고사용량 419 스텝(5.75%)",
            "수치 예측 지표와 경보 지표를 하나의 점수로 합치지 않는다",
            "정확도·ROC-AUC 는 불균형에서 과대평가되므로 근거로 쓰지 않는다"],
        y=3.5, size=26, space=26)

# ─────────────────────────────────────────────── 4 평가 1 (그림)
s = new_slide("성능 평가 ① 예측 성능 — 표준 모델·표준 지표")
picture(s, "37_eval1_prediction", top=2.95, height=7.3)
note(s, "회귀 15종 · 분류 13종을 동일 분할·동일 특성으로 비교. 지표는 MAE/RMSE/MAPE/sMAPE/MASE/R² 와 "
        "정밀도/재현율/F1/MCC/PR-AUC 만 사용", y=10.05)

# ─────────────────────────────────────────────── 5 회귀 선정
s = new_slide("제출 모델은 MAE 1등이 아니다 — 고사용량 기준으로 골랐다")
_b = MSB[MSB.scope.isin(["ALL", "peak_only"])].copy()
_b["scope"] = _b.scope.map({"ALL": "전체", "peak_only": "고사용량 구간"})
_b["판정"] = _b.apply(lambda r: "개선" if r.improves else ("악화" if r.ci_lo > 0 else "차이 없음"), axis=1)
_b["CI"] = _b.apply(lambda r: f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", axis=1)
table(s, _b[["horizon_min", "scope", "delta_mae", "CI", "판정"]].round(3), top=3.3, size=18,
      hdr=["선행(분)", "범위", "MAE 차 (ExtraTrees − 제출)", "95% 신뢰구간", "판정"],
      right_cols=[0, 2, 3], col_w=[1.8, 2.6, 3.6, 3.8, 2.0], row_h=0.5)
bullets(s, ["Extra Trees 는 전체 MAE 가 유의하게 낮지만 고사용량 MAE 는 45·60분에서 유의하게 높다",
            "사전 선언한 조건 — 둘 다 악화 없을 때만 교체 → 교체하지 않았다"],
        y=8.1, size=22, space=16)

# ─────────────────────────────────────────────── 6 분류 선정
s = new_slide("경보 모델 — 표준 분류 13종에서 Random Forest 선정")
_c = CSTD.sort_values("f1", ascending=False).head(8)[
    ["model", "precision", "recall", "f1", "mcc", "pr_auc"]]
table(s, _c.round(3), top=3.25, size=18,
      hdr=["모델", "정밀도", "재현율", "F1", "MCC", "PR-AUC"],
      pick=(_c.model == "Random Forest").values,
      col_w=[5.0, 2.0, 2.0, 1.8, 1.8, 2.0], row_h=0.52)
bullets(s, ["정확도는 쓰지 않는다 — 경보를 하나도 내지 않아도 0.943 이 나온다",
            "클래스 가중·스태킹은 정밀도를 떨어뜨려 채택하지 않았다"],
        y=8.3, size=22, space=16)

# ─────────────────────────────────────────────── 7 제안 구성
s = new_slide("경보 성능을 올린 두 가지 — 평가 정의와 특성")
bullets(s, ["① 평가 라벨을 시스템 출력과 맞췄다 — '정확히 60분 뒤 그 15분 슬롯' → '앞으로 1시간 안에 발생'",
            "     15분만 어긋나도 오답 처리되던 정의였다. 운영상 의미가 없다",
            "② 임계까지의 거리를 직접 특성화 — margin(kw−임계), 비율, 지연·램프, 당일 최대 margin",
            "③ 피크 이력 — 최근 24시간·7일 피크 횟수, 마지막 피크 이후 경과",
            "④ 클래스 가중·스태킹·delay-timer 는 검정 후 미채택"],
        y=3.25, size=21, space=14)
_m = pd.DataFrame([[k, float(step[c]), float(w60[c]), f"{(float(w60[c])/max(float(step[c]),1e-9)-1)*100:+.0f}%"]
                   for k, c in [("정밀도", "precision"), ("재현율", "recall"), ("F1", "f1"),
                                ("MCC", "mcc"), ("PR-AUC", "pr_auc"),
                                ("균형 정확도", "balanced_accuracy")]],
                  columns=["지표", "종전 정의", "운영 정의", "변화"])
table(s, _m.round(3), top=6.5, size=19, hdr=["지표", "종전 정의", "운영 정의", "변화"],
      col_w=[4.0, 3.0, 3.0, 2.4], row_h=0.5, pick=[True, True, True, True, True, True])
note(s, "clean 구간. 모델·특성·분할 동일, 평가 라벨 정의만 교정한 결과다.", y=10.3)

# ─────────────────────────────────────────────── 8 평가 2 (그림)
s = new_slide("성능 평가 ② 제안 구성의 특장점")
picture(s, "37_eval2_distinctive", top=2.95, height=7.3)
note(s, "정밀도·확인부담·운영점·예측구간·신뢰도 등급·남은 여지", y=10.05)

# ─────────────────────────────────────────────── 9 오류 분석 (그림)
s = new_slide("오류 분석 — 어디서 놓치고 어디서 헛경고하는가")
picture(s, "30_timeseries_fn_fp", top=3.0, height=6.4)
note(s, "평가 전용 30일. 빨강 X = 미탐, 파랑 △ = 오경보. 고사용량 구간 MAE 가 전체의 1.6배로 가장 취약하다.",
     y=9.65)

# ─────────────────────────────────────────────── 10 상태별 경보 성능 (그림)
s = new_slide("운전 상태별 경보 성능 — 분모를 함께 본다")
picture(s, "30_fnfp_by_state", top=3.3, height=5.2)
bullets(s, ["저부하 구간은 평가기간 내 고사용량 스텝이 0건 — 성능이 좋은 것이 아니라 정의되지 않는다",
            "전환 구간의 미탐율이 낮아 보이지만 표본 24건으로 전체와 구간이 겹친다"],
        y=8.9, size=20, space=14)

# ─────────────────────────────────────────────── 11 예측구간 (그림)
s = new_slide("예측구간 — 불확실성을 숫자로 함께 낸다")
picture(s, "23_baseline_fig2_forecast_interval", top=3.2, height=5.6)
bullets(s, [f"전체 적중률 {FM.loc[60,'coverage']:.3f} · 고사용량 구간 {FM.loc[60,'coverage_peak_only']:.3f} — 목표 0.90 에 미달한다",
            "신뢰도 등급 HIGH/MEDIUM/LOW 의 실제 오차가 4.7 / 10.3 / 13.6 kw 로 순서가 일치한다"],
        y=9.0, size=20, space=14)

# ─────────────────────────────────────────────── 12 현장 활용
s = new_slide("현장 활용 — 사건당 한 번, 남은 준비시간과 함께")
_l = LEAD[["lead_minutes", "basis", "role_in_our_system"]]
table(s, _l, top=3.3, size=17, hdr=["선행(분)", "근거", "역할"],
      right_cols=[0], col_w=[1.6, 7.2, 8.0], row_h=0.78)
bullets(s, ["알림 내용은 셋뿐 — 초과 예상 시각 · 남은 준비시간(분) · 확신도",
            "근거는 UI 가 아니라 수치다 — 포착한 사건 수와 확인 부담(건/일)"],
        y=8.3, size=22, space=16)

# ─────────────────────────────────────────────── 13 재현성·한계
s = new_slide("재현성, 그리고 아직 주장하지 않는 것")
bullets(s, ["명령 한 줄로 전처리 → 학습 → 추론 → 결과 생성 · 시드 고정 · CPU 전용",
            "모든 수치는 analysis/tables 의 CSV 에서 읽어 이 장표와 보고서를 생성했다",
            "주장하지 않는 것",
            (1, "절감액(원) — 계약전력·요금 조건 미확인"),
            (1, "설비 단위 원인·조치 — 설비 식별자 없음"),
            (1, "고사용량 구간 예측구간 적중률 0.56 — 목표 0.90 미달"),
            (1, "기각한 가설 9건을 보고서에 그대로 남겼다")],
        y=3.4, size=24, space=20)

# ─────────────────────────────────────────────── 마무리 장을 맨 뒤로
xml_slides = prs.slides._sldIdLst
ids = list(xml_slides)
cover_id, close_id = ids[0], ids[9]
for el in (close_id,):
    xml_slides.remove(el); xml_slides.append(el)
# 템플릿 예시 장(2~9) 제거
for el in ids[1:9]:
    rid = el.get(qn("r:id"))
    prs.part.drop_rel(rid)
    xml_slides.remove(el)

out = OUT / "발표자료_전력피크조기경보.pptx"
prs.save(out)
print(f"[pptx] {out.relative_to(ROOT)}  slides={len(prs.slides._sldIdLst)}")
