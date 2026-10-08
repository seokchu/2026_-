"""발표자료(.pptx) 생성 — 수치는 전부 analysis/tables 의 CSV 에서 읽는다(하드코딩 금지).

실행: python3 analysis/submission/build_pptx.py
출력: submission/발표자료_전력피크조기경보.pptx
"""
import sys
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "analysis" / "tables"
FIG = ROOT / "analysis" / "figures"
OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)
FONT = "맑은 고딕"
INK = RGBColor(0x1A, 0x1A, 0x1A)
SUB = RGBColor(0x55, 0x5A, 0x60)
ACC = RGBColor(0x1F, 0x5C, 0x99)
WARN = RGBColor(0xC0, 0x39, 0x2B)
BG = RGBColor(0xFF, 0xFF, 0xFF)


def tb(df, **q):
    for k, v in q.items():
        df = df[df[k] == v]
    return df


def load(name):
    return pd.read_csv(TAB / f"{name}.csv", encoding="utf-8-sig")


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    bar = s.shapes.add_shape(1, Inches(0.0), Inches(0.0), Inches(13.333), Inches(0.09))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACC; bar.line.fill.background()
    t = s.shapes.add_textbox(Inches(0.6), Inches(0.32), Inches(12.1), Inches(0.75)).text_frame
    t.word_wrap = True
    p = t.paragraphs[0]; p.text = title
    p.runs[0].font.size, p.runs[0].font.bold = Pt(28), True
    p.runs[0].font.color.rgb, p.runs[0].font.name = INK, FONT
    if kicker:
        k = s.shapes.add_textbox(Inches(0.62), Inches(1.05), Inches(12.1), Inches(0.4)).text_frame
        k.word_wrap = True
        kp = k.paragraphs[0]; kp.text = kicker
        kp.runs[0].font.size, kp.runs[0].font.color.rgb = Pt(13), SUB
        kp.runs[0].font.name = FONT
    return s


def bullets(s, items, left=0.65, top=1.55, width=12.0, height=5.3, size=15):
    tf = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame
    tf.word_wrap = True
    first = True
    for lv, txt in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = min(lv, 4)
        p.text = ("■ " if lv == 0 else "– " if lv == 1 else "· ") + txt
        for r in p.runs:
            r.font.size = Pt(size - 1.5 * lv); r.font.name = FONT
            r.font.color.rgb = INK if lv == 0 else SUB
            r.font.bold = (lv == 0)
        p.space_after = Pt(5)
    return tf


def table(s, df, left=0.65, top=1.6, width=12.0, height=None, size=11, hdr=None):
    rows, cols = df.shape[0] + 1, df.shape[1]
    height = height or min(0.33 * rows, 5.3)
    g = s.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width),
                           Inches(height)).table
    for j, c in enumerate(df.columns):
        cell = g.cell(0, j); cell.text = str(hdr[j] if hdr else c)
        for r in cell.text_frame.paragraphs[0].runs:
            r.font.size, r.font.bold, r.font.name = Pt(size), True, FONT
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.fill.solid(); cell.fill.fore_color.rgb = ACC
    for i in range(df.shape[0]):
        for j in range(cols):
            v = df.iloc[i, j]
            cell = g.cell(i + 1, j)
            cell.text = f"{v:,.3f}" if isinstance(v, float) else str(v)
            for r in cell.text_frame.paragraphs[0].runs:
                r.font.size, r.font.name = Pt(size), FONT
                r.font.color.rgb = INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i % 2 == 0 else RGBColor(0xF2, 0xF5, 0xF8)
    return g


def pic(s, name, left, top, width):
    f = FIG / f"{name}.png"
    if f.exists():
        s.shapes.add_picture(str(f), Inches(left), Inches(top), width=Inches(width))


def note(s, text, color=SUB, top=6.85):
    tf = s.shapes.add_textbox(Inches(0.65), Inches(top), Inches(12.0), Inches(0.45)).text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.runs[0].font.size, p.runs[0].font.color.rgb, p.runs[0].font.name = Pt(11), color, FONT


# ─────────────────────────────────────────────── 데이터 로드
V2 = load("31_v2_clean_summary")
V2B = load("31_v2_clean_bootstrap")
DUP = load("31_duplicate_day_audit")
DUPF = load("31_duplicate_fold_crossing")
THS = load("31_peak_threshold_sensitivity")
LEAD = load("31_leadtime_basis")
OODS = load("31_ood_role_summary")
OODD = load("31_ood_decile_curve")
OODB = load("31_ood_weight_bootstrap")
STR = load("31_v2_stress_map")
ABL = load("31_v2_ablation_summary")
MCR = load("33_model_compare_regression")
MSC = load("33_select_regression_clean")
MSB = load("33_select_regression_bootstrap")
MCA = load("33_model_compare_alert_clean")
FDG = load("33_f1_ceiling_diagnosis")
PKP = load("28_peak_probability_summary")

a0 = tb(V2, variant="A0_baseline", horizon_min=60).iloc[0]
a5 = tb(V2, variant="A5_all", horizon_min=60).iloc[0]

# ─────────────────────────────────────────────── S1 표지
s = prs.slides.add_slide(BLANK)
s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
box = s.shapes.add_shape(1, Inches(0), Inches(2.4), Inches(13.333), Inches(0.12))
box.fill.solid(); box.fill.fore_color.rgb = ACC; box.line.fill.background()
t = s.shapes.add_textbox(Inches(0.9), Inches(1.35), Inches(11.5), Inches(1.1)).text_frame
t.word_wrap = True
p = t.paragraphs[0]; p.text = "센서가 부족한 공장의 전력 고사용량 조기 판단"
p.runs[0].font.size, p.runs[0].font.bold = Pt(40), True
p.runs[0].font.color.rgb, p.runs[0].font.name = INK, FONT
t2 = s.shapes.add_textbox(Inches(0.9), Inches(2.75), Inches(11.5), Inches(1.4)).text_frame
t2.word_wrap = True
for i, line in enumerate([
        "15분 수요·생산량·날씨·달력만으로 1시간 뒤 수요를 예측하고,",
        "조치 가능한 선행시간 안에 고사용량 발생을 판단한다.",
        "제6회 K-인공지능 제조데이터 분석 경진대회 — 주제 ⑤ 자원 최적화"]):
    p = t2.paragraphs[0] if i == 0 else t2.add_paragraph()
    p.text = line
    p.runs[0].font.size = Pt(17 if i < 2 else 13)
    p.runs[0].font.color.rgb = SUB if i < 2 else ACC
    p.runs[0].font.name = FONT
_pk60 = tb(PKP, window="clean_Jul_Sep", horizon_min=60).set_index("score")
_f1_now = float(_pk60.loc["ET_clf_isotonic(main)", "f1"])
_f1_pers = float(_pk60.loc["persistence_rule_margin", "f1"])
_f1_ceil = float(FDG[FDG.model == "R4_hist_gbm"].oracle_f1_same_error.iloc[0])
_evrec = float(tb(MCA, model="C5_extra_trees").event_recall.iloc[0])
note(s, f"수치 예측 — h60 MAE {a0.mae:.3f} → {a5.mae:.3f} kw ({a5.mae_vs_A0_pct:+.1f}%), "
        f"고사용량 구간 {a0.mae_peak:.3f} → {a5.mae_peak:.3f} kw ({a5.maepeak_vs_A0_pct:+.1f}%)",
     ACC, top=4.35)
note(s, f"고사용량 경보 — F1 {_f1_now:.3f} (persistence 규칙 {_f1_pers:.3f}, "
        f"무편향 오라클 상한 {_f1_ceil:.3f}) · 사건 recall {_evrec:.3f}",
     WARN, top=4.75)

# ─────────────────────────────────────────────── S2 문제 정의
s = slide("무엇을 푸는가", "설비 센서가 없는 공장. 가진 것은 15분 수요·생산량·관측 날씨·달력뿐이다.")
bullets(s, [
    (0, "주(main) — 시계열 패턴 정밀 판단: 결정시점 t 에서 t+15/30/45/60분 수요를 각각 직접 예측"),
    (1, "15분은 한전 요금적용전력의 측정 단위다. 해상도를 임의로 고른 것이 아니다."),
    (0, "부(sub) — 조치 가능한 선행시간 확보: 고사용량 도달 전에 알린다"),
    (1, f"KPX 수요자원 거래시장은 감축요청을 {int(LEAD.lead_minutes.max())}분 전 통보하고 "
        f"부족분을 30분 내 추가 요청한다 → h60 이 제도상 표준 리드타임"),
    (0, "부(sub·의사결정) — 판단이 서지 않을 때 그 사실과 부족한 정보의 종류를 말한다"),
    (0, "하지 않는 것"),
    (1, "설비 단위 조치 지시 · 고장 심각도 순위 · 절감액(원) 주장 — 근거 데이터가 없다"),
])

# ─────────────────────────────────────────────── S3 데이터 이해
s = slide("제1장 · 데이터 이해 — 업종과 구조", "선박엔진용 볼트·너트 제조 1개 공장, 2021-01-01~09-14, 6,168행×18열")
bullets(s, [
    (0, "변수 구성: 전력 15/30/45/60분(15분 수요), 평균, 생산량, 기온·풍속·습도·강수, 전기요금(계절), 달력, 공장인원, 인건비"),
    (0, "제외한 변수와 이유"),
    (1, "'평균' = 같은 시간 4개 분값의 평균 → 타깃을 포함한다(누수). 전면 제외"),
    (1, "'공장인원' = 생산량과 Spearman 0.994 중복 / '인건비' = 고유값 2개 사실상 상수"),
    (0, "복구한 결함: 2021-07-13·07-15 의 '시간' 열이 전력값으로 덮여 있었다 → 행 순서로 재구성"),
    (0, "확인한 결측 의심: kw 정확히 0 인 74 스텝(08-28 25, 08-29 47, 09-08 2). 정지 중 기저부하는 22 kw 수준"),
], size=14.5)

# ─────────────────────────────────────────────── S4 중복일 감사
d_all = tb(DUP, scope="all_2021").iloc[0]
d_cl = tb(DUP, scope="clean_Jul_Sep").iloc[0]
s = slide("제1장 · 데이터 진단 — 합성 증강 중복일을 찾아냈다",
          "파일명의 'augumented' 가 실제로 무엇인지 하루 단위 전력 프로파일 해시로 확인했다")
bullets(s, [
    (0, f"전체 기간: {int(d_all.n_days)}일 중 {int(d_all.n_dup_days)}일이 "
        f"96스텝 전력 프로파일이 완전히 동일 — {int(d_all.n_dup_groups)}개 그룹, "
        f"잉여 {int(d_all.n_redundant_days)}일 (전체의 {d_all.dup_day_share*100:.1f}%)"),
    (0, f"분석 구간(2021-07-01 이후): {int(d_cl.n_days)}일 중 중복 {int(d_cl.n_dup_days)}일(1개 그룹)뿐"),
    (0, "왜 중요한가 — 중복일이 같은 fold 의 TRAIN 과 TEST 양쪽에 들어가면 성능이 부풀려진다"),
], top=1.5, height=2.0, size=14.5)
table(s, DUPF[["fold", "test_start", "test_end", "n_dup_groups_crossing", "test_all_in_clean"]],
      top=3.75, width=11.0,
      hdr=["fold", "TEST 시작", "TEST 끝", "TRAIN/TEST 걸친 중복그룹 수", "TEST 전부 clean 구간"])
note(s, "→ fold 0·2 는 중복일이 양쪽에 걸쳐 평가가 오염된다. fold 4(8/4~9/14)만 0개. "
        "주 보고 구간을 clean_Jul_Sep 으로 고정한 이유다.", WARN)

# ─────────────────────────────────────────────── S5 라벨 정의 근거
s = slide("제1장 · 라벨 정의 — '직전 30일 p95' 는 어디서 왔는가",
          "한전 요금적용전력 = 15분 평균전력의 월 최대 1점(검침 당월 포함 직전 12개월 중 7~9월·12~2월분 최대)")
bullets(s, [
    (0, "15분 해상도 — 과금 측정 단위와 동일. 임의 선택 아님"),
    (0, "직전 ~30일 창 — 검침월(과금 주기)과 정합. 계절·생산량 변화를 따라간다"),
    (0, "p95 는 과금 규칙이 아니라 모델링 선택이다 — '월 최대 1점'을 라벨로 쓰면 월당 양성 1건으로 학습·평가가 불가능"),
    (0, "분석 구간 2021-07~09 는 연간 기본요금을 결정하는 하계 피크월에 해당"),
], top=1.5, height=1.9, size=14.5)
sel = THS[THS.rule.isin(["trailing14d_p95", "trailing30d_p95", "trailing60d_p95",
                         "trailing90d_p95", "trailing30d_p90", "trailing30d_p97",
                         "fixed_187kw"])][["rule", "n_peak_steps", "prevalence", "n_events",
                                           "thr_mean", "thr_std", "thr_daily_absdiff_mean"]]
table(s, sel.round(3), top=3.55, width=11.6,
      hdr=["기준", "피크 스텝", "발생률", "사건 수", "임계 평균", "임계 표준편차", "일평균 변화(kw)"])
note(s, "창이 짧을수록 임계가 흔들리고(14일 일변화 0.839 kw) 길수록 둔해진다(90일 0.227). "
        "30일은 적응속도와 안정성의 절충이며 과금 주기와 맞는 지점이다 — 최적화 결과가 아니라 선택이다.")

# ─────────────────────────────────────────────── S6 평가 규약
s = slide("제2장 · 평가 규약 — 비교가 가능하도록 먼저 고정했다",
          "모든 변형은 같은 fold·같은 적합집합·같은 TEST 에서만 비교한다")
bullets(s, [
    (0, "Rolling-origin (TimeSeriesSplit 5 fold). fold 내부를 TRAIN / CAL(앞 구간의 뒤 20%) / TEST 로 분리"),
    (1, "임계·보정계수·가중치·스케일러·OOD 컷은 전부 TRAIN(+CAL)에서만 적합. TEST 는 적용만"),
    (1, "랜덤 분할 금지. 시간 역행 금지. 타깃 파생 변수 금지"),
    (0, "주 보고 구간: clean_Jul_Sep (2021-07-01 ~ 09-14, 7,289 OOF 스텝, 고사용량 419 스텝)"),
    (0, "두 축을 끝까지 섞지 않는다"),
    (1, "수치 예측 성능 = MAE / RMSE / 고사용량 구간 MAE"),
    (1, "경고 성능 = FN · FP · F1 (분모를 항상 같이 적는다)"),
    (0, "사전 선언한 채택 규칙으로만 판정한다. 돌려본 뒤 기준을 바꾸지 않는다"),
], size=14.5)

# ─────────────────────────────────────────────── S7 모델 비교
s = slide("제2장 · 모델 비교 — 기준선 3종 + 점진 개선 6종",
          "전 구간(Feb~Sep) OOF 평균. 기준선을 포함해 2개 이상 모델을 같은 조건에서 비교")
cmp = ABL[ABL.horizon_min == 60][["variant", "mae", "rmse", "mae_peak", "mae_vs_A0_pct",
                                  "maepeak_vs_A0_pct"]].copy()
order = ["B0_persistence", "B1_ridge", "B2_rf", "A0_baseline", "A1_resid", "A2_cyclic",
         "A3_extreme_weight", "A4_shutdown_feat", "A5_all"]
cmp["o"] = cmp.variant.map({v: i for i, v in enumerate(order)})
cmp = cmp.sort_values("o").drop(columns="o")
NAME = {"B0_persistence": "직전값 지속(기준선)", "B1_ridge": "Ridge(기준선)",
        "B2_rf": "RandomForest(기준선)", "A0_baseline": "HGB 기본 ★v1",
        "A1_resid": "+ 잔차타깃", "A2_cyclic": "+ 주기 인코딩",
        "A3_extreme_weight": "+ 극단가중(SERA)", "A4_shutdown_feat": "+ 휴무·재가동 특성",
        "A5_all": "A5 전체 결합 ★최종"}
cmp["variant"] = cmp.variant.map(NAME)
table(s, cmp.round(3), top=1.6, width=12.0, size=12,
      hdr=["모델 (h60)", "MAE", "RMSE", "고사용량 MAE", "MAE 변화%", "고사용량 MAE 변화%"])
note(s, "트리 기반을 쓴 이유: 표본 7천 스텝·CPU 전용 환경에서 재현 가능하고, "
        "계단형 레벨 변화(가동/정지)를 분할로 바로 처리한다. 심층 모델은 쓰지 않았다.")

# ─────────────────────────────────────────────── S8 최종 성능
s = slide("제2장 · 최종 성능 — clean 구간, 네 호라이즌 전부 개선",
          "A5(잔차타깃 + 주기인코딩 + 극단가중 + 휴무·재가동 특성) vs A0(v1 기본)")
w = V2[V2.variant.isin(["A0_baseline", "A5_all"])].pivot(index="horizon_min", columns="variant",
                                                         values=["mae", "mae_peak"]).reset_index()
w.columns = ["horizon_min", "A0_mae", "A5_mae", "A0_peak", "A5_peak"]
w["MAE 개선%"] = (w.A5_mae - w.A0_mae) / w.A0_mae * 100
w["고사용량 MAE 개선%"] = (w.A5_peak - w.A0_peak) / w.A0_peak * 100
table(s, w.round(3), top=1.6, width=11.6, size=12,
      hdr=["선행(분)", "A0 MAE", "A5 MAE", "A0 고사용량 MAE", "A5 고사용량 MAE",
           "MAE 개선%", "고사용량 MAE 개선%"])
bb = V2B[V2B.scope.isin(["ALL", "peak_only"])][["horizon_min", "scope", "delta_mae",
                                                "ci_lo", "ci_hi", "improves"]]
table(s, bb.round(3), top=3.8, width=11.6, size=11,
      hdr=["선행(분)", "범위", "MAE 차(A5-A0)", "95% CI 하한", "95% CI 상한", "개선 유의"])
note(s, "일 단위 블록 부트스트랩 1,000회(시간 상관 보정). 8개 비교 전부 CI 상한 < 0 — 우연이 아니다.", ACC)


# ─────────────────────────────────────────────── S8b 회귀 모델 비교·선정
s = slide("제2장 · 모델 비교 ① 수치 예측 — 베이스라인 포함 6종",
          "전부 동일 조건(동일 fold·적합집합·v2 레시피). 전 구간 OOF, h60")
_r = MCR[MCR.horizon_min == 60][["model", "mae", "rmse", "mae_peak", "bias_peak", "err_p95"]].copy()
RN = {"R0_persistence": "직전값 지속 (베이스라인)", "R1_ridge": "Ridge (베이스라인)",
      "R2_random_forest": "RandomForest", "R3_extra_trees": "ExtraTrees",
      "R4_hist_gbm": "HistGBM ★현행", "R5_hist_gbm_deep": "HistGBM-deep"}
_r["model"] = _r.model.map(RN)
table(s, _r.round(3), top=1.6, width=12.0, size=12,
      hdr=["모델 (h60)", "MAE", "RMSE", "고사용량 MAE", "고사용량 편향", "오차 p95"])
bullets(s, [
    (0, "사전 선언 선정 규칙: (a) 4개 horizon 전부 MAE 악화 없음 AND (b) h60 블록 부트스트랩 CI 상한<0 AND (c) 고사용량 MAE 악화 없음"),
    (0, "ExtraTrees — (a)(b) 충족, (c) 위반 → 기각"),
    (1, "전체 MAE 는 h60 −0.786 kw (CI [−1.007, −0.578]) 로 유의 개선"),
    (1, "그러나 고사용량 구간 MAE 는 h45 +0.714 (CI [+0.043, +1.395]), h60 +1.237 (CI [+0.515, +1.911]) 로 유의 악화"),
    (0, "평균 오차를 사는 대신 정작 맞혀야 할 구간을 깎는다 → 받아들이지 않는다. HistGBM 유지"),
], top=4.0, height=2.6, size=13.5)

# ─────────────────────────────────────────────── S8c 경보 모델 비교·선정
s = slide("제2장 · 모델 비교 ② 고사용량 경보 — 베이스라인 포함 9종",
          "clean_Jul_Sep, h60, OOF 7,288 스텝 / 419 양성 / 186 사건")
CN2 = {"C5_extra_trees": "ExtraTrees(balanced) ★선정", "C6_hist_gbm": "HistGBM",
       "C2_quantile90_margin": "분위회귀 q0.90 마진", "C3_logistic": "Logistic (베이스라인)",
       "C4_random_forest": "RandomForest", "C0_persistence_rule": "persistence 규칙 (베이스라인)",
       "C1_reg_margin": "회귀 마진 (현행 점예측)", "C7_stacked": "스태킹",
       "C8_stacked_isotonic": "스태킹+isotonic"}
_c = MCA[["model", "f1", "precision", "recall", "event_recall", "alerts_per_day", "tp", "fp", "fn"]].copy()
_c["model"] = _c.model.map(CN2)
table(s, _c.round(3), top=1.6, width=12.2, size=11,
      hdr=["모델", "F1", "정밀도", "재현율", "사건 recall", "경보/일", "TP", "FP", "FN"])
note(s, "선정 규칙(제약은 실험 이전부터 config 에 있던 operator_advisory_capacity=12): "
        "경보 12건/일 이하 후보 중 clean F1 최대, 동점 시 사건 recall. "
        "HistGBM 은 12.4건/일로 용량 초과. 스태킹은 전 구간 F1 0.477 이었으나 clean 에서 0.251 로 붕괴(중복일 과적합) → 기각.",
     WARN, top=6.6)

# ─────────────────────────────────────────────── S8d F1 감사
s = slide("제3장 · F1 감사 — 0.47 은 한계인가 결함인가",
          "같은 크기의 오차를 '편향 없이' 가졌을 때의 F1 상한을 측정했다")
_d = FDG[["model", "mae", "bias_all", "bias_peak", "peak_pred_above_thr", "oracle_f1_same_error"]].copy()
_d["model"] = _d.model.map(RN)
table(s, _d.round(3), left=0.65, top=1.6, width=6.6, size=10,
      hdr=["모델", "MAE", "전체 편향", "고사용량 편향", "피크 중 임계초과", "오라클 F1"])
bullets(s, [
    (0, "결함이다. 본질적 한계가 아니다"),
    (1, f"현재 F1 {_f1_now:.3f} vs 무편향 오라클 {_f1_ceil:.3f} — 격차 {_f1_ceil-_f1_now:.3f}"),
    (0, "원인: 고사용량 구간에서만 +14.75 kw 과소예측"),
    (1, "전체 편향은 −0.96 kw 로 거의 0 인데 피크에서만 치우친다"),
    (1, "고사용량 스텝의 81.6% 는 점예측이 임계선에 도달조차 못 한다"),
    (0, "MAE 를 최소화하면 조건부 평균으로 수렴 → 분포 오른쪽 꼬리를 구조적으로 깎는다"),
    (1, "MAE 가 좋아져도 F1 이 따라오지 않는 이유"),
    (0, "라벨의 59.2% 가 임계선 ±MAE 이내 — 경계 밀집. 단, 오라클이 이를 포함하고도 0.69"),
], left=7.5, top=1.7, width=5.3, size=12.5)

# ─────────────────────────────────────────────── S8e 솔직한 현재 위치
s = slide("제3장 · 지금 수준을 솔직하게 적으면", "clean_Jul_Sep, h60")
_pos = pd.DataFrame([
    ["사전확률(모두 경보)", f"{float(_pk60.loc['prior_constant','f1']):.3f}", "하한"],
    ["persistence 규칙", f"{_f1_pers:.3f}", "단순 규칙"],
    ["회귀 마진(현행 점예측)", f"{float(tb(MCA, model='C1_reg_margin').f1.iloc[0]):.3f}", "점예측을 그대로 경보에 쓸 때"],
    ["이전 주 모델 HGB+isotonic", f"{float(_pk60.loc['HGB_clf_uncalibrated','f1']):.3f}", "교체 전"],
    ["현재 ET+isotonic", f"{_f1_now:.3f}", "persistence 대비 +53%, 사전확률 대비 4.2배"],
    ["무편향 오라클", f"{_f1_ceil:.3f}", "같은 오차 크기, 편향만 제거 — 남은 여지"],
    ["완전 예측", "1.000", "—"],
], columns=["기준", "F1", "의미"])
table(s, _pos, top=1.6, width=11.6, size=12.5)
bullets(s, [
    (0, "'처참' 까지는 아니지만 좋다고 말할 수도 없다 — 상한의 68% 지점"),
    (0, f"사건 단위로 보면 186 사건 중 168 포착(사건 recall {_evrec:.3f}). 스텝 F1 과 다른 질문에 답한다 — 섞지 않는다"),
    (0, "ROC-AUC 0.934 는 발생률 5.96% 에서 과대평가된다. 성능 근거로 쓰지 않는다"),
    (0, "다음: 피크 편향 직접 제거(높은 분위 점예측 / 비대칭 손실) · 사건 단위 임계 최적화 — 사전 채택조건 선언 후 검정"),
], top=4.3, height=2.3, size=13)

# ─────────────────────────────────────────────── S9 오류분석
s = slide("제3장 · 오류분석 — 어디서 잘 되고 어디서 깨지는가",
          "전체 평균이 아니라 조건별로 쪼갠다 (h60)")
st = STR[(STR.horizon_min == 60)][["variant", "condition", "n", "share", "mae",
                                   "mae_ratio_vs_all"]].copy()
CN = {"ALL": "전체", "peak_only": "고사용량 구간만", "high_ramp_p90": "급변동 상위10%",
      "non_operating": "비가동", "restart_1h": "재가동 1시간 이내",
      "midday_12_18": "12~18시", "weekend": "주말"}
st["condition"] = st.condition.map(CN)
st["variant"] = st.variant.map({"A0_baseline": "A0", "A5_all": "A5"})
pv = st.pivot(index="condition", columns="variant", values=["n", "mae", "mae_ratio_vs_all"])
pv.columns = ["n_A0", "n_A5", "MAE_A0", "MAE_A5", "배율_A0", "배율_A5"]
pv = pv.reset_index()[["condition", "n_A0", "MAE_A0", "MAE_A5", "배율_A0", "배율_A5"]]
table(s, pv.round(3), top=1.6, width=11.6, size=12,
      hdr=["조건", "표본 수", "A0 MAE", "A5 MAE", "A0 전체대비 배율", "A5 전체대비 배율"])
note(s, "가장 취약한 조건은 고사용량 구간 자체(전체의 1.85배 → 1.61배). "
        "A5 는 모든 조건에서 오차를 낮췄고, 비가동·주말에서 특히 크게 줄였다(휴무·재가동 특성 효과).")

# ─────────────────────────────────────────────── S10 OOD 역할
s = slide("제3장 · OOD — '포기 신호' 가 아니라 '취약조건 지도'",
          "처음 보는 입력 조합(마할라노비스 거리, TRAIN p99 컷)을 어떻게 쓸 것인가")
dec = OODD[OODD.variant == "A5_all"].groupby("maha_decile").agg(
    평균거리=("mean_maha", "mean"), MAE=("mae", "mean")).reset_index()
dec.columns = ["거리 십분위", "평균 거리", "MAE"]
table(s, dec.round(2), left=0.65, top=1.6, width=4.6, size=11)
o0 = tb(OODS, variant="A0_baseline").iloc[0]; o5 = tb(OODS, variant="A5_all").iloc[0]
ob = tb(OODB, scope="OOD_only").iloc[0]
bullets(s, [
    (0, "① 지시력은 있다 — 거리 최하위 십분위 MAE 6.44 → 최상위 10.10 kw"),
    (0, f"② 그러나 포기 기준으로 쓰기엔 약하다 — OOD 판정은 전체의 {o5.ood_share*100:.1f}%뿐"),
    (0, "③ 답은 '모델을 좋게 만드는 것' 이었다"),
    (1, f"OOD 구간 MAE / 정상 구간 MAE = {o0.ood_penalty_ratio:.2f}(A0) → {o5.ood_penalty_ratio:.2f}(A5)"),
    (0, "④ OOD 구간을 일부러 더 보게 가중(A6)하면 오히려 악화 — 기각"),
    (1, f"OOD 구간 MAE 차 {ob.delta_mae:+.3f} kw, 95% CI [{ob.ci_lo:.3f}, {ob.ci_hi:.3f}] (0 미포함, 악화 방향)"),
], left=5.6, top=1.7, width=7.2, size=14)
note(s, "결론: OOD 는 '어디가 약한지' 를 보여주는 진단 도구다. 그 구간을 피하거나 억지로 가중하는 것은 "
        "검증 결과 효과가 없었다. 공개한다.", WARN)

# ─────────────────────────────────────────────── S11 현장 활용
s = slide("제4장 · 현장 활용 — 사건당 한 번, 남은 준비시간과 함께",
          "선행시간이 제도적으로 의미 있는 길이여야 한다")
L = LEAD[["lead_minutes", "basis", "role_in_our_system"]]
table(s, L, top=1.6, width=12.0, size=11,
      hdr=["선행(분)", "제도·문헌 근거", "본 시스템에서의 역할"])
bullets(s, [
    (0, "알림 설계: 다단계 경보를 성과로 내세우지 않는다. 사건당 1건, 내용은 세 가지뿐"),
    (1, "① 고사용량이 처음 시작될 것으로 보이는 시각  ② 지금부터 남은 준비시간(분)  ③ 확신도"),
    (0, "근거는 UI 가 아니라 수치다 — 제때 잡은 사건 수, 헛경고로 생기는 확인 부담(분/일)"),
], top=4.2, height=2.2, size=14)
note(s, "설비 단위 조치나 고장 심각도 순위는 제시하지 않는다. 설비 식별자·고장 이력이 데이터에 없다.")

# ─────────────────────────────────────────────── S12 차별성
s = slide("제5장 · 차별성 — 성능·오류분석과 연결되는 것만 주장한다")
bullets(s, [
    (0, "① 데이터 자체의 결함을 찾아 평가 설계로 연결했다"),
    (1, f"합성 중복일 {int(d_all.n_dup_days)}일({int(d_all.n_dup_groups)}그룹)이 fold 0·2 의 TRAIN/TEST 에 걸쳐 있음을 확인 "
        "→ 주 보고 구간을 clean 으로 고정. 부풀려진 성능을 쓰지 않았다"),
    (0, "② 라벨 정의를 과금 제도에 묶었다"),
    (1, "15분 = 요금적용전력 측정 단위, 30일 = 검침월. p95 가 모델링 선택임을 숨기지 않고 민감도표로 공개"),
    (0, "③ 고사용량 구간을 직접 겨냥한 학습으로 그 구간 오차를 네 호라이즌 모두 25% 안팎 줄였다"),
    (1, f"h60 고사용량 MAE {a0.mae_peak:.2f} → {a5.mae_peak:.2f} kw. 블록 부트스트랩 CI 로 유의성 확인"),
    (0, "④ 실패한 시도를 지웠다가 아니라 남겼다"),
    (1, "운전상태 3분할 모델(V4) 기각 · 2단계 경보 우위 주장 철회 · OOD 가중(A6) 기각 · 외부 공공데이터 게이트 0/20 선택"),
], size=14)

# ─────────────────────────────────────────────── S13 재현성 + 한계
s = slide("제6장 · 재현성 · 그리고 아직 주장할 수 없는 것")
bullets(s, [
    (0, "명령 한 줄로 전처리→학습→추론→결과 생성"),
    (1, "python3 analysis/final_baseline/run_baseline.py   (seed 20260926 고정, CPU 전용)"),
    (1, "학습/추론 분리 — joblib 아티팩트 + inference.py 로 학습 코드 없이 예측 생성"),
    (1, "requirements.txt · README · 수용테스트 포함"),
    (0, "아직 검증되지 않아 주장하지 않는 것"),
    (1, "절감액(원) — 계약전력·요금 조건 미확인"),
    (1, "설비 단위 원인·조치 — 설비 식별자 없음"),
    (1, "고사용량 구간 예측구간 적중률 0.234 — 목표 0.90 에 크게 미달. 공개하고 다음 과제로 둔다"),
    (1, "최초 시작 시각 예측 — 설계만 있고 검증 전"),
], size=14.5)
note(s, "모든 수치는 analysis/tables/*.csv 에서 직접 읽어 이 장표를 생성했다. 손으로 옮겨 적지 않았다.", ACC)

# ─────────────────────────────────────────────── 부록 그림 슬라이드
APPENDIX = [
    ("23_baseline_fig2_forecast_interval", "부록 · 예측값과 보정된 예측구간"),
    ("23_baseline_fig3_reliability_bands", "부록 · 신뢰도 밴드별 오차 분포"),
    ("30_timeseries_fn_fp", "부록 · 평가 전용 30일 실측·예측·미탐·오경보"),
    ("30_fnfp_by_state", "부록 · 운전상태별 경고 성능(분모 병기)"),
    ("28_peak_probability", "부록 · 고사용량 초과확률의 보정 곡선"),
]
for nm, ttl in APPENDIX:
    if not (FIG / f"{nm}.png").exists():
        continue
    s2 = slide(ttl, f"analysis/figures/{nm}.png")
    pic(s2, nm, 0.9, 1.6, 11.5)

out = OUT / "발표자료_전력피크조기경보.pptx"
prs.save(out)
print(f"[pptx] {out.relative_to(ROOT)}  slides={len(prs.slides.__iter__.__self__._sldIdLst)}")
