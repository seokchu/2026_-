"""31-B. 업종 판정 + 경고 선행시간의 외부 근거 표.

여기서 하는 일은 '조사한 근거를 기계판독 표로 고정'하는 것뿐이다.
수치를 만들어내지 않는다. 각 행은 출처 URL 을 가진다.
"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from common import save_table                                            # noqa: E402

IND = [dict(
    field="업종", value="선박엔진용 볼트·너트 제조 (금속 패스너)",
    evidence="KAMP 자원 최적화 AI 데이터셋 설명 — 1개 공장(선박엔진용 볼트·너트), "
             "2021-01-01~09-14, 257일×24시간 = 6,168행×18열",
    source="KAMP 데이터셋 설명 / 동일 데이터 사용 공개 저장소 2건 교차확인",
    url="https://github.com/junseo0im/fac-power-forecast",
    confidence="2개 독립 출처 일치. 원 데이터 파일 자체에는 업종 표기 없음"),
    dict(field="데이터 출처 표기", value="중소벤처기업부·KAMP·자원 최적화 AI 데이터셋·"
         "KAIST(울산과학기술원, ㈜유피시앤에스), 2021.12.27.",
         evidence="데이터셋 공식 출처 문구", source="KAMP", url="https://www.kamp-ai.kr/",
         confidence="인용 문구 그대로"),
    dict(field="주요 전력부하 공정", value="유도가열(단조 전 예열)·열처리(소입/소려)·가공",
         evidence="패스너 제조에서 유도가열은 단조 전 예열과 열처리에 사용되며 "
                  "수 초 내 단조온도 도달",
         source="Inductotherm / Ajax TOCCO 산업 자료",
         url="https://inductothermgroup.com/wp-content/uploads/Induction-Heating-and-Heat-Treating-of-Fasteners_2019.pdf",
         confidence="업종 일반론. 본 공장 설비구성은 데이터에 없음 → 설비단위 주장 불가")]

LEAD = [dict(
    lead_minutes=15, basis="한전 수요시한(디멘드 타임) 15분",
    detail="최대수요전력 = 15분 평균전력. 기존 최대수요전력제어장치는 수요시한 시작 "
           "동기신호 기준으로 15분 뒤 피크를 예측해 단계적 부하차단",
    role_in_our_system="기존 제어기가 이미 커버하는 구간. h15 는 '기존 수준 재현' 비교군",
    url="https://www.eom.co.kr/3.engineering/2.elec/7.energy%20saving/2.power/demand_control.htm"),
    dict(lead_minutes=60, basis="KPX 수요자원 거래시장 감축요청 통보 리드타임",
         detail="감축요청은 1시간 전 통보, 감축지속 원칙 1시간. "
                "이행 상황을 보고 30분 내 추가 감축요청 가능",
         role_in_our_system="h60 = 제도상 표준 통보 리드타임과 일치. 본 시스템의 주 호라이즌",
         url="https://www.kpx.or.kr/boardDownload.es?bid=0045&list_no=51262&seq=16199"),
    dict(lead_minutes=30, basis="KPX 추가 감축요청 여유(30분)",
         detail="1차 요청 후 부족분을 30분 내 추가 요청하는 운영 관행",
         role_in_our_system="minimum_lead_minutes=30 의 근거. 이보다 짧으면 제도적 조치 창이 없음",
         url="https://www.kpx.or.kr/boardDownload.es?bid=0045&list_no=51262&seq=16199"),
    dict(lead_minutes=30, basis="산업 수요관리 스케줄링의 표준 시간슬롯",
         detail="수요측 관리 스케줄링에서 30분 슬롯이 통상 단위. "
                "산업 에너지수요 예측 가이드라인의 권장 해상도는 1분~1시간",
         role_in_our_system="30분 하한의 문헌적 보강",
         url="https://www.mdpi.com/1996-1073/19/10/2328")]

BILL = [dict(
    rule="요금적용전력 산정", detail="15분 단위 측정 평균전력의 월 최대값. 검침 당월 포함 "
    "직전 12개월 중 7~9월분·12~2월분 및 당월분 중 최대값으로 연간 기본요금 연동",
    implication_for_label="① 측정 단위 15분 = 본 분석 해상도와 동일 "
    "② 과금은 '월 최대 1점' 이므로 직전 ~30일(검침월) 창이 요금 주기와 정합 "
    "③ 단, 월 최대 1점만 라벨로 쓰면 월당 양성 1건 → 학습·평가 불가. "
    "따라서 p95 는 '학습 가능한 양성 표본 확보'를 위한 모델링 선택이지 요금 규칙이 아님",
    url="https://cyber.kepco.co.kr/ckepco/front/jsp/CY/H/C/CYHCHP00203.jsp"),
    dict(rule="하계 피크월", detail="7~9월분이 연간 기본요금을 결정하는 달에 포함",
         implication_for_label="본 분석의 주 구간 clean_Jul_Sep(2021-07-01~09-14)은 "
         "요금 결정에 직접 쓰이는 기간. 구간 선택이 과금 관점에서 정당",
         url="https://cyber.kepco.co.kr/ckepco/front/jsp/CY/H/C/CYHCHP00203.jsp")]

save_table(pd.DataFrame(IND), "31_industry_identification")
save_table(pd.DataFrame(LEAD), "31_leadtime_basis")
save_table(pd.DataFrame(BILL), "31_tariff_label_basis")
print("업종/선행시간/과금 근거 표 3종 저장")
