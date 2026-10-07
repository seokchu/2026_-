"""결과보고서 양식(.hwpx) 에 본문을 채워 넣는 최소 도구.

양식 파일의 문단 스타일(paraPrIDRef/charPrIDRef)을 그대로 재사용한다.
표지(문단 0~17)와 말미(만족도 조사 안내)는 건드리지 않는다.
장 제목 다음의 '휴먼명조 …' 자리표시 문단과 '작성 요령' 표는 제거하고 본문으로 대체한다.
"""
import re, shutil, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

# 양식에서 확인한 스타일 ID (레벨별)
STYLE = {1: (25, 14), 2: (26, 13), 3: (27, 16)}
LINESEG = ('<hp:linesegarray><hp:lineseg textpos="0" vertpos="0" vertsize="1000" '
           'textheight="1000" baseline="850" spacing="600" horzpos="0" horzsize="48188" '
           'flags="393216"/></hp:linesegarray>')


def para(level, text):
    pa, ch = STYLE[level]
    pre = {1: " ◦ ", 2: "   - ", 3: "       * "}[level]
    return (f'<hp:p id="0" paraPrIDRef="{pa}" styleIDRef="0" pageBreak="0" columnBreak="0" '
            f'merged="0"><hp:run charPrIDRef="{ch}"><hp:t>{escape(pre + text)}</hp:t></hp:run>'
            f'{LINESEG}</hp:p>')


def split_paragraphs(xml):
    """최상위 <hp:p> 들을 순서대로 (start,end) 로 반환. 중첩 표 안의 p 는 건너뛴다."""
    out, depth, i = [], 0, 0
    for m in re.finditer(r"<hp:p\b|</hp:p>", xml):
        if m.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                out.append((i, m.end()))
        else:
            if depth == 0:
                i = m.start()
            depth += 1
    return out


def build(template, out_path, chapters, project_name=None, team_name=None, summary=None):
    """chapters: {장제목에 포함된 식별 문자열: [(level, text), ...]}"""
    tmp = Path(out_path).with_suffix(".work")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(template) as z:
        z.extractall(tmp)
    sec = tmp / "Contents" / "section0.xml"
    xml = sec.read_text(encoding="utf-8")
    spans = split_paragraphs(xml)
    texts = ["".join(re.findall(r"<hp:t>(.*?)</hp:t>", xml[a:b], re.S)) for a, b in spans]

    head_idx = [i for i, t in enumerate(texts) if re.search(r"제\d장\.", t)]
    assert len(head_idx) == 6, f"장 제목 {len(head_idx)}개 — 양식이 바뀌었다"
    tail_start = head_idx[-1]
    # 마지막 장 본문 끝 = '※ 추가적으로' 문단 직전
    for i in range(tail_start, len(texts)):
        if "추가적으로 기술할 내용" in texts[i]:
            tail_start = i - 2
            break

    pieces = [xml[:spans[head_idx[0]][0]]]
    for n, hi in enumerate(head_idx):
        pieces.append(xml[spans[hi][0]:spans[hi][1]])          # 장 제목 원본 유지
        key = next((k for k in chapters if k in texts[hi]), None)
        body = chapters.get(key, [(1, "(작성 예정)")])
        pieces.extend(para(lv, tx) for lv, tx in body)
        pieces.append(para(1, "") if False else "")
    pieces.append(xml[spans[tail_start][0]:])
    xml2 = "".join(pieces)

    for lbl, val in (("프로젝트명", project_name), ("팀명", team_name), ("내용요약", summary)):
        if not val:
            continue
        m = re.search(rf"<hp:t>{lbl}</hp:t>", xml2)
        if not m:
            continue
        # 값 셀의 빈 run (<hp:run charPrIDRef="N"/>) 을 텍스트 run 으로 교체
        nxt = re.search(r'<hp:run charPrIDRef="(\d+)"\s*/>', xml2[m.end():])
        if not nxt:
            continue
        a, b = m.end() + nxt.start(), m.end() + nxt.end()
        xml2 = (xml2[:a] + f'<hp:run charPrIDRef="{nxt.group(1)}"><hp:t>{escape(val)}</hp:t></hp:run>'
                + xml2[b:])

    sec.write_text(xml2, encoding="utf-8")

    out = Path(out_path)
    if out.exists(): out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", (tmp / "mimetype").read_bytes(),
                   compress_type=zipfile.ZIP_STORED)
        for p in sorted(tmp.rglob("*")):
            if p.is_file() and p.name != "mimetype":
                z.write(p, str(p.relative_to(tmp)))
    shutil.rmtree(tmp)
    return out
