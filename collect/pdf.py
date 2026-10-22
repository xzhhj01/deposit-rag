"""PDF 문서 수집 — 위키백과와 종류가 다른 두 번째 문서 출처 (F2016)

PDF 를 텍스트로 바꿔 문서(dict) 목록으로 만든다. 문서 모양은 위키 문서와 같아서 색인 · 검색 코드를 그대로 쓴다.
출처 종류에 따라 문서를 나누는 단위를 다르게 한다.

    page     : 2쪽짜리 안내장 · 리플릿. 쪽마다 주제("보호한도", "보호금융상품 목록")가 나뉘어 있다
               → 쪽 = 문서 = 청크 하나 (자르지 않는다 — 표 머리말과 내용이 떨어지지 않게)
    glossary : 용어사전. 고정 길이(1,000자)로 자르면 "뱅크런"과 "밴드웨건효과"처럼 무관한 용어가 한 청크에
               섞여 엉뚱한 근거가 인용된다 → 용어 항목 하나 = 문서 하나. 항목은 "연관검색어" 줄로 끝난다.

원문 출처는 "파일명 · 쪽"으로 남긴다 (F2004 "URL 또는 파일명과 쪽"). 쪽 번호는 PDF 뷰어에 보이는 쪽(1부터)이다.
"""

import re
from datetime import date
from pathlib import Path

from pypdf import PdfReader

import config

PDF_DIR = config.DATA_DIR / "pdf"
RELATED = re.compile(r"^\s*연관검색어\b.*$", re.MULTILINE)
# 용어사전 쪽 머리말: 쪽 번호, 자모 색인(ㄱ, A), 그 쪽의 대표 용어 — 본문이 아니다
HEADER_LINE = re.compile(r"^\s*(\d{1,3}|[ㄱ-ㅎ]|[A-Z])\s*$")
RUNNING_TITLE = re.compile(r"^\s*I\s*경제금융용어\s*\d+선\s*$", re.MULTILINE)  # 짝수 쪽 머리말


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "-", text).strip("-")[:40]


def _strip_page_header(text: str) -> str:
    """쪽 맨 위의 머리말(대표 용어 · 자모 · 쪽 번호) 줄을 걷어낸다. 앞쪽 4줄 안에서만 본다."""
    lines = text.splitlines()
    head = lines[:4]
    keep_from = 0
    for i, line in enumerate(head):
        nxt = head[i + 1] if i + 1 < len(head) else ""
        if HEADER_LINE.match(line) or (i == 0 and HEADER_LINE.match(nxt)):
            keep_from = i + 1
    return "\n".join(lines[keep_from:])


def _page_documents(spec: dict, pages: list[str]) -> list[dict]:
    docs = []
    for number, text in enumerate(pages, start=1):
        text = text.strip()
        if len(text) < 30:
            continue
        docs.append({
            "doc_id": f"pdf-{_slug(spec['name'])}-p{number}",
            "title": f"{spec['name']} {number}쪽",
            "text": text,
            # 쪽을 더 자르지 않는다. 리플릿의 '보호 / 비보호' 2단 표는 텍스트로 뽑으면 열 머리말이 표 맨 위에
            # 한 번만 남는다. 1,000자로 자르면 "뮤추얼펀드"가 머리말 없는 청크에 들어가 보호 상품처럼 읽힌다.
            "chunk_size": len(text) + 1,
        })
    return docs


def _split_unmarked_terms(entry: str) -> list[str]:
    """'연관검색어' 줄 없이 끝난 항목은 다음 용어와 붙어 나온다. 짧은 줄(용어 이름) 바로 다음 줄이
    그 이름으로 시작하면 새 용어로 본다. 예) "금리스왑" / "금리스왑(Interest Rate Swap)은 …"
    """
    lines = entry.splitlines()
    parts, current = [], []
    for i, line in enumerate(lines):
        name = line.strip()
        following = lines[i + 1].strip() if i + 1 < len(lines) else ""
        is_term = (current and 1 < len(name) <= 25 and not name.endswith((".", ",", ")"))
                   and following.startswith(name.split("/")[0].split("(")[0][:4]))
        if is_term:
            parts.append("\n".join(current))
            current = []
        current.append(line)
    parts.append("\n".join(current))
    return [p.strip() for p in parts if p.strip()]


def _glossary_documents(spec: dict, pages: list[str]) -> list[dict]:
    """용어 항목 단위로 나눈다. 항목의 첫 줄이 용어 이름이고, '연관검색어' 줄에서 끝난다."""
    # 머리말 · 목차는 건너뛴다 — 첫 '연관검색어'가 나오는 쪽부터 본문이다
    first = next((i for i, text in enumerate(pages) if RELATED.search(text)), 0)

    # 쪽마다 머리말을 걷어내고 이어 붙이되, 각 글자가 몇 쪽에서 왔는지 기억해 둔다
    body, page_at = "", []
    for number, text in enumerate(pages[first:], start=first + 1):
        cleaned = RUNNING_TITLE.sub("", _strip_page_header(text)) + "\n"
        body += cleaned
        page_at += [number] * len(cleaned)

    docs, start = [], 0
    for match in RELATED.finditer(body):
        block_start, block = start, body[start:match.end()]
        start = match.end()
        offset = block_start + len(block) - len(block.lstrip())
        for entry in _split_unmarked_terms(block.strip()):
            lines = entry.splitlines()
            term = lines[0].strip()
            entry_start = body.find(entry, offset)
            offset = max(offset, entry_start)
            if len(lines) < 2 or not term or len(term) > 60 or len(entry) < 40:
                continue
            page = page_at[min(max(entry_start, 0), len(page_at) - 1)]
            docs.append({
                "doc_id": f"pdf-{_slug(spec['name'])}-{len(docs):04d}",
                "title": f"{term} ({spec['name']} {page}쪽)",
                "text": entry,
            })
    return docs


def load_pdf_documents(spec: dict) -> list[dict]:
    """PDF 한 개 → 문서 목록. 파일이 없으면 빈 목록."""
    path = PDF_DIR / spec["file"]
    if not path.exists():
        return []
    pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    split = _glossary_documents if spec["chunking"] == "glossary" else _page_documents
    collected_at = date.fromtimestamp(Path(path).stat().st_mtime).isoformat()
    return [
        {
            **doc,
            "url": spec["url"],
            "source_type": f"PDF · {spec['publisher']}",
            "search_query": spec["file"],
            "collected_at": collected_at,
        }
        for doc in split(spec, pages)
    ]
