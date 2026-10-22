"""질문 시점 문서 수집 — 검색어 추출 → 위키 검색 → 후보 선별 → 본문 수집 → 색인  (F2003 · F2004)

    ① 프롬프트 1 로 상품 · 질문에서 위키 검색어를 뽑는다 (JSON 배열)
    ② 이 상품에서 이미 수집한 검색어는 건너뛴다 (NF2002) — 기록: cache/collected.json
    ③ 새 검색어마다 위키 검색 → 후보(제목 · 요약)를 모은다. 이미 색인된 문서는 후보에서 뺀다
    ④ 프롬프트 2 가 제목 · 요약만 보고 고른다 — 본문을 받기 전에 버려서 외부 호출을 줄인다
    ⑤ 고른 문서만 본문을 받아 scope=product 로 청킹 · 색인한다

"항목 자체를 다룬 문서"(F2004): 상품마다 위키 문서는 없으므로 상품 공시(메타 청크, 항상 근거 1번)와
그 금융사 위키 문서를 항목 문서로 보고, 선별과 관계없이 항상 색인해 둔다.

LangGraph 는 retrieve · generate 두 노드로 둔다 (F2007). 수집은 그래프를 돌리기 전에 한다.
"""

import json
from datetime import date
from typing import Callable

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

import config
from collect.wiki import fetch_page, search
from domain.products import company_wiki_candidates
from rag.parsing import parse_json_array
from rag.prompts import KEYWORD_SYSTEM, SELECT_SYSTEM, build_keyword_prompt, build_select_prompt
from store.vectordb import index_document

MANIFEST_PATH = config.CACHE_DIR / "collected.json"
# 다른 나라 제도 문서 — 프롬프트 2 규칙으로는 "미국 예금보험공사"가 두 번 선별됐다. 질문이 그 나라를 묻지 않으면
# LLM 에 보이기 전에 코드로 뺀다. (규칙은 프롬프트에도 남겨 둔다)
FOREIGN_PREFIXES = ("미국", "일본", "중국", "영국", "독일", "프랑스", "캐나다", "호주", "홍콩", "대만", "싱가포르", "유럽")
SEARCH_LIMIT = 5      # 검색어당 후보 수
MAX_KEYWORDS = 3


# ── 수집 기록 (NF2002) ───────────────────────────────────────────────
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict) -> None:
    config.CACHE_DIR.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_keywords(model: BaseChatModel, item_label: str, item_kind: str, question: str) -> list[str]:
    """프롬프트 1 — 위키 검색어 목록. 공시만으로 답할 수 있는 질문이면 빈 목록."""
    answer = model.invoke([
        {"role": "system", "content": KEYWORD_SYSTEM},
        {"role": "user", "content": build_keyword_prompt(item_label, item_kind, question)},
    ]).content
    keywords = [k.strip() for k in parse_json_array(answer) if isinstance(k, str) and k.strip()]
    return list(dict.fromkeys(keywords))[:MAX_KEYWORDS]


def select_candidates(model: BaseChatModel, question: str, candidates: list[dict]) -> list[int]:
    """프롬프트 2 — 고른 후보의 0부터 시작하는 위치 목록."""
    answer = model.invoke([
        {"role": "system", "content": SELECT_SYSTEM},
        {"role": "user", "content": build_select_prompt(question, candidates)},
    ]).content
    picked = [n for n in parse_json_array(answer) if isinstance(n, int) and 1 <= n <= len(candidates)]
    return sorted({n - 1 for n in picked})


# ── 항목 자체 문서 ───────────────────────────────────────────────────
def fetch_company_page(company: str) -> dict | None:
    """금융사 문서를 정확한 제목(넘겨주기 포함)으로만 찾는다. 검색 결과는 엉뚱한 문서가 섞여 쓰지 않는다.
    넘겨주기로 도착한 문서 제목에 금융사 이름이 없으면 버린다.
    """
    names = company_wiki_candidates(company)
    for title in names:
        page = fetch_page(title)
        if page and names[-1].replace(" ", "") in page["title"].replace(" ", ""):
            return page
    return None


def _indexed_doc_ids(docs_store: Chroma, product: dict) -> set[str]:
    """이 상품의 검색 범위(공통 · 금융사 · 상품)에 이미 들어 있는 문서 ID."""
    found = docs_store.get(
        where={"$or": [{"scope": "common"}, {"company": product["company"]}, {"product_id": product["product_id"]}]},
        include=["metadatas"],
    )
    return {m["doc_id"] for m in found["metadatas"]}


# ── 수집 ─────────────────────────────────────────────────────────────
def collect_for_question(
    product: dict,
    item_label: str,
    question: str,
    model: BaseChatModel,
    docs_store: Chroma,
    on_progress: Callable[[str], None] = lambda message: None,
) -> dict:
    """질문 하나에 필요한 문서를 모아 색인한다. 화면 · 검증용 보고서를 돌려준다."""
    product_id = product["product_id"]
    manifest = load_manifest()
    record = manifest.setdefault(product_id, {
        "keywords": [], "selected": [], "rejected": [], "page_ids": [], "company_checked": False,
    })
    record.setdefault("questions", {})  # 질문 → 그때 뽑은 검색어
    report = {"keywords": [], "new_keywords": [], "selected": [], "rejected": [],
              "already_indexed": [], "chunks": 0, "from_cache": False}

    # 같은 상품에 같은 질문이 다시 오면 검색어 추출(LLM)부터 건너뛴다.
    # LLM은 같은 질문에도 매번 조금 다른 검색어를 내서, 그대로 두면 같은 질문에 수집이 또 일어난다.
    question_key = " ".join(question.split())
    if question_key in record["questions"]:
        report["keywords"] = record["questions"][question_key]
        report["from_cache"] = True
        on_progress("이미 수집한 질문입니다. 저장된 문서를 사용합니다.")
        return report

    indexed = _indexed_doc_ids(docs_store, product)

    # 항목 자체 문서 — 금융사 문서는 선별을 거치지 않고 항상 색인한다. (상품당 한 번만 확인)
    if not record["company_checked"]:
        on_progress(f"{product['company']} 문서 확인 중...")
        page = fetch_company_page(product["company"])
        if page and page["doc_id"] not in indexed:
            report["chunks"] += index_document(docs_store, {
                **page, "scope": "company", "company": product["company"], "search_query": product["company"],
            })
            report["selected"].append(f"{page['title']} (금융사 문서 · 항상 포함)")
            indexed.add(page["doc_id"])
        record["company_checked"] = True

    on_progress("검색어를 고르는 중...")
    keywords = extract_keywords(model, item_label, product["kind_name"], question)
    new_keywords = [k for k in keywords if k not in record["keywords"]]
    report["keywords"], report["new_keywords"] = keywords, new_keywords

    if new_keywords:
        candidates, seen = [], set()
        for keyword in new_keywords:
            on_progress(f"위키백과 검색 중: {keyword}")
            for hit in search(keyword, limit=SEARCH_LIMIT):
                doc_id = f"wiki-{hit['page_id']}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                if doc_id in indexed:
                    report["already_indexed"].append(hit["title"])
                    continue
                country = next((c for c in FOREIGN_PREFIXES if hit["title"].startswith(c)), None)
                if country and country not in question:
                    report["rejected"].append(hit["title"])
                    continue
                candidates.append({**hit, "doc_id": doc_id, "keyword": keyword})

        if candidates:
            on_progress(f"후보 {len(candidates)}건 중 근거로 쓸 문서를 고르는 중...")
            picked = set(select_candidates(model, question, candidates))
            for i, candidate in enumerate(candidates):
                if i not in picked:
                    report["rejected"].append(candidate["title"])
                    continue
                on_progress(f"문서 수집 · 색인 중: {candidate['title']}")
                page = fetch_page(candidate["title"])
                if not page:
                    continue
                report["chunks"] += index_document(docs_store, {
                    **page, "scope": "product", "product_id": product_id,
                    "company": product["company"], "search_query": candidate["keyword"],
                })
                report["selected"].append(page["title"])
                record["page_ids"].append(page["doc_id"])

        record["keywords"] += new_keywords
        record["selected"] += [t for t in report["selected"] if "항상 포함" not in t]
        record["rejected"] += report["rejected"]

    record["questions"][question_key] = keywords
    record["collected_at"] = date.today().isoformat()
    _save_manifest(manifest)
    return report
