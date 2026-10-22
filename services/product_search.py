"""조건 기반 상품 탐색 (F2011) — 자연어 조건 → 맞는 상품만 카드로

    ① LLM  : 요청을 필터 조건(유형 · 금융권 · 기간 · 최저금리 · 우대조건 없음)과
             있어야 할 핵심어(include) · 피해야 할 핵심어(exclude)로 나눈다
    ② 코드 : 필터 조건으로 상품을 거른다 — 숫자 비교 · 범주는 임베딩이 못 한다 (1단계 문제 2)
    ③ 후보 : include → 우대조건 줄 단위 색인에서 의미 검색 (1단계 문제 1: 상품 전체 390위 → 줄 단위 6위)
             exclude → 의미 검색은 반대 상품을 가져오므로 코드로 단어가 없는 상품을 앞에 둔다
             둘 다 없으면 금리 높은 순
    ④ LLM  : 후보의 공시 내용을 보고 조건을 '모두' 만족하는 상품만 고른다
"""

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from domain.products import label
from rag.parsing import parse_json_array, parse_json_object
from rag.prompts import SEARCH_PARSE_SYSTEM, SEARCH_VERIFY_SYSTEM, build_search_verify_prompt

MAX_CANDIDATES = 15  # LLM 검증에 넘길 후보 수
SEMANTIC_K = 60      # 우대조건 줄 검색에서 가져올 줄 수 (여러 줄이 한 상품일 수 있어 넉넉히)


def parse_request(model: BaseChatModel, request: str) -> dict:
    raw = parse_json_object(model.invoke([
        {"role": "system", "content": SEARCH_PARSE_SYSTEM},
        {"role": "user", "content": request},
    ]).content)
    return {
        "kind": raw.get("kind") if raw.get("kind") in ("saving", "deposit") else None,
        "group": raw.get("group") if raw.get("group") in ("은행", "저축은행") else None,
        "term_months": raw.get("term_months") if isinstance(raw.get("term_months"), int) else None,
        "min_rate": float(raw["min_rate"]) if isinstance(raw.get("min_rate"), (int, float)) else None,
        "no_condition": raw.get("no_condition") is True,
        "include_keywords": _words(raw.get("include_keywords")),
        "exclude_keywords": _words(raw.get("exclude_keywords")),
    }


def _words(value) -> list[str]:
    return [w.strip() for w in value if isinstance(w, str) and w.strip()] if isinstance(value, list) else []


def rate_for(product: dict, term: int | None) -> float:
    """비교에 쓸 최고금리 — 기간을 정했으면 그 기간, 아니면 전 기간 중 최고."""
    rates = [o["max_rate"] or 0 for o in product["options"] if term is None or o["term"] == term]
    return max(rates, default=0.0)


def _passes(product: dict, f: dict) -> bool:
    if f["kind"] and product["kind"] != f["kind"]:
        return False
    if f["group"] and product["group"] != f["group"]:
        return False
    if f["term_months"] and f["term_months"] not in {o["term"] for o in product["options"]}:
        return False
    if f["min_rate"] is not None and rate_for(product, f["term_months"]) < f["min_rate"]:
        return False
    if f["no_condition"] and product["special_condition"]:
        return False
    return True


def _candidate_text(product: dict, term: int | None) -> str:
    rates = " / ".join(
        f"{o['term']}개월 기본 {o['base_rate']}% · 최고 {o['max_rate']}%"
        for o in product["options"] if term is None or o["term"] == term
    )
    condition = product["special_condition"][:500] or "없음"
    return (f"{label(product)} · {product['group']}\n금리: {rates}\n가입 대상: {product['join_member']}"
            f" · 가입 방법: {product['join_way']}\n우대조건: {condition}")


def search_products(model: BaseChatModel, condition_store: Chroma, products: dict[str, dict], request: str) -> dict:
    """products: 상품 ID → 상품. 결과: {'filters', 'candidate_count', 'results': [{'product_id', 'reason'}]}"""
    f = parse_request(model, request)
    passed = {pid for pid, p in products.items() if _passes(p, f)}

    by_rate = sorted(passed, key=lambda pid: -rate_for(products[pid], f["term_months"]))

    if f["include_keywords"] and not f["no_condition"]:
        # 있어야 하는 조건 — 핵심어마다 우대조건 줄 단위 색인에서 찾고, 핵심어별 순위를 번갈아 합친다.
        # 긴 문장으로 찾으면 "우대금리 · 적금" 같은 공통어에 끌려 정작 '대중교통' 상품이 60위 밖으로 밀렸다.
        per_keyword = []
        for keyword in f["include_keywords"]:
            ranked = []
            for doc in condition_store.similarity_search(keyword, k=SEMANTIC_K):
                pid = doc.metadata["product_id"]
                if pid in passed and pid not in ranked:
                    ranked.append(pid)
            per_keyword.append(ranked)
        candidates = []
        for rank in range(SEMANTIC_K):
            for ranked in per_keyword:
                if rank < len(ranked) and ranked[rank] not in candidates:
                    candidates.append(ranked[rank])
    elif f["exclude_keywords"]:
        # 피해야 하는 조건 — 의미 검색은 '급여이체'로 찾으면 급여이체 상품을 가져온다. 대신 코드로 거른다:
        # 우대조건이 있는 상품 중 그 단어가 없는 상품을 금리 순으로 앞에, 단어가 있는 상품은 뒤에 둔다.
        # (단어가 있어도 다른 방법으로 우대받을 수 있으면 조건을 만족하므로 버리지 않고 LLM 검증에 맡긴다)
        with_condition = [pid for pid in by_rate if products[pid]["special_condition"]]
        mentions = lambda pid: any(w in products[pid]["special_condition"] for w in f["exclude_keywords"])
        candidates = [pid for pid in with_condition if not mentions(pid)] + \
                     [pid for pid in with_condition if mentions(pid)]
    else:
        # 필터만 있는 요청 — 금리 높은 순
        candidates = by_rate
    candidates = candidates[:MAX_CANDIDATES]

    results = []
    if candidates:
        picked = parse_json_array(model.invoke([
            {"role": "system", "content": SEARCH_VERIFY_SYSTEM},
            {"role": "user", "content": build_search_verify_prompt(
                request, [_candidate_text(products[pid], f["term_months"]) for pid in candidates])},
        ]).content)
        by_name = {products[pid]["name"]: pid for pid in candidates}
        for item in picked:
            if not isinstance(item, dict):
                continue
            # 번호(P3)와 상품명이 가리키는 후보가 다르면 상품명을 믿는다 — 번호는 0부터/1부터를 헷갈리기 쉽다.
            pid = by_name.get(str(item.get("name") or "").strip())
            if pid is None:
                number = str(item.get("id") or "").lstrip("Pp")
                if number.isdigit() and 1 <= int(number) <= len(candidates):
                    pid = candidates[int(number) - 1]
            if pid and pid not in {r["product_id"] for r in results}:
                results.append({"product_id": pid, "reason": str(item.get("reason") or "")})

    return {"filters": f, "passed_count": len(passed), "candidate_count": len(candidates), "results": results[:6]}
