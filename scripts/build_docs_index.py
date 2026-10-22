"""외부 문서 색인 (RAG 2) — 위키 공통 개념 · 금융사 문서 → finance_docs 컬렉션  (F2005)

실행 (project 폴더에서):
    python scripts/build_docs_index.py
    CHUNK_SIZE=500 python scripts/build_docs_index.py      ← 청크 크기를 바꿔 청크 수 변화 확인 (검증 4.2)

받은 문서는 cache/wiki_pages/ 에 남아 두 번째 실행부터는 위키를 다시 부르지 않는다. (NF2002)
상품별 문서(scope=product)는 질문할 때 수집한다 — 5단계.
"""

import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from collect.pipeline import fetch_company_page  # noqa: E402
from collect.wiki import fetch_page  # noqa: E402
from domain.products import COMMON_WIKI_TITLES, load_products  # noqa: E402
from store.vectordb import count, get_store, index_document  # noqa: E402


def _normalized(text: str) -> str:
    return text.replace(" ", "")


def main() -> None:
    products = load_products()
    if not products:
        raise SystemExit("data/products.json 이 없습니다. 먼저 python scripts/fetch_products.py 를 실행하세요.")

    store = get_store(config.DOCS_COLLECTION)
    print(f"청크 크기 {config.CHUNK_SIZE} · 중복 {config.CHUNK_OVERLAP} · 색인 전 {count(store)}청크\n")
    stats = Counter()

    print("── 공통 개념 문서 (scope=common) ──")
    for title in COMMON_WIKI_TITLES:
        page = fetch_page(title)
        if not page:
            print(f"  ✗ {title} — 문서 없음")
            stats["missing"] += 1
            continue
        # 넘겨주기가 다른 주제로 데려가는 경우가 있다. 예) "단리" → "청풍면 (제천시)" (마을 이름)
        if _normalized(title.split(" (")[0]) not in _normalized(page["title"]):
            print(f"  ✗ {title} → {page['title']} — 넘겨주기가 다른 주제라 버림")
            store._collection.delete(where={"$and": [{"doc_id": page["doc_id"]}, {"scope_key": "common"}]})
            stats["missing"] += 1
            continue
        n = index_document(store, {**page, "scope": "common", "search_query": title})
        print(f"  ✓ {title} → {page['title']} ({len(page['text']):,}자, {n}청크)")
        stats["common"] += 1
        stats["chunks"] += n

    print("\n── 금융사 문서 (scope=company) ──")
    companies = sorted({(p["group"], p["company"]) for p in products})
    found = []
    for group, company in companies:
        page = fetch_company_page(company)
        if not page:
            stats[f"miss_{group}"] += 1
            continue
        n = index_document(store, {**page, "scope": "company", "company": company, "search_query": company})
        found.append(f"{company}→{page['title']}({n})")
        stats[f"hit_{group}"] += 1
        stats["chunks"] += n
    print("  " + ", ".join(found))

    print(f"\n공통 문서 {stats['common']}건 (없음 {stats['missing']})")
    print(f"금융사 문서 — 은행 {stats['hit_은행']}/{stats['hit_은행'] + stats['miss_은행']}, "
          f"저축은행 {stats['hit_저축은행']}/{stats['hit_저축은행'] + stats['miss_저축은행']}")
    print(f"이번 실행에서 색인한 청크 {stats['chunks']} · 컬렉션 전체 {count(store)}청크")


if __name__ == "__main__":
    main()
