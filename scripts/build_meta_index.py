"""상품 메타데이터 색인 (RAG 1) — data/products.json → product_meta 컬렉션  (F2002)

실행 (project 폴더에서):
    python scripts/build_meta_index.py

상품 1개 = 청크 1개, 청크 id = 상품 ID.
두 번 실행해도 같은 id 를 덮어쓰기만 하므로 건수가 늘지 않는다. (검증 4.1)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from domain.products import build_item_metadata, build_item_text, load_products  # noqa: E402
from store.vectordb import count, get_store, upsert  # noqa: E402


def main() -> None:
    products = load_products()
    if not products:
        raise SystemExit("data/products.json 이 없습니다. 먼저 python scripts/fetch_products.py 를 실행하세요.")

    store = get_store(config.META_COLLECTION)
    before = count(store)

    upsert(
        store,
        texts=[build_item_text(p) for p in products],
        ids=[p["product_id"] for p in products],
        metadatas=[build_item_metadata(p) for p in products],
    )

    after = count(store)
    print(f"로드한 상품 {len(products)}건 · 색인 전 {before}건 → 색인 후 {after}건")
    print("일치" if after == len(products) else "불일치 — 확인 필요")

    # 상품 1건의 색인 텍스트를 출력해 의도한 속성이 다 들어갔는지 본다. (검증 4.1)
    sample = next(p for p in products if p["special_condition"] and p["kind"] == "saving")
    print("\n── 색인 텍스트 예시 ──")
    print(build_item_text(sample))


if __name__ == "__main__":
    main()
