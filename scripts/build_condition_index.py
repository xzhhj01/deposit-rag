"""조건 탐색용 보조 색인 — 우대조건 한 줄 = 청크 하나  (F2011)

실행 (project 폴더에서):
    python scripts/build_condition_index.py

product_meta(상품 1개 = 청크 1개)는 질의응답 근거용으로 그대로 두고,
"대중교통 우대", "공과금 자동이체" 같은 조건으로 상품을 찾을 때만 이 색인을 쓴다.
매번 전체를 지우고 다시 만든다 — 공시가 바뀌면 줄 수가 달라져 예전 줄이 남을 수 있기 때문이다.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from domain.products import build_item_metadata, load_products, split_conditions  # noqa: E402
from store.vectordb import count, get_store, upsert  # noqa: E402


def main() -> None:
    products = load_products()
    if not products:
        raise SystemExit("data/products.json 이 없습니다. 먼저 python scripts/fetch_products.py 를 실행하세요.")

    store = get_store(config.CONDITION_COLLECTION)
    existing = store.get(include=[])["ids"]
    if existing:
        store.delete(ids=existing)

    texts, ids, metadatas = [], [], []
    for p in products:
        meta = build_item_metadata(p)
        for i, line in enumerate(split_conditions(p)):
            texts.append(f"{p['name']} 우대조건: {line}")
            ids.append(f"{p['product_id']}:cond:{i}")
            metadatas.append({**meta, "line_index": i})
    upsert(store, texts, ids, metadatas)

    with_lines = len({m["product_id"] for m in metadatas})
    print(f"우대조건이 있는 상품 {with_lines}개 → 조건 줄 {len(texts)}개 색인 · 컬렉션 {count(store)}청크")


if __name__ == "__main__":
    main()
