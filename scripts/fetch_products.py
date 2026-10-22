"""상품 메타데이터 수집 — 금감원 API → data/products.json  (F2001)

실행 (project 폴더에서):
    python scripts/fetch_products.py

공시는 매월 바뀐다. 다시 받고 싶을 때만 실행한다. 색인은 build_meta_index.py 가 따로 한다.
"""

import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from domain.products import fetch_products, save_products  # noqa: E402


def main() -> None:
    if not config.FINLIFE_KEY:
        raise SystemExit("FINLIFE_KEY가 .env 에 없습니다.")

    print("금감원 API에서 예금 · 적금을 받는 중...")
    products = fetch_products()
    save_products(products)

    by_type = Counter(f"{p['kind_name']} · {p['group']}" for p in products)
    with_condition = sum(1 for p in products if p["special_condition"])
    print(f"저장: {config.PRODUCTS_PATH}")
    print(f"상품 {len(products)}건 — " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    print(f"우대조건 원문이 있는 상품 {with_condition}건 · 공시월 {products[0]['disclosed_month']}")


if __name__ == "__main__":
    main()
