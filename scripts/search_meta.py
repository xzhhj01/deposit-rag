"""메타 색인 검색 실험 — 의미 검색이 무엇을 잘하고 무엇을 못하는지 직접 본다.

실행 (project 폴더에서):
    python scripts/search_meta.py "급여이체 없이 우대금리 받는 적금"
    python scripts/search_meta.py "금리 3% 넘는 적금"
    python scripts/search_meta.py "금리 3% 넘는 적금" --min-rate 3.0     ← 메타데이터 필터를 함께 걸기

점수(distance)는 코사인 거리다. 0에 가까울수록 질문과 의미가 가깝다.
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from store.vectordb import get_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--min-rate", type=float, help="최고금리(best_max_rate) 하한 필터")
    args = parser.parse_args()

    store = get_store(config.META_COLLECTION)
    where = {"best_max_rate": {"$gte": args.min_rate}} if args.min_rate is not None else None

    print(f"질문: {args.query}" + (f"  (필터: 최고금리 ≥ {args.min_rate}%)" if where else ""))
    results = store.similarity_search_with_score(args.query, k=args.k, filter=where)
    for rank, (doc, distance) in enumerate(results, start=1):
        m = doc.metadata
        condition = doc.page_content.split("우대조건: ")[1].split("\n")[0][:60]
        print(f"{rank}. [{distance:.3f}] {m['name']} · {m['company']} ({m['kind']}) "
              f"최고 {m['best_max_rate']}%\n     우대조건: {condition}")


if __name__ == "__main__":
    main()
