"""외부 문서 색인 검색 실험 — 범위 필터(공통 + 그 금융사)를 걸었을 때 무엇이 나오는지 본다.

실행 (project 폴더에서):
    python scripts/search_docs.py "저축은행에 맡긴 돈도 보호돼?" --company 웰컴저축은행
    python scripts/search_docs.py "이자에 세금이 붙어?"
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
    parser.add_argument("--company", help="이 금융사 문서도 범위에 넣는다")
    parser.add_argument("-k", type=int, default=config.DOCS_TOP_K)
    args = parser.parse_args()

    scope = [{"scope": "common"}] + ([{"company": args.company}] if args.company else [])
    where = {"$or": scope} if len(scope) > 1 else scope[0]

    print(f"질문: {args.query}  (범위: 공통" + (f" + {args.company})" if args.company else ")"))
    for rank, (doc, distance) in enumerate(
        get_store(config.DOCS_COLLECTION).similarity_search_with_score(args.query, k=args.k, filter=where), 1
    ):
        m = doc.metadata
        body = doc.page_content.split("\n", 1)[-1].replace("\n", " ")[:70]
        print(f"{rank}. [{distance:.3f}] {m['title']} ({m['scope']}) #{m['chunk_index']}  {body}...")


if __name__ == "__main__":
    main()
