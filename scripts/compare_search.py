"""유사도 검색 vs MMR 검색 — 같은 질문의 근거 목록 비교 (F2014 · 검증 4.5)

실행 (project 폴더에서):
    python scripts/compare_search.py           → 결과를 docs/mmr_comparison.md 에 저장

외부 문서 근거만 비교한다 (메타 청크 1건은 두 방식 모두 항상 근거 1번).
다양성 지표: 고유 문서 수 · 한 문서에서 가장 많이 뽑힌 청크 수 · 근거끼리의 평균 코사인 유사도(낮을수록 다양)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter  # noqa: E402
from itertools import combinations  # noqa: E402

import numpy as np  # noqa: E402

import config  # noqa: E402
from domain.products import label, load_products  # noqa: E402
from rag.graph import _doc_sources  # noqa: E402
from store.vectordb import get_embeddings, get_store  # noqa: E402
from domain.products import build_item_metadata  # noqa: E402

# (상품명 일부, 질문, 수집 단계 검색어) — 검색어는 프롬프트 1이 실제로 뽑은 값을 고정해 두어 비교를 재현 가능하게 한다
CASES = [
    ("웰뱅 회전 정기예금", "저축은행이 파산하면 내 예금은 어떻게 돼?", ["예금보험", "예금보험공사"]),
    ("KB맑은하늘적금", "이자에 세금은 얼마나 떼?", ["원천징수", "금융소득종합과세"]),
    ("카카오뱅크 자유적금", "인터넷전문은행은 일반 은행이랑 뭐가 달라?", ["네오뱅크"]),
    ("KB맑은하늘적금", "단리랑 복리 차이가 뭐야?", ["복리", "이자"]),
]


def diversity(sources: list[dict], embeddings) -> tuple[int, int, float]:
    if not sources:
        return 0, 0, 0.0
    titles = Counter(s["title"] for s in sources)
    vectors = np.array(embeddings.embed_documents([s["text"] for s in sources]))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = [float(vectors[a] @ vectors[b]) for a, b in combinations(range(len(sources)), 2)]
    return len(titles), max(titles.values()), (sum(pairs) / len(pairs) if pairs else 0.0)


def main() -> None:
    docs_store, embeddings = get_store(config.DOCS_COLLECTION), get_embeddings()
    products = load_products()
    lines = [
        "# 유사도 검색 vs MMR 검색 비교 (F2014)",
        "",
        f"설정: Top-k {config.DOCS_TOP_K} · MMR fetch_k {config.MMR_FETCH_K} · λ {config.MMR_LAMBDA} · "
        f"거리 기준선 {config.DOCS_MAX_DISTANCE}  ",
        "지표: 고유 문서 수(높을수록 다양) · 한 문서 최다 청크 수(낮을수록 덜 몰림) · 근거 간 평균 유사도(낮을수록 다양)",
        "",
    ]
    for name, question, keywords in CASES:
        product = next(p for p in products if name in label(p))
        meta = build_item_metadata(product)
        lines += [f"## {product['name']} — “{question}”", "", "| 방식 | 근거 (외부 문서) | 고유 문서 | 최다 청크 | 평균 유사도 |",
                  "|---|---|---|---|---|"]
        print(f"\n[{product['name']}] {question}")
        for search_type in ("similarity", "mmr"):
            sources = _doc_sources(docs_store, meta, question, keywords, search_type)
            unique, most, similarity = diversity(sources, embeddings)
            listing = " · ".join(f"{s['title']}#{s['key'].split(':')[-1]}" for s in sources)
            print(f"  {search_type:10} 고유 {unique} · 최다 {most} · 유사도 {similarity:.3f} | {listing}")
            lines.append(f"| {search_type} | {listing} | {unique} | {most} | {similarity:.3f} |")
        lines.append("")

    out = config.PROJECT_DIR / "docs" / "mmr_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
