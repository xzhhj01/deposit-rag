"""RAG 그래프를 터미널에서 실행 — 근거 목록과 답을 함께 본다.

실행 (project 폴더에서):
    python scripts/ask.py "KB맑은하늘적금" "우대조건 뭐야?"
    DOCS_TOP_K=2 python scripts/ask.py "웰뱅 회전 정기예금" "여기 맡긴 돈도 보호돼?"    ← Top-k 바꿔 비교 (검증 4.3)
    python scripts/ask.py --graph                                              ← 그래프 구조 출력 (검증 4.3)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from domain.products import label, load_products  # noqa: E402
from rag.graph import build_graph  # noqa: E402
from rag.llm import get_model  # noqa: E402
from store.vectordb import get_store  # noqa: E402


def main() -> None:
    graph = build_graph(get_store(config.META_COLLECTION), get_store(config.DOCS_COLLECTION), get_model())

    if sys.argv[1:] == ["--graph"]:
        mermaid = graph.get_graph().draw_mermaid()
        out = config.PROJECT_DIR / "docs" / "rag_graph.md"
        out.write_text(f"# RAG 그래프 구조\n\n```mermaid\n{mermaid}```\n", encoding="utf-8")
        print(mermaid)
        print(f"저장: {out}")
        return

    name, question = sys.argv[1], sys.argv[2]
    product = next((p for p in load_products() if name in label(p)), None)  # "상품명 · 금융사" 일부로 찾는다
    if not product:
        raise SystemExit(f"'{name}'이(가) 들어간 상품이 없습니다.")

    print(f"상품: {product['name']} · {product['company']}  |  질문: {question}  |  Top-k {config.DOCS_TOP_K}")
    result = graph.invoke({"product_id": product["product_id"], "question": question, "history": []})

    print("\n── 근거 목록 (retrieve) ──")
    for i, s in enumerate(result["sources"], start=1):
        print(f"[근거 {i}] {s['title']} — {s['source_type']}  (거리 {s['distance']:.2f})")
    print("\n── 답변 (generate) ──")
    print(result["answer"])


if __name__ == "__main__":
    main()
