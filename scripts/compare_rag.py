"""RAG 사용 전후 답변 비교 (F2015 · 검증 4.6)

실행 (project 폴더에서):
    python scripts/compare_rag.py      → 원본 답을 docs/rag_comparison_raw.md 에 저장

RAG 없이: 같은 역할 · 같은 모델에 상품명과 질문만 준다 (모델의 기억으로 답한다)
RAG 사용: 앱과 같은 경로 — 문서 수집(프롬프트 1 · 2) → retrieve → generate
각 질문에는 공시 · 수집 문서로 확인한 '정답 기준'을 함께 적어 두고, 평가는 docs/rag_comparison.md 에 정리한다.
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from collect.pipeline import collect_for_question  # noqa: E402
from domain.products import label, load_products  # noqa: E402
from rag.graph import build_graph  # noqa: E402
from rag.llm import get_model  # noqa: E402
from store.vectordb import get_store  # noqa: E402

NO_RAG_SYSTEM = "너는 은행 예·적금 상품을 설명하는 금융상품 안내 전문가다. 한국어로 간결하게 답한다."

# (상품명 일부, 질문, 정답 기준 — 공시 데이터 · 수집 문서에서 직접 확인한 사실)
CASES = [
    ("KB맑은하늘적금", "1년 만기 기본금리와 최고금리, 우대조건을 알려줘",
     "12개월 기본 2.65% · 최고 3.45%. 우대: 종이통장 0.1 · 종이서식 0.2 · 대중교통 0.4 · 퀴즈 0.1 (%p, 1년제 최고 0.8%p)"),
    ("우리SUPER주거래적금", "급여이체하면 우대금리를 몇 %p 받아?",
     "급여/연금 이체 연 0.7%p — 우리은행 입출식 계좌의 실적 월 수가 계약기간의 1/2 이상일 때"),
    ("웰뱅 회전 정기예금", "저축은행에 맡긴 돈도 얼마까지 보호돼?",
     "예금자보호 한도 1인당 원리금 1억 원 (2025년 9월 1일부터, 이전 5천만 원) — 위키백과 예금보험"),
    ("KB맑은하늘적금", "대중교통 미션은 구체적으로 어떻게 달성해?",
     "공시에는 '대중교통 미션: 1년제 연 0.4%p'만 있고 달성 방법은 없음 → '확인되지 않음'이 정답"),
    ("카카오뱅크 자유적금", "카카오뱅크는 언제 영업을 시작했어?",
     "2017년 7월 27일 영업 시작 — 위키백과 카카오뱅크"),
    ("카카오뱅크 자유적금", "6개월이랑 12개월 금리가 각각 얼마야?",
     "6개월 기본 3.6% · 최고 3.8% / 12개월 기본 3.75% · 최고 3.95%"),
]


def main() -> None:
    model, docs_store = get_model(), get_store(config.DOCS_COLLECTION)
    graph = build_graph(get_store(config.META_COLLECTION), docs_store, model)
    products = load_products()

    lines = ["# RAG 사용 전후 답변 원본 (F2015)", "",
             f"모델: {config.CHAT_MODEL} · 같은 모델 · 같은 질문. 평가는 rag_comparison.md", ""]
    for n, (name, question, truth) in enumerate(CASES, start=1):
        product = next(p for p in products if name in label(p))
        print(f"[{n}/{len(CASES)}] {product['name']} — {question}")

        started = time.time()
        no_rag = model.invoke([
            {"role": "system", "content": NO_RAG_SYSTEM},
            {"role": "user", "content": f"[상품] {label(product)}\n[질문] {question}"},
        ]).content
        no_rag_seconds = time.time() - started

        started = time.time()
        report = collect_for_question(product, label(product), question, model, docs_store)
        result = graph.invoke({"product_id": product["product_id"], "question": question,
                               "keywords": report["keywords"], "history": []})
        rag_seconds = time.time() - started
        sources = "  \n".join(f"[근거 {i}] {s['title']} — {s['source_type']}"
                              for i, s in enumerate(result["sources"], start=1))

        lines += [
            f"## {n}. {product['name']} — “{question}”", "",
            f"**정답 기준**: {truth}", "",
            f"### RAG 없이 ({no_rag_seconds:.1f}초)", "", no_rag.strip(), "",
            f"### RAG 사용 ({rag_seconds:.1f}초)", "", result["answer"].strip(), "",
            "<details><summary>검색된 근거</summary>", "", sources, "", "</details>", "", "---", "",
        ]

    out = config.PROJECT_DIR / "docs" / "rag_comparison_raw.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
