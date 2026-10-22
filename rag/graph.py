"""RAG 파이프라인 — LangGraph 로 검색(retrieve)과 생성(generate)을 두 노드로 나눈다. (F2006 · F2007)

    START → retrieve → generate → END

retrieve
    ① 선택한 상품의 메타 청크 — 검색 없이 id 로 꺼내 항상 근거 1번에 둔다
    ② 외부 문서 청크 — 여러 검색어로 찾아 병합한다
         · 질문 원문                  → 개념 질문("단리랑 복리 차이")에 강하다
         · 금융사 + 상품명 + 질문      → "이 은행", "여기"처럼 대상을 생략한 질문에 강하다
         · 프롬프트 1 이 뽑은 검색어들  → 주제가 여럿 섞인 질문에 강하다
       검색 범위 = 공통 문서 + 그 금융사 문서 + 그 상품 문서 (다른 금융사 문서가 섞이지 않게)
    ③ 너무 먼 청크는 버리고, 같은 문서의 같은 조각은 하나로 합친 뒤(중복 제거)
       두 검색의 순위를 RRF 로 합쳐 Top-k
generate
    프롬프트 3 + 최근 대화로 답한다. 근거 번호는 retrieve 가 만든 순서 그대로다.
"""

from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

import config
from rag.prompts import ANSWER_SYSTEM, build_answer_prompt

META_SOURCE_TYPE = "상품 공시"
RRF_K = 60  # RRF 관례값. 1등과 2등의 점수 차이를 완만하게 만든다.


class RagState(TypedDict, total=False):
    # 입력
    product_id: str
    question: str
    keywords: list[str]      # 수집 단계(프롬프트 1)가 뽑은 검색어 — 검색에도 함께 쓴다
    history: list[dict]      # 최근 대화 [{"role", "content"}]
    # retrieve 결과
    item_label: str
    sources: list[dict]      # [{"title", "source_type", "url", "text", "distance"}]
    # generate 결과
    answer: str


def _meta_source(meta_store: Chroma, product_id: str) -> dict | None:
    found = meta_store.get(ids=[product_id])
    if not found["ids"]:
        return None
    m = found["metadatas"][0]
    return {
        "key": f"meta:{product_id}",
        "title": f"{m['name']} ({m['company']})",
        "source_type": f"{META_SOURCE_TYPE} · {m['disclosed_month'][:4]}-{m['disclosed_month'][4:]}",
        "url": m["url"],
        "text": found["documents"][0],
        "distance": 0.0,
        "_meta": m,
    }


def _doc_sources(docs_store: Chroma, meta: dict, question: str, keywords: list[str]) -> list[dict]:
    where = {"$or": [
        {"scope": "common"},
        {"company": meta["company"]},
        {"product_id": meta["product_id"]},
    ]}
    # 주제가 둘 섞인 질문("세금은? 중도해지는?")은 질문 전체로 검색하면 어느 쪽에도 가깝지 않다.
    # 수집 단계가 뽑은 검색어로도 따로 찾아 RRF 로 합친다.
    queries = [question, f"{meta['company']} {meta['name']} {question}", *keywords]

    merged: dict[str, dict] = {}
    for query in queries:
        results = docs_store.similarity_search_with_score(query, k=config.DOCS_TOP_K, filter=where)
        for rank, (doc, distance) in enumerate(results, start=1):
            if distance > config.DOCS_MAX_DISTANCE:  # 너무 먼 청크는 순위와 관계없이 버린다
                continue
            m = doc.metadata
            # 같은 위키 문서가 공통 · 상품 범위로 두 번 색인돼 있어도 내용은 같다 → 문서 ID + 순번으로 묶는다.
            key = f"{m['doc_id']}:{m['chunk_index']}"
            source = merged.setdefault(key, {
                "key": key,
                "title": m["title"],
                "source_type": m["source_type"],
                "url": m["url"],
                "text": doc.page_content.split("\n", 1)[-1],  # 색인할 때 붙인 제목 줄은 뺀다
                "distance": distance,
                "rrf": 0.0,
            })
            source["distance"] = min(source["distance"], distance)
            # Reciprocal Rank Fusion — 거리 값이 아니라 '각 검색에서 몇 등이었나'로 합친다.
            # 거리로만 합치면 상품명을 붙인 검색의 금융사 문서가 늘 이겨, 개념 문서(복리 등)가 밀려난다.
            source["rrf"] += 1 / (RRF_K + rank)

    return sorted(merged.values(), key=lambda s: -s["rrf"])[: config.DOCS_TOP_K]


def build_graph(meta_store: Chroma, docs_store: Chroma, model: BaseChatModel, history_turns: int = 6):
    """저장소와 모델을 받아 그래프를 만든다. 앱은 한 번 만들어 캐시해 두고 계속 쓴다."""

    def retrieve(state: RagState) -> RagState:
        meta_source = _meta_source(meta_store, state["product_id"])
        if meta_source is None:
            return {"sources": [], "item_label": state["product_id"]}
        meta = meta_source.pop("_meta")
        sources = [meta_source] + _doc_sources(docs_store, meta, state["question"], state.get("keywords", []))
        label = f"{meta['name']} · {meta['company']}"
        return {"sources": sources, "item_label": label}

    def generate(state: RagState) -> RagState:
        history = state.get("history", [])[-history_turns:]
        messages = (
            [{"role": "system", "content": ANSWER_SYSTEM}]
            + history
            + [{"role": "user", "content": build_answer_prompt(state["item_label"], state["sources"], state["question"])}]
        )
        return {"answer": model.invoke(messages).content}

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
