"""금리 돋보기 — 광고 금리 말고, 내가 받을 금리를 근거와 함께 보여 주는 예·적금 도우미

실행 (project 폴더에서):
    streamlit run app.py

streamlit_lab/step7_skeleton.py 골격에서 출발했다. 지금 단계(2단계)에서 채운 것:
    ① 상품 선택 — 유형 · 금융권으로 좁힌 뒤 고른다              (F2009)
    ② 대화 기록을 상품별로 따로 저장한다                        (F2009)
    ③ 키 누락 · 데이터 없음 · 색인 없음 안내, 금융 면책 안내     (F2012)
    ④ 근거: 선택한 상품의 공시 정보(메타 청크) 1건만            ← 외부 문서 검색은 3~4단계에서 붙인다
"""

import re

import streamlit as st

import config
from domain.products import build_item_text, label, load_products
from rag.llm import get_model
from store.vectordb import count, get_store

DISCLAIMER = (
    "본 서비스는 금융감독원 공시 자료를 바탕으로 한 참고 정보이며, 금융상품 가입 권유나 "
    "전문가 상담을 대신하지 않습니다. 실제 금리와 조건은 가입 전 해당 금융사에서 확인하세요."
)

# 답변 본문의 [근거 3], [근거 3, 5] 표기를 찾는다. (step7 그대로)
CITE_PATTERN = re.compile(r"\[근거\s*([\d,\s]+)\]")
HISTORY_TURNS = 6  # LLM에 함께 넘길 최근 대화 수 — 길어질수록 느려지고 비싸진다 (step6 한 걸음 더)


# ── 자원 (한 번만 만들고 모든 세션이 함께 쓴다) ─────────────────────────
@st.cache_resource
def cached_model():
    return get_model()


@st.cache_resource
def cached_meta_store():
    return get_store(config.META_COLLECTION)


@st.cache_data
def cached_products() -> dict[str, dict]:
    """상품 ID → 상품. 파일을 매번 다시 읽지 않도록 값을 캐시한다."""
    return {p["product_id"]: p for p in load_products()}


# ── 근거 표시 (F2008) — step7 그대로, 출처 유형만 더했다 ───────────────
def cited_sources(answer: str, sources: list[dict]) -> list[tuple[int, dict]]:
    """답변이 실제로 인용한 근거만 (번호, 문서)로 돌려준다. 인용이 없으면 빈 목록."""
    numbers = {
        int(n) for group in CITE_PATTERN.findall(answer)
        for n in group.split(",") if n.strip().isdigit()
    }
    return [(i, s) for i, s in enumerate(sources, start=1) if i in numbers]


def render_sources(answer: str, sources: list[dict]) -> None:
    used = cited_sources(answer, sources)
    if not used:
        st.caption("ⓘ 답변에 인용된 근거가 없습니다.")
        return
    with st.expander(f"답변에 사용한 근거 {len(used)}건"):
        for number, source in used:
            st.markdown(
                f"**[근거 {number}]** {source['title']} · {source['source_type']} · [원문]({source['url']})"
            )


# ── 근거 수집 (지금은 메타 청크 1건) ──────────────────────────────────
def retrieve_sources(product_id: str) -> list[dict]:
    """선택한 상품의 메타 청크를 Vector DB에서 id로 꺼낸다. 검색 없이 항상 포함하는 근거다. (F2006)"""
    found = cached_meta_store().get(ids=[product_id])
    if not found["ids"]:
        return []
    meta = found["metadatas"][0]
    return [{
        "title": f"{meta['name']} ({meta['company']})",
        "source_type": f"상품 공시 · {meta['disclosed_month'][:4]}-{meta['disclosed_month'][4:]}",
        "url": meta["url"],
        "text": found["documents"][0],
    }]


def build_prompt(product: dict, sources: list[dict], question: str) -> str:
    context = "\n\n".join(
        f"[근거 {i}] {s['title']} ({s['source_type']})\n{s['text']}" for i, s in enumerate(sources, start=1)
    )
    return (
        "너는 은행 예·적금 상품을 설명하는 금융상품 안내 전문가다.\n"
        "규칙:\n"
        "1. 아래 [근거]에 있는 내용만 사용한다. 기억이나 추측으로 답하지 않는다.\n"
        "2. 근거에 없으면 '공시 자료에서 확인되지 않습니다'라고 답하고, 금융사에 확인하라고 안내한다.\n"
        "3. 근거에서 온 문장 뒤에는 [근거 N] 형태로 번호를 붙인다.\n"
        "4. 가입을 권유하거나 '좋은 상품'이라고 평가하지 않는다. 사실만 설명한다.\n\n"
        f"[근거]\n{context}\n\n[선택한 상품]\n{label(product)}\n\n[질문]\n{question}"
    )


# ── 화면 ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="금리 돋보기", page_icon="🔍")
st.title("🔍 금리 돋보기")
st.caption("광고 금리 말고, 내가 받을 금리를 근거와 함께 — 예·적금 공시 기반 안내")
st.warning(DISCLAIMER, icon="⚠️")  # 금융은 판단이 위험한 도메인 — 항상 보인다 (F2012)

# ③ 실행 전제 점검 — 하나라도 없으면 무엇을 해야 하는지 안내하고 멈춘다. (F2012 · NF2004: 상세 오류는 숨긴다)
products = cached_products()
if not config.GMS_KEY:
    st.error("GMS_KEY가 설정되지 않았습니다. project/.env 에 키를 넣고 다시 실행하세요.")
    st.stop()
if not products:
    st.error("상품 데이터가 없습니다. 터미널에서 `python scripts/fetch_products.py` 를 실행하세요.")
    st.stop()
meta_count = count(cached_meta_store())
if meta_count == 0:
    st.error("상품 색인이 비어 있습니다. 터미널에서 `python scripts/build_meta_index.py` 를 실행하세요.")
    st.stop()

if "chats" not in st.session_state:
    st.session_state.chats = {}  # 상품 ID → 그 상품의 대화 목록 (F2009)

# ① 상품 선택 — 771개를 한 목록에 두면 찾기 어렵다. 유형 · 금융권으로 먼저 좁힌다.
with st.sidebar:
    st.header("상품 선택")
    kind = st.radio("유형", ["적금", "정기예금"], horizontal=True, key="kind")
    group = st.radio("금융권", ["은행", "저축은행"], horizontal=True, key="group")
    candidates = sorted(
        (p for p in products.values() if p["kind_name"] == kind and p["group"] == group),
        key=lambda p: (p["company"], p["name"]),
    )
    product_id = st.selectbox(
        f"{kind} · {group} {len(candidates)}개",
        [p["product_id"] for p in candidates],
        format_func=lambda pid: label(products[pid]),
        index=None,
        placeholder="상품명이나 금융사를 입력하세요",
        key="product_id",
    )
    if product_id and st.button("이 상품 대화 지우기", width="stretch"):
        st.session_state.chats[product_id] = []

    st.divider()
    st.caption(f"상품 메타데이터 색인: {meta_count}건")
    st.caption(f"출처: 금융감독원 금융상품한눈에 · 공시 {next(iter(products.values()))['disclosed_month']}")

if not product_id:
    st.info("왼쪽에서 상품을 먼저 고르세요.")
    st.stop()

product = products[product_id]
messages = st.session_state.chats.setdefault(product_id, [])

# 선택한 상품 요약 — 질문하기 전에 무엇을 물을 수 있는지 보이게 한다.
st.subheader(label(product))
with st.expander("공시 정보 보기", expanded=not messages):
    st.text(build_item_text(product))

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message["content"], message.get("sources", []))  # 다시 그릴 때도 근거를 함께

question = st.chat_input("이 상품에 대해 궁금한 점을 물어보세요 (예: 우대금리 조건이 뭐야?)")

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("근거를 찾고 있습니다...", expanded=False) as status:
            sources = retrieve_sources(product_id)
            status.update(label=f"근거 {len(sources)}건 확보", state="complete")
            st.caption(" · ".join(s["title"] for s in sources))

        # 최근 대화를 함께 넘겨 "그럼 2년이면?" 같은 이어지는 질문을 이해하게 한다. (step6)
        history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1][-HISTORY_TURNS:]]
        payload = history + [{"role": "user", "content": build_prompt(product, sources, question)}]

        with st.spinner("답변을 만드는 중..."):
            try:
                answer = cached_model().invoke(payload).content
            except Exception:
                answer = ""

        if answer:
            st.markdown(answer)
            render_sources(answer, sources)
        else:
            st.error("답변을 만들지 못했습니다. 잠시 후 다시 시도해 주세요.")

    if answer:
        messages.append({"role": "assistant", "content": answer, "sources": sources})
