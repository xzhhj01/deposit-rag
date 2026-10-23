"""금리 돋보기 — 광고 금리 말고, 내가 받을 금리를 근거와 함께 보여 주는 예·적금 도우미

실행 (project 폴더에서):
    streamlit run app.py

streamlit_lab/step7_skeleton.py 골격에서 출발했다.
    ① 상품 선택 — 유형 · 금융권으로 좁힌 뒤 고른다              (F2009)
    ② 대화 기록을 상품별로 따로 저장한다                        (F2009)
    ③ 키 누락 · 데이터 없음 · 색인 없음 안내, 금융 면책 안내     (F2012)
    ④ 💬 질의응답 — 문서 수집(프롬프트 1 · 2) → RAG 그래프(retrieve → generate) → 인용된 근거만 표시
                                                                          (F2003 ~ F2008)
    ⑤ 🧮 내 금리 계산 — 우대조건 판정(LLM) + 금리 · 이자 계산(코드)       (F2010)
    ⑥ 🔎 조건으로 찾기 — 자연어 조건 → 필터 + 우대조건 검색 → LLM 검증 → 카드 → 대화로 이동 (F2011)
UI 는 화면만 그린다. 수집 · 검색 · 생성 · 계산은 각 모듈이 한다. (NF2005)
"""

import re

import openai
import streamlit as st

import config
from collect.pipeline import collect_for_question
from domain.products import build_item_text, label, load_products
from rag.graph import build_graph
from rag.llm import get_model
from services.product_search import rate_for, search_products
from services.rate_check import estimate, judge_conditions, terms
from store.vectordb import count, get_store

DISCLAIMER = (
    "본 서비스는 금융감독원 공시 자료를 바탕으로 한 참고 정보이며, 금융상품 가입 권유나 "
    "전문가 상담을 대신하지 않습니다. 실제 금리와 조건은 가입 전 해당 금융사에서 확인하세요."
)
MODE_CHAT, MODE_RATE, MODE_SEARCH = "💬 질의응답", "🧮 내 금리 계산", "🔎 조건으로 찾기"

# 답변 본문의 [근거 3], [근거 3, 5] 표기를 찾는다. (step7 그대로)
CITE_PATTERN = re.compile(r"\[근거\s*([\d,\s]+)\]")
HISTORY_TURNS = 6  # LLM에 함께 넘길 최근 대화 수 — 길어질수록 느려지고 비싸진다 (step6 한 걸음 더)
VERDICT_ICON = {"충족": "✅", "미충족": "❌", "확인 불가": "❔"}


# ── 자원 (한 번만 만들고 모든 세션이 함께 쓴다) ─────────────────────────
@st.cache_resource
def cached_meta_store():
    return get_store(config.META_COLLECTION)


@st.cache_resource
def cached_docs_store():
    return get_store(config.DOCS_COLLECTION)


@st.cache_resource
def cached_condition_store():
    return get_store(config.CONDITION_COLLECTION)


@st.cache_resource
def cached_graph():
    return build_graph(cached_meta_store(), cached_docs_store(), get_model(), history_turns=HISTORY_TURNS)


@st.cache_data
def cached_products() -> dict[str, dict]:
    """상품 ID → 상품. 파일을 매번 다시 읽지 않도록 값을 캐시한다."""
    return {p["product_id"]: p for p in load_products()}


def open_product(product_id: str) -> None:
    """조건 탐색 카드 → 그 상품의 질의응답으로. (F2011)

    위젯이 그려지기 '전'에 도는 콜백에서 key 로 값을 바꾼다. 위젯의 index 를 계산해 넘기는 방식은
    선택이 한 번 무시되는 함정이 있다. (streamlit_lab/plus_widget_key.py)
    """
    product = cached_products()[product_id]
    st.session_state.kind = product["kind_name"]
    st.session_state.group = product["group"]
    st.session_state.product_id = product_id
    st.session_state.mode = MODE_CHAT


# ── 근거 표시 (F2008) — step7 에서 출발, 출처 유형 표시 · 같은 문서 묶기를 더했다 ──
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
    # 같은 문서의 다른 조각(예: 예금보험공사 #1, #3)은 한 줄로 묶는다 — [근거 3, 4] 예금보험공사
    grouped: dict[tuple, list[int]] = {}
    for number, source in used:
        grouped.setdefault((source["title"], source["source_type"], source["url"]), []).append(number)
    with st.expander(f"답변에 사용한 근거 {len(grouped)}건"):
        for (title, source_type, url), numbers in grouped.items():
            st.markdown(f"**[근거 {', '.join(map(str, numbers))}]** {title} · {source_type} · [원문]({url})")


def error_message(error: Exception) -> str:
    """오류 종류만 골라 사용자가 할 일을 알려 준다. 오류 내용 · 트레이스백은 보이지 않는다. (F2012 · NF2004)"""
    if isinstance(error, openai.AuthenticationError):
        return "GMS 키가 올바르지 않거나 만료되었습니다. project/.env 의 GMS_KEY 를 확인하고 앱을 다시 실행하세요."
    if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
        return "LLM 서버에 연결하지 못했습니다. 네트워크를 확인하고 잠시 후 다시 시도해 주세요."
    if isinstance(error, openai.RateLimitError):
        return "요청이 많아 잠시 제한되었습니다. 잠시 후 다시 시도해 주세요."
    return "처리 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요."


def collect_summary(report: dict) -> str:
    """수집 결과 한 줄 — 무엇을 찾아 무엇을 골랐는지. 프롬프트 전문은 보이지 않는다. (NF2004)"""
    if report["from_cache"]:
        return "✓ 이미 수집한 질문 — 저장된 문서 사용"
    if not report["keywords"]:
        return "✓ 상품 공시만으로 답할 수 있는 질문 — 추가 수집 없음"
    parts = [f"검색어: {', '.join(report['keywords'])}"]
    if report["selected"]:
        parts.append(f"새로 수집: {', '.join(report['selected'])}")
    if report["rejected"]:
        parts.append(f"제외 {len(report['rejected'])}건")
    if not report["new_keywords"]:
        parts.append("모두 이미 수집됨")
    return " · ".join(parts)


# ── 💬 질의응답 ──────────────────────────────────────────────────────
def render_chat(product: dict) -> None:
    product_id = product["product_id"]
    messages = st.session_state.chats.setdefault(product_id, [])

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message["content"], message.get("sources", []))  # 다시 그릴 때도 근거를 함께

    question = st.chat_input("이 상품에 대해 궁금한 점을 물어보세요 (예: 우대금리 조건이 뭐야?)")
    if not question:
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 최근 대화를 함께 넘겨 "그럼 2년이면?" 같은 이어지는 질문을 이해하게 한다. (step6)
        history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
        sources, answer, failure = [], "", "잠시 후 다시 시도해 주세요."

        # 수집 → 그래프(retrieve → generate). 단계마다 진행 상태를 바꾼다. (F2012)
        with st.status("관련 문서를 확인하고 있습니다...", expanded=False) as status:
            try:
                report = collect_for_question(
                    product, label(product), question, get_model(), cached_docs_store(),
                    on_progress=lambda message: status.update(label=message),
                )
                st.caption(collect_summary(report))

                state = {"product_id": product_id, "question": question,
                         "keywords": report["keywords"], "history": history}
                status.update(label="근거를 찾고 있습니다...")
                for update in cached_graph().stream(state, stream_mode="updates"):
                    if "retrieve" in update:
                        sources = update["retrieve"]["sources"]
                        status.update(label=f"근거 {len(sources)}건 확보 · 답변을 만드는 중...")
                        st.caption(" · ".join(f"{s['title']}({s['source_type'].split(' ·')[0]})" for s in sources))
                    elif "generate" in update:
                        answer = update["generate"]["answer"]
                status.update(label=f"완료 · 검색한 근거 {len(sources)}건", state="complete")
            except Exception as error:  # 상세 오류(트레이스백)는 화면에 내보내지 않는다 (NF2004)
                status.update(label="처리 중 문제가 생겼습니다", state="error")
                failure = error_message(error)

        if answer:
            st.markdown(answer)
            render_sources(answer, sources)
        else:
            st.error(f"답변을 만들지 못했습니다. {failure}")

    if answer:
        messages.append({"role": "assistant", "content": answer, "sources": sources})


# ── 🧮 내 금리 계산 (F2010) ───────────────────────────────────────────
def render_rate_check(product: dict) -> None:
    product_id, saving = product["product_id"], product["kind"] == "saving"
    available = terms(product)
    if not available:
        st.info("이 상품은 공시된 금리 옵션이 없어 계산할 수 없습니다.")
        return

    st.markdown("광고에 나온 **최고금리** 말고, **내 상황에서 실제로 받을 금리**와 세후 이자를 계산합니다.")
    left, right = st.columns(2)
    term = left.selectbox("가입기간", available, format_func=lambda t: f"{t}개월", key=f"term_{product_id}",
                          index=available.index(12) if 12 in available else 0)
    amount = right.number_input("월 납입액(원)" if saving else "예치금(원)", min_value=10_000, step=10_000,
                                value=500_000 if saving else 10_000_000, key=f"amount_{product_id}")

    situation = ""
    if product["special_condition"]:
        with st.expander("이 상품의 우대조건 원문"):
            st.text(product["special_condition"])
        situation = st.text_area(
            "내 상황", key=f"situation_{product_id}", height=90,
            placeholder="예) 월급은 다른 은행으로 받고, 이 은행 카드는 없어요. 대중교통은 매일 타요.",
        )
    else:
        st.info("이 상품은 공시된 우대조건이 없습니다. 기본금리가 곧 최고금리입니다.")

    if st.button("판정하고 계산하기", type="primary"):
        judgment = {"cap": None, "conditions": []}
        if product["special_condition"]:
            with st.status("우대조건을 하나씩 판정하는 중...", expanded=False) as status:
                try:
                    judgment = judge_conditions(get_model(), product, term, situation)
                    status.update(label=f"우대조건 {len(judgment['conditions'])}개 판정 완료", state="complete")
                except Exception as error:  # NF2004
                    status.update(label="판정 중 문제가 생겼습니다", state="error")
                    st.error(error_message(error))
                    return
        st.session_state.rate_results[product_id] = {
            "judgment": judgment, "estimate": estimate(product, term, int(amount), judgment),
        }

    result = st.session_state.rate_results.get(product_id)
    if result:
        render_rate_result(product, result["judgment"], result["estimate"])


def render_rate_result(product: dict, judgment: dict, e: dict) -> None:
    low, high = e["low"], e["high"]
    rate_text = f"{low['rate']}%" if low["rate"] == high["rate"] else f"{low['rate']} ~ {high['rate']}%"

    st.divider()
    a, b, c = st.columns(3)
    a.metric("광고 최고금리", f"{e['advertised_max']}%", help="공시된 최고우대금리")
    b.metric("내 예상 금리", rate_text, delta=f"{low['rate'] - e['advertised_max']:+.2f}%p", delta_color="off")
    # 범위를 지표 하나에 넣으면 칸이 좁아 잘린다("86,609 ~ 92,…") → 확실히 받는 금액만 지표로, 위쪽 끝은 아래 줄에
    c.metric(f"세후 이자 ({e['term']}개월)", f"{low['after_tax']:,}원",
             help=f"최고금리를 다 받으면 {e['max']['after_tax']:,}원")
    if high["after_tax"] != low["after_tax"]:
        c.caption(f"확인 불가까지 받으면 최대 {high['after_tax']:,}원")
    st.caption(f"기본금리 {e['base_rate']}% · {e['rate_type']} · 원금 {e['principal']:,}원 · "
               "범위의 앞쪽은 '충족'만, 뒤쪽은 '확인 불가'까지 모두 받는 경우")

    if judgment["conditions"]:
        st.markdown("**우대조건 판정**")
        for cond in judgment["conditions"]:
            rate = f"+{cond['rate']}%p" if cond["rate"] is not None else "금리 미표기"
            st.markdown(f"{VERDICT_ICON[cond['verdict']]} **{cond['name']}** ({rate}) — {cond['verdict']}  \n"
                        f"{cond['reason']}")
            notes = []
            if cond["verdict"] != "충족" and cond["how_to"]:
                notes.append(f"받으려면: {cond['how_to']}")
            if cond["quote"]:
                notes.append(f"공시 원문: “{cond['quote']}”")
            if not cond["verified"]:
                notes.append("⚠ 원문에서 이 금리 숫자를 확인하지 못해 계산에서 뺐습니다")
            if notes:
                st.caption(" · ".join(notes))
    if e["capped"]:
        st.caption("ⓘ 우대금리 합계가 공시된 상한을 넘어 상한까지만 반영했습니다.")

    # 근거와 계산 가정을 나눠 보여 준다 — 근거는 공시, 세율 · 납입 방식은 계산 가정 (F2010)
    with st.expander("근거와 계산 가정"):
        st.markdown(f"**[근거 1]** {product['name']} ({product['company']}) · 상품 공시 · "
                    f"{product['disclosed_month'][:4]}-{product['disclosed_month'][4:]} · [원문]({product['url']})  \n"
                    "기간별 기본 · 최고금리, 우대조건 원문")
        assumptions = [f"이자소득세 {config.INTEREST_TAX_RATE * 100:.1f}% (일반과세 기준, 비과세 · 세금우대 상품은 다름)"]
        if product["kind"] == "saving":
            assumptions.append("매월 같은 날 같은 금액을 납입한다고 가정 (실제 일할 계산과 차이가 날 수 있음)")
        st.markdown("**계산 가정** — 근거 문서가 아니라 계산을 위해 정한 값입니다  \n" + "  \n".join(f"· {a}" for a in assumptions))


# ── 🔎 조건으로 찾기 (F2011) ──────────────────────────────────────────
def describe_filters(f: dict) -> str:
    parts = [{"saving": "적금", "deposit": "정기예금"}.get(f["kind"], "예금·적금 전체")]
    if f["group"]:
        parts.append(f["group"])
    if f["term_months"]:
        parts.append(f"{f['term_months']}개월")
    if f["min_rate"] is not None:
        parts.append(f"최고금리 {f['min_rate']}% 이상")
    if f["no_condition"]:
        parts.append("우대조건 없음")
    if f["include_keywords"]:
        parts.append(f"포함: {', '.join(f['include_keywords'])}")
    if f["exclude_keywords"]:
        parts.append(f"제외: {', '.join(f['exclude_keywords'])}")
    return " · ".join(parts)


def render_search(products: dict[str, dict]) -> None:
    st.markdown("원하는 조건을 말로 쓰면, 공시 내용으로 **조건에 맞는 상품만** 골라 보여 줍니다.")
    with st.form("search_form", border=False):
        request = st.text_input("찾는 조건", placeholder="예) 급여이체 없이 우대받는 1년 적금 / 대중교통 타면 금리 더 주는 적금")
        submitted = st.form_submit_button("찾기", type="primary")

    if submitted and request.strip():
        with st.status("조건을 정리하고 상품을 찾는 중...", expanded=False) as status:
            try:
                result = search_products(get_model(), cached_condition_store(), products, request.strip())
                st.session_state.search_result = {"request": request.strip(), **result}
                status.update(label=f"맞는 상품 {len(result['results'])}개", state="complete")
            except Exception as error:  # NF2004
                status.update(label="검색 중 문제가 생겼습니다", state="error")
                st.error(error_message(error))

    result = st.session_state.get("search_result")
    if not result:
        return
    f = result["filters"]
    st.caption(f"“{result['request']}” → {describe_filters(f)}  \n"
               f"조건 통과 {result['passed_count']}개 → 검토한 후보 {result['candidate_count']}개 → "
               f"맞는 상품 {len(result['results'])}개")
    if not result["results"]:
        st.info("조건에 맞는 상품을 찾지 못했습니다. 조건을 조금 넓혀 다시 찾아보세요.")
        return

    columns = st.columns(2)
    for i, item in enumerate(result["results"]):
        p = products[item["product_id"]]
        with columns[i % 2].container(border=True):
            st.markdown(f"**{p['name']}**  \n{p['company']} · {p['group']} {p['kind_name']}")
            st.markdown(f"최고 **{rate_for(p, f['term_months'])}%**"
                        + (f" ({f['term_months']}개월)" if f["term_months"] else ""))
            st.caption(item["reason"])
            st.button("이 상품 질문하기", key=f"open_{item['product_id']}", width="stretch",
                      on_click=open_product, args=(item["product_id"],))
    st.caption("근거: 각 상품의 금융감독원 공시(금리 · 우대조건 · 가입 대상) — 금융감독원 금융상품한눈에")


# ── 화면 ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="금리 돋보기", page_icon="🔍")
st.title("🔍 금리 돋보기")
st.caption("광고 금리 말고, 내가 받을 금리를 근거와 함께 — 예·적금 공시 기반 안내")
st.warning(DISCLAIMER, icon="⚠️")  # 금융은 판단이 위험한 도메인 — 항상 보인다 (F2012)

# 실행 전제 점검 — 하나라도 없으면 무엇을 해야 하는지 안내하고 멈춘다. (F2012 · NF2004: 상세 오류는 숨긴다)
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

for key, initial in [("chats", {}), ("rate_results", {}), ("mode", MODE_CHAT)]:
    if key not in st.session_state:
        st.session_state[key] = initial  # chats: 상품 ID → 대화 (F2009) · rate_results: 상품 ID → 계산 결과

# 상품 선택 — 771개를 한 목록에 두면 찾기 어렵다. 유형 · 금융권으로 먼저 좁힌다.
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
    st.caption(f"색인 — 상품 메타데이터 {meta_count}건 · 외부 문서 {count(cached_docs_store())}청크 · "
               f"우대조건 {count(cached_condition_store())}줄")
    st.caption(f"출처: 금융감독원 금융상품한눈에 · 공시 {next(iter(products.values()))['disclosed_month']} / "
               "한국어 위키백과 (CC BY-SA 4.0)")

mode = st.segmented_control("기능", [MODE_CHAT, MODE_RATE, MODE_SEARCH], key="mode",
                            label_visibility="collapsed") or MODE_CHAT

if mode == MODE_SEARCH:
    render_search(products)
    st.stop()

if not product_id:
    st.info("왼쪽에서 상품을 먼저 고르거나, 🔎 조건으로 찾기에서 상품을 찾아보세요.")
    st.stop()

product = products[product_id]
st.subheader(label(product))
with st.expander("공시 정보 보기", expanded=False):
    st.text(build_item_text(product))

if mode == MODE_CHAT:
    render_chat(product)
else:
    render_rate_check(product)
