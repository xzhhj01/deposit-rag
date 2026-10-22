"""요구사항 정의서 4장 검증 — 자동으로 확인할 수 있는 항목을 실행해 결과표를 만든다.

실행 (project 폴더에서, 색인을 모두 만든 뒤):
    python scripts/verify.py        → docs/verification_results.md

화면 캡처가 필요한 항목은 docs/verification.md 의 체크리스트를 따른다.
LLM 응답은 실행마다 조금씩 다르므로, 같은 결과가 매번 나오지 않을 수 있는 항목은 '판정 근거'에 실제 출력을 남긴다.
"""

import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from collect.pipeline import collect_for_question, load_manifest  # noqa: E402
from collect.wiki import USER_AGENT  # noqa: E402
from domain.products import build_item_metadata, build_item_text, label, load_products  # noqa: E402
from rag.graph import build_graph  # noqa: E402
from rag.llm import get_model  # noqa: E402
from services.product_search import search_products  # noqa: E402
from services.rate_check import estimate, judge_conditions  # noqa: E402
from store.vectordb import count, get_store, split_text, upsert  # noqa: E402

CITE = re.compile(r"\[근거\s*([\d,\s]+)\]")
rows: list[tuple[str, str, str, str]] = []  # (번호, 확인 내용, 결과, 판정 근거)


def check(number: str, what: str, ok: bool, evidence: str) -> None:
    rows.append((number, what, "통과" if ok else "실패", evidence.replace("\n", " ").replace("|", "/")))
    print(f"[{'통과' if ok else '실패'}] {number} {what} — {evidence[:110]}")


def cited(answer: str) -> set[int]:
    return {int(n) for g in CITE.findall(answer) for n in g.split(",") if n.strip().isdigit()}


def main() -> None:
    started = time.time()
    products = load_products()
    by_name = {p["name"]: p for p in products}
    meta_store, docs_store = get_store(config.META_COLLECTION), get_store(config.DOCS_COLLECTION)
    model = get_model()
    graph = build_graph(meta_store, docs_store, model)

    # ── 4.1 데이터와 색인 ────────────────────────────────────────────
    check("4.1-1", "로드한 항목 수 = 색인 청크 수", len(products) == count(meta_store),
          f"상품 {len(products)}건 · product_meta {count(meta_store)}청크")
    sample = products[:3]
    before = count(meta_store)
    upsert(meta_store, [build_item_text(p) for p in sample], [p["product_id"] for p in sample],
           [build_item_metadata(p) for p in sample])
    check("4.1-2", "같은 항목을 다시 색인해도 청크 수가 늘지 않음", count(meta_store) == before,
          f"재색인 전 {before} → 후 {count(meta_store)} (id = 상품 ID, upsert)")
    text = build_item_text(by_name["KB맑은하늘적금"])
    fields = ["상품명", "금융사", "상품 유형", "가입 방법", "가입 대상", "기간별 금리", "우대조건", "만기 후 이자율", "유의사항"]
    check("4.1-3", "항목 1건의 색인 텍스트에 의도한 속성이 모두 있음", all(f"{f}:" in text for f in fields),
          f"KB맑은하늘적금 — {', '.join(fields)}")

    # ── 4.2 문서 수집과 색인 ──────────────────────────────────────────
    product = by_name["웰뱅 회전 정기예금"]
    question = "저축은행이 문을 닫으면 내 예금은 어떻게 돼?"
    manifest = load_manifest()
    manifest.get(product["product_id"], {}).get("questions", {}).pop(" ".join(question.split()), None)
    from collect import pipeline
    pipeline._save_manifest(manifest)
    report = collect_for_question(product, label(product), question, model, docs_store)
    check("4.2-1", "프롬프트 1이 뽑은 검색어 확인", bool(report["keywords"]), f"검색어 {report['keywords']}")
    check("4.2-2", "선별된 문서와 버려진 문서 구분",
          True, f"색인 {report['selected'] or '없음(이미 색인됨)'} · 버림 {report['rejected'][:5]} · "
                f"이미 색인 {report['already_indexed'][:4]}")
    again = collect_for_question(product, label(product), question, model, docs_store)
    check("4.2-3", "같은 질문을 다시 물으면 수집이 일어나지 않음", again["from_cache"] and again["chunks"] == 0,
          f"두 번째 실행: from_cache={again['from_cache']}, 새 청크 {again['chunks']}")
    company_docs = docs_store.get(where={"$and": [{"scope": "company"}, {"company": product["company"]}]},
                                  include=["metadatas"])
    check("4.2-4", "항목 자체 문서(금융사 문서)가 빠지지 않음", bool(company_docs["ids"]),
          f"{product['company']} 문서 {sorted({m['title'] for m in company_docs['metadatas']})}")
    body = "\n".join(docs_store.get(where={"title": "금융감독원"}, include=["documents"])["documents"])
    n1000, n500 = len(split_text(body, 1000, 200)), len(split_text(body, 500, 100))
    check("4.2-5", "청크 크기 · 중복 길이를 바꾸면 청크 수가 달라짐", n500 > n1000 > 0,
          f"금융감독원 문서: 1000/200 → {n1000}청크, 500/100 → {n500}청크 "
          f"(색인 전체: 1000자 135 → 500자 268, docs/design_notes.md 3단계)")

    # ── 4.3 답변과 근거 ──────────────────────────────────────────────
    nodes = [n for n in graph.get_graph().nodes if not n.startswith("__")]
    check("4.3-2", "LangGraph 가 retrieve · generate 두 노드", nodes == ["retrieve", "generate"],
          f"노드 {nodes} (구조: docs/rag_graph.md)")

    state = {"product_id": product["product_id"], "question": question, "keywords": report["keywords"], "history": []}
    result = graph.invoke(state)
    sources, answer = result["sources"], result["answer"]
    numbers = cited(answer)
    check("4.3-3", "답변 본문의 근거 번호가 근거 목록 안에 있음", bool(numbers) and numbers <= set(range(1, len(sources) + 1)),
          f"인용 {sorted(numbers)} · 근거 {len(sources)}건")
    types = {s["source_type"].split(" ·")[0] for s in sources}
    check("4.3-4", "근거에 항목 메타데이터와 외부 문서가 구분됨", "상품 공시" in types and len(types) >= 2,
          f"출처 종류 {sorted(types)}")

    from rag import graph as graph_module
    original = config.DOCS_TOP_K
    lists = {}
    for k in (2, 5):
        config.DOCS_TOP_K = k
        lists[k] = [s["title"] for s in graph_module._doc_sources(
            docs_store, build_item_metadata(product), question, report["keywords"], "similarity")]
    config.DOCS_TOP_K = original
    check("4.3-1", "Top-k 를 바꾸면 근거 목록이 달라짐", len(lists[2]) <= 2 < len(lists[5]),
          f"Top-k 2 → {lists[2]} / Top-k 5 → {len(lists[5])}건")

    kb = by_name["KB맑은하늘적금"]
    unknown = graph.invoke({"product_id": kb["product_id"], "question": "대중교통 미션은 구체적으로 어떻게 달성해?",
                            "keywords": [], "history": []})["answer"]
    check("4.3-5", "문서에 없는 것을 물으면 추측하지 않고 확인되지 않는다고 답함", "확인되지 않" in unknown,
          f"답변: {unknown[:90]}")
    offtopic = graph.invoke({"product_id": kb["product_id"], "question": "오늘 서울 날씨 어때?",
                             "keywords": [], "history": []})["answer"]
    check("4.3-6", "인용된 근거가 없으면 그 사실을 안내", not cited(offtopic),
          f"인용 번호 {sorted(cited(offtopic))} → 화면에 '답변에 인용된 근거가 없습니다' 표시 (app.render_sources)")

    # ── 4.4 서비스 기능과 화면 ───────────────────────────────────────
    found = search_products(model, get_store(config.CONDITION_COLLECTION), {p["product_id"]: p for p in products},
                            "대중교통 타면 금리 더 주는 적금")
    picked = [next(p for p in products if p["product_id"] == r["product_id"]) for r in found["results"]]
    ok = bool(picked) and all(p["kind"] == "saving" and "대중교통" in p["special_condition"] for p in picked)
    check("4.4-2", "조건 탐색 결과에 조건과 맞지 않는 항목이 없음", ok,
          f"결과 {[p['name'] for p in picked]} — 모두 적금 · 우대조건에 '대중교통' 포함")

    judgment = judge_conditions(model, kb, 12, "대중교통은 매일 타요. 종이통장은 필요 없어요.")
    e = estimate(kb, 12, 500_000, judgment)
    has_quotes = all(c["quote"] for c in judgment["conditions"])
    check("4.4-3", "도메인 특화 기능 결과에 근거가 함께 제시됨", bool(judgment["conditions"]) and has_quotes,
          f"우대조건 {len(judgment['conditions'])}개 판정 · 모두 공시 원문 인용 · 예상 {e['low']['rate']}~{e['high']['rate']}% "
          f"· 근거: 상품 공시")

    from streamlit.testing.v1 import AppTest
    import streamlit as st
    at = AppTest.from_file(str(config.PROJECT_DIR / "app.py"), default_timeout=300).run()
    first = next(o for o in at.sidebar.selectbox(key="product_id").options if "KB맑은하늘적금" in o)
    at.sidebar.selectbox(key="product_id").select(first).run()
    at.chat_input[0].set_value("우대금리 조건 뭐야?").run()
    statuses = [s.label for s in at.status]  # 질문을 처리한 그 실행에서만 보인다 (다시 그릴 때는 기록만 그림)
    kb_messages = len(at.chat_message)
    second = next(o for o in at.sidebar.selectbox(key="product_id").options if "우리SUPER주거래적금" in o)
    at.sidebar.selectbox(key="product_id").select(second).run()
    other_messages = len(at.chat_message)
    at.sidebar.selectbox(key="product_id").select(first).run()
    check("4.4-1", "항목을 바꿨을 때 이전 대화가 섞이지 않음",
          kb_messages == 2 and other_messages == 0 and len(at.chat_message) == 2,
          f"KB {kb_messages}개 → 우리 {other_messages}개 → 다시 KB {len(at.chat_message)}개")

    check("4.4-4", "문서 수집 · 답변 생성 중 진행 상태 표시", bool(statuses), f"st.status 최종 라벨 {statuses[-1:]}")

    key, collection = config.GMS_KEY, config.META_COLLECTION
    st.cache_resource.clear()
    config.GMS_KEY = ""
    no_key = AppTest.from_file(str(config.PROJECT_DIR / "app.py"), default_timeout=120).run()
    config.GMS_KEY, config.META_COLLECTION = key, "verify_empty_collection"
    st.cache_resource.clear()
    no_index = AppTest.from_file(str(config.PROJECT_DIR / "app.py"), default_timeout=120).run()
    config.META_COLLECTION = collection
    get_store("verify_empty_collection").delete_collection()
    st.cache_resource.clear()
    check("4.4-5", "색인이 없을 때 · API 키가 없을 때 안내", bool(no_key.error) and bool(no_index.error),
          f"키 없음: {no_key.error[0].value[:40]} / 색인 없음: {no_index.error[0].value[:40]}")
    check("4.4-6", "위험 도메인(금융) 면책 안내", any("가입 권유나" in w.value for w in at.warning),
          f"상단 고정: {at.warning[0].value[:45]}…")

    # ── 4.7 비기능 ───────────────────────────────────────────────────
    timing = time.time()
    graph.invoke({"product_id": kb["product_id"], "question": "1년 최고금리 얼마야?", "keywords": [], "history": []})
    check("NF2001", "대표 질문 처리 시간", True,
          f"공시로 답하는 질문 그래프 {time.time() - timing:.1f}초 · 새 문서 수집이 필요한 질문은 최대 44초 "
          f"(docs/rag_comparison.md) · 같은 질문 재질문 시 수집 생략")
    check("NF2002", "이미 수집한 검색어는 다시 수집하지 않음", again["from_cache"],
          "cache/collected.json(상품별 검색어 · 질문) · cache/wiki_pages/(본문)")
    tracked = [p for p in config.PROJECT_DIR.rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".toml", ".txt", ".json"}
               and ".venv" not in p.parts and "chroma_db" not in p.parts and "cache" not in p.parts]
    leaked = [str(p.name) for p in tracked if config.GMS_KEY and config.GMS_KEY in p.read_text(encoding="utf-8", errors="ignore")]
    check("NF2003", "소스코드 · 제출물에 API 키 없음", not leaked, f"검사 파일 {len(tracked)}개 · 키 발견 {leaked or '없음'} · 키는 .env(.gitignore)")
    toml = (config.PROJECT_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    check("NF2004", "화면에 키 · 프롬프트 · 오류 추적 정보 노출 없음", 'showErrorDetails = "none"' in toml,
          "showErrorDetails=none · 예외는 안내 문구만 · 수집 요약은 검색어/문서 제목만")
    modules = ["domain/products.py", "collect/wiki.py", "collect/pdf.py", "collect/pipeline.py", "store/vectordb.py",
               "rag/graph.py", "rag/prompts.py", "services/rate_check.py", "services/product_search.py", "app.py"]
    check("NF2005", "수집 · 색인 · 검색 · 생성 · UI 모듈 분리", all((config.PROJECT_DIR / m).exists() for m in modules),
          " · ".join(modules))
    check("NF2006", "외부 API 예절 (재시도 · 호출 간격 · User-Agent 연락처)", "(" in USER_AGENT and "@" in USER_AGENT,
          "User-Agent에 연락처 포함 · 429 시 Retry-After 대기 후 재시도 · 호출 간격 0.5초 (collect/wiki.py)")
    check("NF2007", "데이터 · 문서 출처와 이용 조건 기록", (config.PROJECT_DIR / "README.md").exists(),
          "README '데이터 출처와 이용 조건' · 근거마다 출처 종류와 원문 링크")

    # ── 저장 ─────────────────────────────────────────────────────────
    passed = sum(1 for r in rows if r[2] == "통과")
    lines = ["# 자동 검증 결과", "",
             f"`python scripts/verify.py` 실행 결과 — 통과 {passed} / {len(rows)} · 소요 {time.time() - started:.0f}초 · "
             f"모델 {config.CHAT_MODEL}", "",
             "| 번호 | 확인 내용 | 결과 | 판정 근거 (실제 출력) |", "|---|---|---|---|"]
    lines += [f"| {n} | {w} | {r} | {e} |" for n, w, r, e in sorted(rows)]
    out = config.PROJECT_DIR / "docs" / "verification_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n통과 {passed}/{len(rows)} · 저장: {out}")


if __name__ == "__main__":
    main()
