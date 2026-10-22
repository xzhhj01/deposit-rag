"""0단계 점검 — 키 · LLM · 임베딩 · 데이터 경로가 모두 준비됐는지 확인한다.

실행 (project 폴더에서):
    python scripts/check_env.py

키 값은 출력하지 않는다. (NF2004)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 한글 · 특수문자 출력 오류 방지
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))  # project/ 를 import 경로에 추가

import config  # noqa: E402


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK' if ok else '실패'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    results = []

    # 1. 키
    results.append(check("GMS_KEY", bool(config.GMS_KEY), "설정됨" if config.GMS_KEY else ".env 에 없음"))
    results.append(check("WIKI_CONTACT", bool(config.WIKI_CONTACT),
                         "설정됨" if config.WIKI_CONTACT else "없음 — 위키 API가 분당 10회로 제한된다"))

    results.append(check("FINLIFE_KEY", bool(config.FINLIFE_KEY), "설정됨" if config.FINLIFE_KEY else ".env 에 없음"))

    if not config.GMS_KEY:
        print("\nGMS_KEY가 없어 LLM · 임베딩 점검을 건너뜁니다.")
        sys.exit(1)

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # 3. LLM 한 번 호출
    try:
        llm = ChatOpenAI(model=config.CHAT_MODEL, api_key=config.GMS_KEY, base_url=config.GMS_BASE_URL)
        answer = llm.invoke("한 단어로만 답해: 대한민국의 수도는?").content
        results.append(check(f"LLM ({config.CHAT_MODEL})", True, f"응답: {answer.strip()[:30]}"))
    except Exception as error:  # 점검 스크립트라 오류 종류만 보여 준다
        results.append(check(f"LLM ({config.CHAT_MODEL})", False, type(error).__name__))

    # 4. 임베딩 한 번 호출 — 문장이 숫자 벡터로 바뀌는 것을 본다
    try:
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL, api_key=config.GMS_KEY, base_url=config.GMS_BASE_URL
        )
        vector = embeddings.embed_query("우주를 배경으로 한 가족 영화")
        results.append(check(f"임베딩 ({config.EMBEDDING_MODEL})", True,
                             f"벡터 길이 {len(vector)} · 앞 3개 {[round(v, 4) for v in vector[:3]]}"))
    except Exception as error:
        results.append(check(f"임베딩 ({config.EMBEDDING_MODEL})", False, type(error).__name__))

    print("\n모두 통과" if all(results) else "\n실패한 항목을 확인하세요.")


if __name__ == "__main__":
    main()
