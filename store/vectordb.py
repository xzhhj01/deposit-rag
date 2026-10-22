"""Vector Store 영역 — Chroma 컬렉션 생성 · 조회 · 색인 (F2002 · F2005)

도메인을 모른다. 텍스트 · id · 메타데이터만 받아 저장하고 검색한다. (NF2005)
"""

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import config


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL, api_key=config.GMS_KEY, base_url=config.GMS_BASE_URL
    )


def get_store(collection_name: str) -> Chroma:
    """컬렉션을 연다. 없으면 만든다. 디스크(chroma_db/)에 저장되므로 앱을 껐다 켜도 남는다."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        collection_metadata={"hnsw:space": "cosine"},  # 거리 = 1 - 코사인 유사도
    )


def count(store: Chroma) -> int:
    """컬렉션에 저장된 청크 수. 화면 왼쪽 '색인 건수'와 검증(4.1)에 쓴다."""
    return store._collection.count()


def upsert(store: Chroma, texts: list[str], ids: list[str], metadatas: list[dict],
           batch_size: int = 200) -> None:
    """같은 id 가 있으면 덮어쓰고, 없으면 새로 넣는다 → 몇 번을 다시 실행해도 중복이 생기지 않는다.

    langchain_chroma 의 add_texts 는 내부에서 Chroma 의 upsert 를 호출한다.
    한 번에 너무 많이 보내지 않도록 batch_size 씩 나눈다 (임베딩 API 요청 크기 제한).
    """
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        store.add_texts(texts=texts[start:end], ids=ids[start:end], metadatas=metadatas[start:end])
