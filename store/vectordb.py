"""Vector Store 영역 — Chroma 컬렉션 생성 · 조회 · 색인 (F2002 · F2005)

도메인을 모른다. 텍스트 · id · 메타데이터만 받아 저장하고 검색한다. (NF2005)
"""

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# 외부 문서 청크에 함께 저장하는 메타데이터 — 근거 표시(F2008)와 검색 범위 필터(F2006)에 쓴다.
DOC_METADATA_FIELDS = ("doc_id", "title", "url", "source_type", "scope", "company", "product_id",
                       "search_query", "collected_at")


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


# ── 외부 문서 (F2005) ────────────────────────────────────────────────
def scope_key(doc: dict) -> str:
    """이 문서를 '누구의 근거로' 색인하는지. 청크 id 앞부분이 된다.

    같은 위키 문서라도 공통 개념으로 한 번, 특정 상품의 근거로 한 번 들어갈 수 있다.
    scope_key 가 다르면 청크 id 가 달라 서로 덮어쓰지 않는다.
    """
    if doc["scope"] == "common":
        return "common"
    if doc["scope"] == "company":
        return f"company:{doc['company']}"
    return doc["product_id"]


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """문단 → 줄 → 문장 → 글자 순으로 자를 곳을 찾아 chunk_size 이하로 나눈다. 값은 config 에서 온다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def index_document(store: Chroma, doc: dict) -> int:
    """문서 1건을 청킹해 색인한다. 저장한 청크 수를 돌려준다.

    청크 id = {scope_key}:{doc_id}:{순번}   예) common:wiki-12345:0, company:국민은행:wiki-678:3
    다시 색인할 때는 같은 문서의 이전 청크를 먼저 지운다 — 청크 크기를 줄였다 늘렸다 하면
    순번 개수가 달라져, 지우지 않으면 예전 청크가 남아 섞인다.
    """
    key = scope_key(doc)
    store._collection.delete(where={"$and": [{"doc_id": doc["doc_id"]}, {"scope_key": key}]})

    chunks = split_text(doc["text"])
    metadata = {field: doc.get(field) or "" for field in DOC_METADATA_FIELDS}
    metadata["scope_key"] = key
    upsert(
        store,
        texts=[f"{doc['title']}\n{chunk}" for chunk in chunks],  # 청크마다 제목을 붙여 무슨 문서의 조각인지 알게 한다
        ids=[f"{key}:{doc['doc_id']}:{i}" for i in range(len(chunks))],
        metadatas=[{**metadata, "chunk_index": i} for i in range(len(chunks))],
    )
    return len(chunks)
