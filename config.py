"""실행 설정 — 경로 · 모델 · 청킹 · 검색 값을 한곳에 모은다.

품질에 영향을 주는 값(청크 크기, 중복 길이, Top-k, 검색 방식)은 코드가 아니라 여기서 바꾼다.
(F2005 · F2006 · 요구사항정의서 2.5 검색 설정 설계)
값을 바꿔 다시 색인 · 검색하면서 근거가 어떻게 달라지는지 비교하는 것이 검증 단계의 일이다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── 경로 ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")  # 어느 폴더에서 실행해도 project/.env 를 읽는다.

DATA_DIR = PROJECT_DIR / "data"
PRODUCTS_PATH = DATA_DIR / "products.json"  # RAG 1 — 예·적금 상품 메타데이터 (금감원 API 수집본)

CHROMA_DIR = PROJECT_DIR / "chroma_db"  # Vector DB 저장 위치
CACHE_DIR = PROJECT_DIR / "cache"       # 수집 캐시(이미 수집한 검색어) 위치 (NF2002)

# ── 키 · 외부 호출 ───────────────────────────────────────────────────
# 키는 환경변수로만 읽는다. 값을 코드에 쓰지 않는다. (NF2003)
GMS_KEY = os.getenv("GMS_KEY", "").strip()
GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"
WIKI_CONTACT = os.getenv("WIKI_CONTACT", "").strip()
FINLIFE_KEY = os.getenv("FINLIFE_KEY", "").strip()  # 금융감독원 금융상품한눈에 Open API

# ── 모델 ─────────────────────────────────────────────────────────────
CHAT_MODEL = "gpt-5.4-nano"
EMBEDDING_MODEL = "text-embedding-3-small"  # 0단계에서 GMS 동작 확인 (1536차원)

# ── 컬렉션 ───────────────────────────────────────────────────────────
META_COLLECTION = "product_meta"   # 상품 1개 = 1청크, id = 상품 ID
DOCS_COLLECTION = "finance_docs"   # 외부 문서(위키 · PDF) 청크, id = 상품 ID:문서 ID:순번

# ── 청킹 (F2005) ─────────────────────────────────────────────────────
# 실험할 때는 파일을 고치지 않고 환경변수로 덮어쓸 수 있다.  예) CHUNK_SIZE=500 python scripts/build_docs_index.py
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))     # 글자 수
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # 앞 청크와 겹치는 글자 수 — 문장이 경계에서 잘려도 의미가 이어지게

# ── 검색 (F2006 · F2014) ─────────────────────────────────────────────
DOCS_TOP_K = int(os.getenv("DOCS_TOP_K", 5))  # 외부 문서 청크를 몇 개 가져올지 (선택 상품의 메타 청크 1개는 항상 따로 포함)
# 코사인 거리가 이보다 먼 청크는 근거에서 뺀다. 실험: 관련 문서 0.43~0.58, 무관한 문서 0.59~0.78 (docs/design_notes.md)
DOCS_MAX_DISTANCE = float(os.getenv("DOCS_MAX_DISTANCE", 0.65))
SEARCH_TYPE = "similarity"  # "similarity" 또는 "mmr"
MMR_FETCH_K = 30        # MMR일 때 먼저 넉넉히 가져올 후보 수
