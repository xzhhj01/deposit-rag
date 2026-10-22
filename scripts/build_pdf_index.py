"""PDF 문서 색인 (F2016) — data/pdf/*.pdf → finance_docs 컬렉션 (scope=common)

실행 (project 폴더에서):
    python scripts/build_pdf_index.py

PDF 는 저장소에 없다 (각 기관 저작권). 내려받을 곳은 domain/products.py 의 PDF_SOURCES 와 README 참고.
위키 문서와 같은 컬렉션에 넣고, 근거에는 출처 종류("위키백과" / "PDF · 예금보험공사" …)를 구분해 표시한다.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from collect.pdf import PDF_DIR, load_pdf_documents  # noqa: E402
from domain.products import PDF_SOURCES  # noqa: E402
from store.vectordb import count, get_store, index_documents  # noqa: E402


def main() -> None:
    store = get_store(config.DOCS_COLLECTION)
    print(f"청크 크기 {config.CHUNK_SIZE} · 중복 {config.CHUNK_OVERLAP} · 색인 전 {count(store)}청크\n")

    for spec in PDF_SOURCES:
        docs = load_pdf_documents(spec)
        if not docs:
            print(f"✗ {spec['file']} — {PDF_DIR} 에 없음. {spec['url']} 에서 내려받으세요.")
            continue
        chunks = index_documents(store, [{**doc, "scope": "common"} for doc in docs])
        unit = "용어 항목" if spec["chunking"] == "glossary" else "쪽"
        print(f"✓ {spec['name']} ({spec['publisher']}) — {unit} {len(docs)}개 → {chunks}청크")

    print(f"\n컬렉션 전체 {count(store)}청크")


if __name__ == "__main__":
    main()
