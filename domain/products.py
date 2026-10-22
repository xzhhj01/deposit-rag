"""도메인 데이터 준비 — 예·적금 상품 (F2001)

금융에 묶인 코드는 이 파일에만 둔다. 다른 도메인으로 바꾸면 이 파일만 갈아 끼운다. (NF2005)

    fetch_products()       금감원 API에서 예금 · 적금 전체를 받아 상품 단위로 정리
    save / load_products() data/products.json 으로 저장 · 읽기
    build_item_text()      상품 1개 → 검색에 쓸 텍스트 1개
    build_item_metadata()  상품 1개 → Vector DB에 함께 저장할 메타데이터 (필터 · 화면 표시용)

출처: 금융감독원 금융상품한눈에 (https://finlife.fss.or.kr) — 공시 자료, 매월 갱신 (NF2007)
"""

import json
import time
from datetime import date

import requests

import config

API_URL = "https://finlife.fss.or.kr/finlifeapi/{kind}ProductsSearch.json"
SOURCE_NAME = "금융감독원 금융상품한눈에"
SOURCE_URL = "https://finlife.fss.or.kr"

KINDS = {"deposit": "정기예금", "saving": "적금"}
GROUPS = {"020000": "은행", "030300": "저축은행"}
JOIN_DENY = {"1": "제한 없음", "2": "서민전용", "3": "일부 제한"}

# 우대조건 칸에 "없음"류만 적힌 경우 — 우대조건이 없는 상품으로 본다.
NO_CONDITION = {"", "없음", "해당없음", "해당사항없음", "우대조건없음", "-"}


# ── 수집 ─────────────────────────────────────────────────────────────
def _request(kind: str, group: str, page: int, max_retries: int = 3) -> dict:
    """API 한 페이지를 받는다. 네트워크 오류는 간격을 늘려 가며 다시 시도한다. (NF2006)"""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                API_URL.format(kind=kind),
                params={"auth": config.FINLIFE_KEY, "topFinGrpNo": group, "pageNo": page},
                timeout=20,
            )
            response.raise_for_status()
            result = response.json()["result"]
            if result.get("err_cd") != "000":
                raise RuntimeError(f"API 오류 {result.get('err_cd')}: {result.get('err_msg')}")
            return result
        except requests.RequestException:
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _clean(value) -> str:
    return (value or "").strip()


def _has_condition(text: str) -> bool:
    return text.replace(" ", "").replace("\n", "") not in NO_CONDITION


def fetch_products() -> list[dict]:
    """예금 · 적금 × 은행 · 저축은행 전체를 받아, 상품 하나에 기간별 금리를 묶어 돌려준다."""
    products = []
    collected_at = date.today().isoformat()

    for kind, kind_name in KINDS.items():
        for group, group_name in GROUPS.items():
            bases, options, page = [], [], 1
            while True:
                result = _request(kind, group, page)
                bases += result["baseList"]
                options += result["optionList"]
                if page >= int(result["max_page_no"]):
                    break
                page += 1
                time.sleep(0.3)  # 호출 간격 (NF2006)

            # optionList 는 기간별 금리 행이 따로 온다 → 상품 코드로 묶는다.
            options_by_product: dict[str, list] = {}
            for o in options:
                options_by_product.setdefault((o["fin_co_no"], o["fin_prdt_cd"]), []).append({
                    "term": int(o["save_trm"]),
                    "rate_type": o["intr_rate_type_nm"],          # 단리 / 복리
                    "reserve_type": o.get("rsrv_type_nm"),        # 적금만: 정액적립식 / 자유적립식
                    "base_rate": o["intr_rate"],                  # 기본금리
                    "max_rate": o["intr_rate2"],                  # 최고우대금리
                })

            for b in bases:
                special = _clean(b["spcl_cnd"])
                products.append({
                    # 같은 상품 코드가 다른 금융사 · 유형에서 겹쳐도 충돌하지 않게 셋을 묶는다.
                    "product_id": f"{kind}:{b['fin_co_no']}:{b['fin_prdt_cd']}",
                    "kind": kind,
                    "kind_name": kind_name,
                    "group": group_name,
                    "company": b["kor_co_nm"],
                    "name": b["fin_prdt_nm"],
                    "join_way": _clean(b["join_way"]),
                    "join_member": _clean(b["join_member"]),
                    "join_deny": JOIN_DENY.get(b["join_deny"], "확인 필요"),
                    "special_condition": special if _has_condition(special) else "",
                    "maturity_interest": _clean(b["mtrt_int"]),
                    "etc_note": _clean(b["etc_note"]),
                    "max_limit": b["max_limit"],
                    "options": sorted(
                        options_by_product.get((b["fin_co_no"], b["fin_prdt_cd"]), []),
                        key=lambda o: o["term"],
                    ),
                    "disclosed_month": b["dcls_month"],
                    "collected_at": collected_at,
                    "source": SOURCE_NAME,
                    "url": SOURCE_URL,
                })
            time.sleep(0.3)
    return products


def save_products(products: list[dict]) -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    config.PRODUCTS_PATH.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")


def load_products() -> list[dict]:
    """저장된 상품 목록을 읽는다. 없으면 빈 목록 (화면에서 '수집 스크립트를 실행하라' 안내에 쓴다)."""
    if not config.PRODUCTS_PATH.exists():
        return []
    return json.loads(config.PRODUCTS_PATH.read_text(encoding="utf-8"))


# ── 색인 텍스트 (F2001) ───────────────────────────────────────────────
def label(product: dict) -> str:
    """화면 목록에 보일 이름. 예) KB맑은하늘적금 · 국민은행 (적금)"""
    return f"{product['name']} · {product['company']} ({product['kind_name']})"


def _format_rates(options: list[dict]) -> str:
    if not options:
        return "공시된 금리 없음"
    return " / ".join(
        f"{o['term']}개월 기본 {o['base_rate']}% · 최고 {o['max_rate']}%"
        + (f" ({o['reserve_type']})" if o.get("reserve_type") else "")
        for o in options
    )


def build_item_text(product: dict) -> str:
    """상품 1개를 검색용 텍스트 1개로 만든다.

    넣는 기준 — 사용자가 무엇으로 상품을 찾을까?
      · 이름 · 금융사 · 유형        "국민은행 적금"
      · 가입 방법 · 대상             "스마트폰으로 가입", "청년 전용"
      · 우대조건 원문                "급여이체 없이 우대받는", "대중교통 이용하면"  ← 의미 검색의 핵심
      · 만기 후 이자 · 유의사항       상품 질의응답에서 근거로 쓰인다
    금리 숫자도 넣지만, 숫자 크기 비교("3% 넘는")는 임베딩이 못 한다 → 메타데이터 필터로 따로 처리한다.
    """
    rate_types = sorted({o["rate_type"] for o in product["options"]})
    lines = [
        f"상품명: {product['name']}",
        f"금융사: {product['company']} ({product['group']})",
        f"상품 유형: {product['kind_name']}" + (f" ({', '.join(rate_types)})" if rate_types else ""),
        f"가입 방법: {product['join_way'] or '정보 없음'}",
        f"가입 대상: {product['join_member'] or '정보 없음'} (가입 제한: {product['join_deny']})",
        f"기간별 금리: {_format_rates(product['options'])}",
        f"우대조건: {product['special_condition'] or '없음'}",
        f"만기 후 이자율: {product['maturity_interest'] or '정보 없음'}",
        f"유의사항: {product['etc_note'] or '없음'}",
    ]
    if product["max_limit"]:
        lines.append(f"가입 한도: {product['max_limit']:,}원")
    return "\n".join(lines)


def build_item_metadata(product: dict) -> dict:
    """Vector DB에 텍스트와 함께 저장할 값. Chroma 메타데이터는 str · int · float · bool 만 된다.

    숫자 필드(best_max_rate 등)는 "최고금리 3% 이상" 같은 조건을 필터로 거르는 데 쓴다. (F2011)
    """
    options = product["options"]
    return {
        "product_id": product["product_id"],
        "name": product["name"],
        "company": product["company"],
        "kind": product["kind"],
        "group": product["group"],
        "best_base_rate": max((o["base_rate"] or 0 for o in options), default=0.0),
        "best_max_rate": max((o["max_rate"] or 0 for o in options), default=0.0),
        "has_special_condition": bool(product["special_condition"]),
        "terms": ",".join(str(o["term"]) for o in options),
        "join_deny": product["join_deny"],
        "disclosed_month": product["disclosed_month"],
        "source": product["source"],
        "url": product["url"],
    }
