"""금감원 금융상품한눈에 API 탐색 — 상품 수와 우대조건 원문을 눈으로 확인한다.

실행 (project 폴더에서):
    python scripts/probe_finlife.py
"""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

import config  # noqa: E402

API = "https://finlife.fss.or.kr/finlifeapi/{kind}ProductsSearch.json"
KINDS = {"deposit": "정기예금", "saving": "적금"}
GROUPS = {"020000": "은행", "030300": "저축은행"}


def fetch(kind: str, group: str, page: int = 1) -> dict:
    response = requests.get(
        API.format(kind=kind),
        params={"auth": config.FINLIFE_KEY, "topFinGrpNo": group, "pageNo": page},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()["result"]


def main() -> None:
    if not config.FINLIFE_KEY:
        raise SystemExit("FINLIFE_KEY가 .env 에 없습니다.")

    samples = {}
    for kind, kind_name in KINDS.items():
        for group, group_name in GROUPS.items():
            result = fetch(kind, group)
            if result.get("err_cd") != "000":
                print(f"[오류] {kind_name}/{group_name}: {result.get('err_cd')} {result.get('err_msg')}")
                continue
            print(f"{kind_name:4} · {group_name:4} — 상품 {result['total_count']:>4}건 · "
                  f"금리옵션 {len(result['optionList']):>4}건 · 페이지 {result['max_page_no']}")
            samples.setdefault(kind, result)
            time.sleep(0.3)

    # 적금 1건을 통째로 들여다본다 — 우대조건(spcl_cnd)이 기능 A에 쓸 만한지
    saving = samples.get("saving")
    if saving and saving["baseList"]:
        base = max(saving["baseList"], key=lambda b: len(b.get("spcl_cnd") or ""))
        options = [o for o in saving["optionList"] if o["fin_prdt_cd"] == base["fin_prdt_cd"]]
        print("\n── 적금 예시 (우대조건이 가장 긴 상품) — baseList 필드 ──")
        print(json.dumps(base, ensure_ascii=False, indent=2))
        print("── 같은 상품의 optionList (기간별 금리) ──")
        print(json.dumps(options[:3], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
