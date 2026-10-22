"""외부 문서 수집 — 한국어 위키백과 API (streamlit_lab/step8_wiki_api.py 에서 옮겨 왔다)

    search()      검색어 → 후보 문서 제목 · 요약 (본문 없음)
    fetch_page()  제목 → 본문 · 원문 URL. 받은 문서는 cache/wiki_pages/ 에 저장해 다시 받지 않는다. (NF2002)

API 예절 (NF2006)
    · User-Agent 에 연락처(WIKI_CONTACT)를 넣는다 — 없으면 분당 10회, 있으면 200회
    · 429 를 받으면 서버가 알려 준 시간만큼 기다렸다가 다시 시도한다
    · 호출 사이에 CALL_INTERVAL 만큼 쉰다

출처: 한국어 위키백과 — CC BY-SA 4.0. 근거에 문서 제목과 원문 링크를 함께 표시한다. (NF2007)
"""

import hashlib
import html
import json
import re
import time
from datetime import date

import requests

import config

API_URL = "https://ko.wikipedia.org/w/api.php"
SOURCE_TYPE = "위키백과"
CALL_INTERVAL = 0.5  # 초
USER_AGENT = (
    f"RateLens-SSAFY/0.1 ({config.WIKI_CONTACT}) python-requests"
    if config.WIKI_CONTACT else "RateLens-SSAFY/0.1 python-requests"
)
PAGE_CACHE_DIR = config.CACHE_DIR / "wiki_pages"

# 본문 끝의 참고용 절은 근거로 쓸 내용이 없어 잘라낸다.
TAIL_SECTIONS = re.compile(r"\n\n\n(같이 보기|각주|참고 문헌|외부 링크|출처)\n")

_last_call = 0.0


def _request(params: dict, max_retries: int = 4) -> dict:
    global _last_call
    wait = 2.0
    for _ in range(max_retries):
        time.sleep(max(0.0, CALL_INTERVAL - (time.time() - _last_call)))
        _last_call = time.time()
        response = requests.get(
            API_URL, params={**params, "format": "json", "formatversion": 2},
            headers={"User-Agent": USER_AGENT}, timeout=20,
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After") or wait))
            wait *= 2
            continue
        response.raise_for_status()
    raise RuntimeError("위키백과 요청 제한이 계속되어 실패했습니다.")


def search(query: str, limit: int = 5) -> list[dict]:
    """검색어로 후보 문서를 찾는다. 본문은 주지 않고 제목과 짧은 요약만 준다."""
    data = _request({"action": "query", "list": "search", "srsearch": query, "srlimit": limit})
    return [
        {
            "page_id": item["pageid"],
            "title": item["title"],
            "snippet": html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", ""))),
        }
        for item in data["query"]["search"]
    ]


def _cache_path(title: str):
    return PAGE_CACHE_DIR / f"{hashlib.sha1(title.encode('utf-8')).hexdigest()[:16]}.json"


def fetch_page(title: str) -> dict | None:
    """제목으로 본문과 원문 URL을 받는다. 없는 문서면 None. 캐시에 있으면 API를 부르지 않는다."""
    path = _cache_path(title)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached or None  # 빈 dict = '없는 문서'였다는 기록

    data = _request({
        "action": "query", "prop": "extracts|info", "titles": title,
        "explaintext": 1, "inprop": "url", "redirects": 1,
    })
    page = data["query"]["pages"][0]
    result = {} if page.get("missing") else {
        "doc_id": f"wiki-{page['pageid']}",
        "title": page["title"],
        "url": page["fullurl"],
        "text": TAIL_SECTIONS.split(page.get("extract", ""), maxsplit=1)[0].strip(),
        "source_type": SOURCE_TYPE,
        "collected_at": date.today().isoformat(),
    }
    PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result or None
