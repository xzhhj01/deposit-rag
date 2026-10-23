"""특화 기능 A — 내 조건 기준 실제 금리 판정 (F2010)

광고 금리("최고 연 3.45%")가 아니라, 내 상황에서 실제로 받을 금리와 세후 이자를 보여 준다.

    ① LLM  : 우대조건 원문을 항목으로 나누고, 고객 상황으로 항목마다 충족 / 미충족 / 확인 불가를 판정
    ② 코드 : 원문에 그 숫자가 실제로 있는지 검사 (LLM이 숫자를 지어내지 않았는지)
    ③ 코드 : 기본금리 + 충족 우대금리 = 예상 금리, 상한(cap · 공시 최고금리) 적용, 이자 · 세금 계산

판단(언어)은 LLM, 계산(숫자)은 코드가 맡는다. LLM은 덧셈 · 이자 계산을 자주 틀린다.
근거: 상품 공시(우대조건 원문 · 기간별 금리). 세율은 근거가 아니라 계산 가정(config.INTEREST_TAX_RATE).
"""

import re

from langchain_core.language_models import BaseChatModel

import config
from domain.products import split_conditions
from rag.parsing import parse_json_object
from rag.prompts import CONDITION_SYSTEM, build_condition_prompt

VERDICTS = ("충족", "미충족", "확인 불가")


def terms(product: dict) -> list[int]:
    return sorted({o["term"] for o in product["options"]})


def option_for(product: dict, term: int) -> dict | None:
    """가입기간의 금리 옵션. 자유 · 정액적립식이 둘 다 있으면 공시 순서상 첫 번째를 쓴다."""
    return next((o for o in product["options"] if o["term"] == term), None)


def _rate_in_text(rate: float, text: str) -> bool:
    """LLM이 읽어 온 우대금리 숫자가 원문에 실제로 있는지. 0.4 ↔ '0.4%p', '0.40%', '.4%' 모두 허용."""
    numbers = {float(n) for n in re.findall(r"\d*\.?\d+", text.replace(",", ""))}
    return any(abs(rate - n) < 1e-9 for n in numbers)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _source_quote(quote: str, name: str, original: str) -> str:
    """화면에 '공시 원문'으로 보여 줄 구절. LLM이 준 구절이 원문에 그대로 있을 때만 쓴다.

    LLM은 원문 대신 고객이 입력한 문장("종이통장은 필요 없어요")을 인용하기도 했다 → 근거가 아닌 것이 근거처럼 보인다.
    원문에 없으면 조건 이름이 들어 있는 원문 줄로 바꾸고, 그것도 없으면 비워 둔다.
    """
    if quote and _squash(quote) in _squash(original):
        return quote.strip()
    key = _squash(name)[:6]
    for line in split_conditions({"special_condition": original}):
        if key and key in _squash(line):
            return line[:80]
    return ""


def judge_conditions(model: BaseChatModel, product: dict, term: int, situation: str) -> dict:
    """LLM 판정 + 숫자 검사. {'cap', 'conditions': [...]} — 조건마다 verified(원문에 숫자 있음) 표시."""
    answer = model.invoke([
        {"role": "system", "content": CONDITION_SYSTEM},
        {"role": "user", "content": build_condition_prompt(product["special_condition"], term, situation)},
    ]).content
    result = parse_json_object(answer)

    conditions = []
    for c in result.get("conditions", []):
        if not isinstance(c, dict) or not c.get("name"):
            continue
        rate = c.get("rate")
        rate = float(rate) if isinstance(rate, (int, float)) else None
        conditions.append({
            "name": str(c["name"]),
            "rate": rate,
            "verdict": c.get("verdict") if c.get("verdict") in VERDICTS else "확인 불가",
            "reason": str(c.get("reason") or ""),
            "how_to": str(c.get("how_to") or ""),
            "quote": _source_quote(str(c.get("quote") or ""), str(c["name"]), product["special_condition"]),
            # 원문에 없는 숫자는 계산에 넣지 않는다 — 근거 없는 숫자로 금리를 부풀리지 않게
            "verified": rate is not None and _rate_in_text(rate, product["special_condition"]),
        })
    cap = result.get("cap")
    cap = float(cap) if isinstance(cap, (int, float)) and _rate_in_text(float(cap), product["special_condition"]) else None
    return {"cap": cap, "conditions": conditions}


# ── 계산 (코드) ──────────────────────────────────────────────────────
def interest_before_tax(kind: str, rate_type: str, amount: int, months: int, annual_rate: float) -> float:
    """세전 이자 (원). 적금은 매월 같은 금액을 납입한다고 가정한다.

    적금 단리 : 1회차 돈은 n개월, 2회차 돈은 n-1개월 … 이자가 붙는다 → 월납입액 × 연이율/12 × n(n+1)/2
    적금 복리 : 회차마다 월복리 — Σ 월납입액 × ((1 + 연이율/12)^남은개월 - 1)
    예금 단리 : 예치금 × 연이율 × 개월/12
    예금 복리 : 예치금 × ((1 + 연이율/12)^개월 - 1)
    """
    r = annual_rate / 100
    if kind == "saving":
        if rate_type == "복리":
            return sum(amount * ((1 + r / 12) ** k - 1) for k in range(1, months + 1))
        return amount * r / 12 * months * (months + 1) / 2
    if rate_type == "복리":
        return amount * ((1 + r / 12) ** months - 1)
    return amount * r * months / 12


def estimate(product: dict, term: int, amount: int, judgment: dict) -> dict:
    """판정 결과로 예상 금리 범위와 세후 이자를 계산한다."""
    option = option_for(product, term)
    base, max_rate = option["base_rate"], option["max_rate"]
    conditions = judgment["conditions"]

    def bonus(verdicts: tuple[str, ...]) -> float:
        return sum(c["rate"] for c in conditions if c["verdict"] in verdicts and c["verified"])

    # 상한: 원문의 우대 합계 상한(cap)과 공시 최고금리 중 더 엄격한 쪽
    bonus_cap = max_rate - base
    if judgment["cap"] is not None:
        bonus_cap = min(bonus_cap, judgment["cap"])

    low_bonus = min(bonus(("충족",)), bonus_cap)                  # 확실히 받는 우대
    high_bonus = min(bonus(("충족", "확인 불가")), bonus_cap)     # 확인 불가 조건까지 다 받으면
    low, high = round(base + low_bonus, 2), round(base + high_bonus, 2)

    principal = amount * term if product["kind"] == "saving" else amount
    rate_type = option["rate_type"]

    def money(rate: float) -> dict:
        before = interest_before_tax(product["kind"], rate_type, amount, term, rate)
        tax = before * config.INTEREST_TAX_RATE
        return {"rate": rate, "before_tax": round(before), "tax": round(tax), "after_tax": round(before - tax)}

    return {
        "term": term, "amount": amount, "principal": principal, "rate_type": rate_type,
        "base_rate": base, "advertised_max": max_rate,
        "low": money(low), "high": money(high), "max": money(max_rate),
        "capped": bonus(("충족", "확인 불가")) > bonus_cap + 1e-9,
        "unverified": [c["name"] for c in conditions if not c["verified"]],
    }
