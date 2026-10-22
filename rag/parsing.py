"""LLM 출력 파싱 — 앞뒤에 설명이 붙거나 코드 블록(```json)에 싸여 와도 JSON 값을 꺼낸다. 실패하면 빈 값.

설명 문장 안에 "[P1]" 같은 괄호가 먼저 나와도 되도록, 여는 괄호 위치마다 JSON 해석을 시도해
원하는 타입으로 처음 해석되는 값을 돌려준다.
"""

import json
import re

_decoder = json.JSONDecoder()
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _find(text: str, opener: str, kind: type):
    # 코드 블록이 있으면 그 안부터 본다
    for block in _FENCE.findall(text):
        value = _find(block, opener, kind) if "```" not in block else None
        if isinstance(value, kind):
            return value
    for match in re.finditer(re.escape(opener), text):
        try:
            value, _ = _decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, kind):
            return value
    return None


def parse_json_array(text: str) -> list:
    return _find(text, "[", list) or []


def parse_json_object(text: str) -> dict:
    return _find(text, "{", dict) or {}
