"""LLM 클라이언트 — 생성 영역에서 함께 쓴다. 키가 없으면 None 을 돌려주고, 안내는 UI가 한다."""

from langchain_openai import ChatOpenAI

import config


def get_model() -> ChatOpenAI | None:
    if not config.GMS_KEY:
        return None
    return ChatOpenAI(model=config.CHAT_MODEL, api_key=config.GMS_KEY, base_url=config.GMS_BASE_URL)
