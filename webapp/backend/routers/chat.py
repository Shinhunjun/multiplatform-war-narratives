"""Chat endpoint."""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..services.llm_service import chat

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    answer: str


@router.post("", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    answer = chat(req.question, req.history)
    return ChatResponse(answer=answer)
