from fastapi import Request

from llm_lab.models.manager import ModelService


def model_service(request: Request) -> ModelService:
    return request.app.state.model_service
