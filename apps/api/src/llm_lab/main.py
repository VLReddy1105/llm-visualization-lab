import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llm_lab.api import health, inference, models
from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.models.manager import ModelService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, service: ModelService | None = None) -> FastAPI:
    configuration = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service = service or ModelService(configuration)
        if configuration.auto_load:
            try:
                app.state.model_service.start_loading()
            except ServiceError:
                logger.exception("Automatic model load could not be started")
        yield
        if service is None:
            await asyncio.to_thread(app.state.model_service.close)

    app = FastAPI(title="LLM Visualization Lab API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configuration.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_identity(request: Request, call_next):
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        headers = {"Retry-After": "2"} if exc.status_code == 429 else None
        return JSONResponse(
            status_code=exc.status_code,
            headers=headers,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "retryable": exc.retryable,
                    "request_id": request.state.request_id,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo the raw prompt or unserializable exception context.
        fields = [
            {"field": ".".join(map(str, error["loc"])), "message": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Check the prompt and top_k request fields.",
                    "details": {"fields": fields},
                    "retryable": False,
                    "request_id": request.state.request_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled API error for %s", request.state.request_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "The backend encountered an unexpected error.",
                    "details": {},
                    "retryable": False,
                    "request_id": request.state.request_id,
                }
            },
        )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(inference.router)
    return app


app = create_app()
