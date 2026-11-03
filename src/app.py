"""Main FastAPI application entrypoint for Cognigy Gemini Speech Adapter."""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routes import router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cognigy-gemini-speech")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting Cognigy Gemini Speech Adapter | project=%s | location=%s | transcribe_model=%s | auth=%s | log_level=%s",
        settings.project_id,
        settings.location,
        settings.transcribe_model,
        "ENABLED" if settings.cognigy_api_key else "DISABLED",
        settings.log_level.upper(),
    )
    yield
    logger.info("Shutting down Cognigy Gemini Speech Adapter")


app = FastAPI(
    title="Cognigy Gemini Speech Adapter",
    description="Cognigy Voice Gateway Custom Speech Adapter powered by Gemini 3.8 and Gemini Live",
    version="0.1.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming HTTP requests with timing and status codes."""
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    logger.info("[HTTP-IN] %s %s from %s", request.method, request.url.path, client_ip)
    
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "[HTTP-OUT] %s %s -> %d in %.1f ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "[HTTP-OUT] %s %s FAILED with %s after %.1f ms",
            request.method,
            request.url.path,
            e,
            elapsed_ms,
        )
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host=settings.host, port=settings.port, reload=True)
