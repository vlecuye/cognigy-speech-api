"""Main FastAPI application entrypoint for Cognigy Gemini Speech Adapter."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routes import router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cognigy-gemini-speech")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting Cognigy Gemini Speech Adapter (project=%s, location=%s, auth=%s)",
        settings.project_id,
        settings.location,
        "ENABLED" if settings.cognigy_api_key else "DISABLED",
    )
    yield
    logger.info("Shutting down Cognigy Gemini Speech Adapter")


app = FastAPI(
    title="Cognigy Gemini Speech Adapter",
    description="Cognigy Voice Gateway Custom Speech Adapter powered by Gemini 3.8 and Gemini Live",
    version="0.1.0",
    lifespan=lifespan,
)

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
