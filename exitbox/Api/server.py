"""
FastAPI application composition root.

This file now acts more like Program.cs in ASP.NET Core:
- create the app
- configure middleware
- register routes
- run startup wiring
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Api.routes.query_routes import router as query_router
from Api.routes.system_routes import router as system_router
from Api.services.runtime_service import runtime_service
from Core.app_text import app_text


def create_app() -> FastAPI:
    app = FastAPI(
        title=app_text["api"]["title"],
        description=app_text["api"]["description"],
        version=app_text["api"]["version"],
        docs_url="/docs",
        redoc_url="/redoc",
    )

    cors_origins = [
        origin.strip()
        for origin in os.getenv("OMNISSIAH_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        runtime_service.startup()

    app.include_router(system_router)
    app.include_router(query_router)
    return app


app = create_app()
