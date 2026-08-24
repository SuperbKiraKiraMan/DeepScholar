"""
app/main.py

FastAPI 应用入口 —— 类比 Spring Boot 的 @SpringBootApplication 主类。

Uvicorn ≈ 内置 Tomcat：负责启动 HTTP 服务器，把请求交给 FastAPI 处理。
启动命令：uvicorn app.main:app --reload
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.core.config import config
from app.mcp.manager import mcp_manager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application startup and shutdown lifecycle."""
    await mcp_manager.initialize()
    print(f"[startup] {config.APP_NAME} v{config.APP_VERSION} started.")
    print(f"[startup] Swagger UI: http://{config.HOST}:{config.PORT}/docs")
    print(f"[startup] Dashboard: http://{config.HOST}:{config.PORT}/")
    try:
        yield
    finally:
        await mcp_manager.shutdown()
        print("[shutdown] Server shutting down.")


# ---- 创建 FastAPI 应用实例 ----
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="An Agent-first academic research copilot that plans, searches, "
                "extracts evidence, checks citations, evaluates quality, and generates "
                "cited research reports.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---- CORS 中间件 ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 静态文件（Frontend Dashboard） ----
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")


@app.get("/")
async def serve_dashboard():
    """Serve the Frontend Dashboard at /."""
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "web")
    index_path = os.path.join(web_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-store, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "Dashboard not found. Visit /docs for API documentation."}


# ---- 注册路由 ----
app.include_router(api_router)
