"""FastAPI Application Factory and Lifespan Management."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rice_ai.api.routes import router
from rice_ai.models.regression_loader import LoadedRegressionProvider
from rice_ai.models.vision_models import VisionModelProvider
from rice_ai.pipeline.runner import RicePipeline
from rice_ai.settings import Settings

logger = logging.getLogger("rice_ai.api.application")


def create_app(
    settings: Optional[Settings] = None,
    pipeline: Optional[RicePipeline] = None,
    vision_provider: Optional[VisionModelProvider] = None,
    regression_provider: Optional[LoadedRegressionProvider] = None,
) -> FastAPI:
    """Tạo và cấu hình ứng dụng FastAPI hoàn chỉnh."""
    app_settings = settings or Settings()
    vis_prov = vision_provider or VisionModelProvider(app_settings)
    reg_prov = regression_provider or LoadedRegressionProvider(app_settings)
    pipe = pipeline or RicePipeline(app_settings, vis_prov, reg_prov)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Khởi tạo semaphore giới hạn inference đồng thời
        app.state.admission_semaphore = asyncio.Semaphore(app_settings.max_concurrent_inferences)
        app.state.pending_workers = set()

        # 2. Thử nạp trước mô hình hồi quy tại thời điểm bootstrap
        try:
            reg = reg_prov.get_regression()
            logger.info(f"Lifespan: Đã nạp thành công mô hình hồi quy '{reg.model_name}'")
        except Exception as ex:
            logger.warning(f"Lifespan: Chưa thể nạp mô hình hồi quy ({ex}). Dịch vụ chuyển sang degraded mode.")

        yield

        # 3. Dọn dẹp khi shutdown: chờ pending workers
        pending = getattr(app.state, "pending_workers", set())
        if pending:
            logger.info(f"Lifespan shutdown: Đang chờ {len(pending)} pending worker(s) hoàn tất...")
            await asyncio.gather(*list(pending), return_exceptions=True)
        logger.info("Lifespan: Ứng dụng Rice AI đã tắt an toàn.")

    app = FastAPI(
        title="Rice Vision AI Inference API",
        description="Dịch vụ AI ước lượng số lượng và đánh giá chất lượng hạt giống lúa",
        version="2.2.0",
        lifespan=lifespan,
    )

    # Khởi tạo trạng thái ứng dụng đồng bộ
    app.state.settings = app_settings
    app.state.pipeline = pipe
    app.state.vision_provider = vis_prov
    app.state.regression_provider = reg_prov
    app.state.admission_semaphore = asyncio.Semaphore(app_settings.max_concurrent_inferences)
    app.state.pending_workers = set()

    # Cấu hình CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lưu trữ dependencies vào app.state
    app.state.settings = app_settings
    app.state.pipeline = pipe
    app.state.vision_provider = vis_prov
    app.state.regression_provider = reg_prov

    # Đăng ký routes
    app.include_router(router)

    return app
