from fastapi import APIRouter
from . import auth, data, gan, data_sources

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(data.router, prefix="/data", tags=["data"])
router.include_router(gan.router, prefix="/gan", tags=["gan"])
router.include_router(data_sources.router, prefix="/data-sources", tags=["data-sources"])
