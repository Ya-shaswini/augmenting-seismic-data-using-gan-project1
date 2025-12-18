from fastapi import APIRouter
from . import auth, data, gan

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(data.router, prefix="/data", tags=["data"])
router.include_router(gan.router, prefix="/gan", tags=["gan"])
