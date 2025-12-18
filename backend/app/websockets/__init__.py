from fastapi import APIRouter
from . import gateway

router = APIRouter()
router.include_router(gateway.router, tags=["websockets"])
