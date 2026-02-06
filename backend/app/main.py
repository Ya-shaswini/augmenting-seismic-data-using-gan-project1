from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from app.websockets import router as ws_router
from app.database import connect_to_mongo, close_mongo_connection

app = FastAPI(title="Seismic GAN API", description="Real-time seismic data augmentation and monitoring")

# CORS setup
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
app.include_router(ws_router, prefix="/ws")

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    await connect_to_mongo()
    from app.services.event_monitor import event_monitor
    # Start earthquake monitoring (check every 5 minutes for magnitude 4.0+)
    await event_monitor.start(check_interval_seconds=300, min_magnitude=4.0)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    await close_mongo_connection()
    from app.services.event_monitor import event_monitor
    await event_monitor.stop()

@app.get("/")
async def root():
    return {"message": "Seismic GAN Backend Operational"}
