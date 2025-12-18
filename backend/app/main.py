from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from app.websockets import router as ws_router

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

@app.get("/")
async def root():
    return {"message": "Seismic GAN Backend Operational"}
