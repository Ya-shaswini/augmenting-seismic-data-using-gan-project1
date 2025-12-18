from fastapi import APIRouter
import numpy as np
import math

router = APIRouter()

@router.get("/waveform")
async def get_waveform():
    # Simulate a P-wave and S-wave
    t = np.linspace(0, 10, 500)
    noise = np.random.normal(0, 0.1, 500)
    
    # Signal creation
    signal = np.sin(2 * np.pi * 1 * t) * np.exp(-0.5 * (t - 2)**2)  # P-arrival at 2s
    signal += 1.5 * np.sin(2 * np.pi * 0.8 * t) * np.exp(-0.3 * (t - 5)**2) # S-arrival at 5s
    
    raw = signal + noise
    
    return {
        "timestamp": 1234567890,
        "sample_rate": 50,
        "data": raw.tolist(),
        "clean_candidate": signal.tolist() # Cheating for demo
    }

@router.post("/enrich")
async def enrich_signal(data: dict):
    # This is where the GAN model would run
    # For now, return a smoothed version
    raw_data = data.get("data", [])
    # Simple moving average as 'cleaning'
    window_size = 5
    cleaned = np.convolve(raw_data, np.ones(window_size)/window_size, mode='valid')
    return {"cleaned_data": cleaned.tolist()}
