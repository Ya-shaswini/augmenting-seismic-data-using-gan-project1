from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.gan_service import gan_service
from fastapi import UploadFile, File
import numpy as np
import io

router = APIRouter()

class TrainRequest(BaseModel):
    data: List[List[float]]
    epochs: int = 1000
    batch_size: int = 64
    label: str

class GenerateRequest(BaseModel):
    num_samples: int = 1
    label: str

@router.post("/train")
async def train_gan(request: TrainRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="No data provided")
    
    # Check dimensions
    if len(request.data[0]) != 1024:
         raise HTTPException(status_code=400, detail="Data length must be 1024 per sample")

    started = gan_service.start_training(request.data, request.label, request.epochs, request.batch_size)
    if not started:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    return {"message": "Training started"}

@router.get("/status")
async def get_status():
    return gan_service.get_status()

@router.post("/generate")
async def generate_data(request: GenerateRequest):
    try:
        data = gan_service.generate(request.num_samples, request.label)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/synthesize")
async def synthesize_event(magnitude: float = 5.0, distance: float = 50.0):
    """
    Demonstrates synthesis: Generates a structured earthquake with specific
    parameters to help earthquake centers analyze events.
    """
    from app.services.prediction_service import prediction_service
    
    # 1. Generate Ground Truth (Clean)
    clean_signal = prediction_service.generate_synthetic_earthquake(magnitude=magnitude, distance_km=distance)
    
    # 2. Add Heavy Noise (MEMS Simulation) - noise level also scaled by magnitude to prevent masking everything
    noise_level = 0.25 if magnitude > 4 else 0.1
    noise = np.random.normal(0, noise_level, len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # 3. Use GAN/Denoiser to convert back to High Quality
    result = gan_service.denoise_and_predict(noisy_signal.tolist())
    
    return {
        "title": f"M{magnitude} Earthquake at {distance}km",
        "parameters": {
            "magnitude": magnitude,
            "distance_km": distance,
            "sample_rate_hz": 100,
            "units": "Gal (Acceleration)"
        },
        "raw_noisy": noisy_signal.tolist(),
        "synthesized_clean": result['denoised_data'],
        "snr_improvement": result['snr_improvement'],
        "prediction": result['prediction']
    }

@router.get("/research/batch-synthesize")
async def batch_synthesize(
    num_samples: int = 10, 
    min_mag: float = 4.0, 
    max_mag: float = 7.0,
    min_dist: float = 10.0,
    max_dist: float = 100.0
):
    """
    Generates a batch of synthetic records for research center benchmarking.
    """
    from app.services.prediction_service import prediction_service
    dataset = []
    
    for _ in range(num_samples):
        mag = np.random.uniform(min_mag, max_mag)
        dist = np.random.uniform(min_dist, max_dist)
        
        # Generate Ground Truth
        clean_signal = prediction_service.generate_synthetic_earthquake(magnitude=mag, distance_km=dist)
        
        # Add Noise
        noise_level = 0.2 if mag > 5 else 0.05
        noise = np.random.normal(0, noise_level, len(clean_signal))
        noisy_signal = clean_signal + noise
        
        # Denoise
        result = gan_service.denoise_and_predict(noisy_signal.tolist())
        
        dataset.append({
            "magnitude": round(mag, 2),
            "distance_km": round(dist, 2),
            "clean_data": result['denoised_data']
        })
        
    return {
        "status": "success",
        "count": len(dataset),
        "dataset": dataset
    }

@router.post("/batch-denoise")
async def batch_denoise(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
         raise HTTPException(status_code=400, detail="Only CSV files are supported for now")
    
    try:
        contents = await file.read()
        # Assume CSV is simple list of numbers or time-series
        # converting bytes to string then to numpy array
        # This is a basic implementation assuming a single column or comma separated values
        text_content = contents.decode('utf-8')
        
        # Simple parsing logic for demo:
        # 1. Try to parse as single row of comma separated values
        # 2. Or multiple rows
        data = []
        lines = text_content.strip().split('\n')
        for line in lines:
            if not line: continue
            try:
                # remove potential header if it's not a number
                if line[0].isalpha(): continue 
                
                parts = line.split(',')
                row = [float(p) for p in parts if p.strip()]
                data.extend(row)
            except ValueError:
                continue
                
        if not data:
             raise HTTPException(status_code=400, detail="Could not parse any numerical data from CSV")

        # Process exactly 1024 points for the model
        if len(data) < 1024:
            # Pad with zeros if too short
            padded_data = data + [0.0] * (1024 - len(data))
            process_data = padded_data
        else:
            # Use first 1024 points
            process_data = data[:1024]
            
        result = gan_service.denoise_and_predict(process_data)
        
        return {
            "filename": file.filename,
            "original_samples": len(data),
            "status": "completed",
            "message": "File successfully denoised and analyzed for earthquake prediction.",
            "snr_improvement": f"{result['snr_improvement']:.2f} dB",
            "prediction": result['prediction'],
            "preview_data": result['denoised_data'][:100],
            "cleaned_data": result['denoised_data']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
