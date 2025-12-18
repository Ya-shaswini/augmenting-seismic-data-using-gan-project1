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

        # Simulate Denoising (Mock)
        # In real world, we'd pass 'data' to gan_service.denoise(data)
        data_np = np.array(data)
        # Apply simple smoothing for "denoised" effect
        cleaned = np.convolve(data_np, np.ones(5)/5, mode='same')
        
        return {
            "filename": file.filename,
            "original_samples": len(data),
            "status": "completed",
            "message": "File successfully processed and denoised.",
            "preview_data": cleaned[:100].tolist(),
            "cleaned_data": cleaned.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
