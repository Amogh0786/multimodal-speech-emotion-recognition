from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse
from .schemas import PredictionResponse
from .dependencies import get_inference_pipeline
from src.utils.audio_utils import buffer_to_tensor
import uvicorn
from loguru import logger

app = FastAPI(title="Speech Emotion Recognition API", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(
    file: UploadFile = File(...),
    pipeline = Depends(get_inference_pipeline)
):
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .wav, .mp3, or .flac")
        
    try:
        buffer = await file.read()
        audio_tensor = buffer_to_tensor(buffer, target_sr=16000)
        
        # Preprocessor expects exactly 3s. The pipeline handles this if we call preprocessor
        # We need to ensure the inference pipeline runs preprocessor.
        # Calling preprocessor directly on tensor
        # Wait, preprocessor is written for files. Let's fix preprocessor or do it here.
        
        # Trim silence and pad/chunk (re-implementing the preprocessor logic for tensors)
        import librosa
        import numpy as np
        wav = audio_tensor.numpy()
        wav, _ = librosa.effects.trim(wav, top_db=25)
        
        target_length = 48000
        if len(wav) < target_length:
            pad_len = target_length - len(wav)
            wav = np.pad(wav, (0, pad_len), mode='constant')
        elif len(wav) > target_length:
            start = (len(wav) - target_length) // 2
            wav = wav[start:start + target_length]
            
        import torch
        processed_tensor = torch.from_numpy(wav).float()
        
        result = pipeline.predict(processed_tensor)
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during processing.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
