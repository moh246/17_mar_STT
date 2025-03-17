from fastapi import FastAPI, HTTPException
import torchaudio
import torch
import time
import numpy as np
import io
import librosa
import base64
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()

# Load ASR model with error handling
try:
    pipe_seamless = pipeline(
        "automatic-speech-recognition",
        model="facebook/seamless-m4t-v2-large",
        trust_remote_code=True
    )
except Exception as e:
    raise RuntimeError(f"Failed to load ASR model: {e}")

class AudioInput(BaseModel):
    audio_base64: str  # Expecting a Base64-encoded string

@app.post("/transcribe/")
async def transcribe_audio(audio_data: AudioInput):
    try:
        # Decode Base64 audio
        audio_bytes = base64.b64decode(audio_data.audio_base64)
        audio_stream = io.BytesIO(audio_bytes)

        # Load audio using torchaudio
        audio_tensor, sample_rate = torchaudio.load(audio_stream)

        # Convert multi-channel to mono
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        # Convert to NumPy format, ensuring it is 1D float32
        audio_array = audio_tensor.numpy().astype(np.float32).flatten()

        # Perform transcription
        start_time = time.time()
        transcription = pipe_seamless(audio_array, generate_kwargs={"tgt_lang": "arb"})
        end_time = time.time()
        print(transcription)

        return {
            "transcription": transcription.get("text", ""),
            "processing_time": f"{end_time - start_time:.2f} seconds"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

# Run server on port 9000
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="debug")
