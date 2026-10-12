import io
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import VitsModel, VitsTokenizer

app = FastAPI(title="Bangla TTS API")

# 1. Enable CORS so your frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Select compute device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Load Meta MMS Bengali TTS Model into memory on app startup
MODEL_NAME = "facebook/mms-tts-ben"
tokenizer = VitsTokenizer.from_pretrained(MODEL_NAME)
model = VitsModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


class TTSRequest(BaseModel):
    text: str


@app.post("/api/v1/tts")
async def text_to_speech(payload: TTSRequest):
    if not payload.text.strip():
        raise HTTPException(
            status_code=400, detail="Text payload cannot be empty."
        )

    try:
        # Tokenize input text and send to target device
        inputs = tokenizer(payload.text, return_tensors="pt").to(device)

        # Generate audio waveform
        with torch.no_grad():
            output = model(**inputs).waveform

        # Convert tensor to 1D numpy array
        audio_data = output.squeeze().cpu().numpy()

        # Write audio waveform array into BytesIO memory stream
        audio_bytes = io.BytesIO()
        sf.write(
            audio_bytes,
            audio_data,
            model.config.sampling_rate,
            format="WAV",
        )
        audio_bytes.seek(0)

        return StreamingResponse(
            audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="response.wav"'},
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"TTS generation failed: {str(e)}"
        )