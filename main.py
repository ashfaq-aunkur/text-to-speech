import io
import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Text-to-Speech API")

# Enable CORS so your frontend can call this backend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    voice: str = "bn-BD-NabanitaNeural"  # Default high-quality neural voice


@app.get("/")
def read_root():
    return {"message": "TTS API is running!"}


@app.post("/api/v1/tts")
async def text_to_speech(payload: TTSRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Initialize edge-tts with requested text and voice
        communicate = edge_tts.Communicate(payload.text, payload.voice)

        # Buffer audio chunks into memory
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])

        # Reset stream position to beginning
        audio_stream.seek(0)

        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="response.mp3"'},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"TTS generation failed: {str(e)}"
        )