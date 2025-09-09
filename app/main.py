from fastapi import FastAPI, UploadFile, File, Form
from app.services.whisper.asr import WhisperASR
import os

app = FastAPI()
asr_model = WhisperASR(model_size="small")  # can be "tiny", "base", "small", "medium", "large"

@app.get("/")
def root():
    return {"message": "Hello, World! FastAPI is running 🚀"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), text: str = Form(...)):
    """
    Upload audio + expected transcript string.
    Audio should be WAV/PCM16/mono/16kHz
    """
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        transcription = asr_model.transcribe_audio(temp_path)
        return {"transcription": transcription, "expected_text": text}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
