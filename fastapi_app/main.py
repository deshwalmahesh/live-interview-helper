from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import re
import pyaudio
import torch 
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import threading
from collections import deque
import logging
from pydantic import BaseModel

app = FastAPI()
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


TRANSCRIPTION_MODEL_NAME = "openai/whisper-large-v3-turbo"

# Audio settings
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
LENGTH_IN_SEC: int = 7    # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE

# Whisper settings
WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 1

# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 128

audio_buffer = deque(maxlen=LENGTH_IN_SEC)


device_name = torch.device("cuda") if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
device = torch.device(device_name)
torch_dtype = torch.bfloat16

transcription_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    TRANSCRIPTION_MODEL_NAME, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
transcription_model.to(device)

processor = AutoProcessor.from_pretrained(TRANSCRIPTION_MODEL_NAME)

transcription_pipeline = pipeline(
    "automatic-speech-recognition",
    model=transcription_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s = min(LENGTH_IN_SEC, 30), # it uses sliding window and other protocol
    torch_dtype=torch_dtype,
    device=device,
)

logger.info(f"{TRANSCRIPTION_MODEL_NAME} loaded")

audio_buffer = deque(maxlen=LENGTH_IN_SEC * CHUNK)
START = False
RESUMING = False
active_connections = set()


class TranscriptionRequest(BaseModel):
    transcription: list

@app.post("/get_answers")
async def get_answers(request: TranscriptionRequest):
    transcription_history = request.transcription
    logger.info(f"Transcription History: {transcription_history}")
    answer = "This is a sample answer based on the provided transcription."
    return answer * (np.random.randint(5,50))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)

async def send_transcription(transcription: str):
    for connection in active_connections:
        await connection.send_text(transcription)

def producer_thread():
    global START
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    while START:
        try:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_buffer.append(audio_data)
            logger.info(f"Producer added data. Buffer size: {len(audio_buffer)}")
        except Exception as e:
            logger.error(f"Error in producer thread: {str(e)}")

    stream.stop_stream()
    stream.close()
    audio.terminate()

async def consumer_thread():
    global START
    while START:
        try:
            if len(audio_buffer) >= LENGTH_IN_SEC:
                audio_data_to_process = b''.join(list(audio_buffer))
                audio_buffer.clear()

                audio_data_array = np.frombuffer(audio_data_to_process, np.int16).astype(np.float32) / 32768.0

                transcription = transcription_pipeline(
                    {"array": audio_data_array, "sampling_rate": RATE},
                    return_timestamps=True,
                    generate_kwargs={"language": "english", "return_timestamps": True, "max_new_tokens": MAX_SENTENCE_CHARACTERS}
                )["text"]

                transcription = transcription.rstrip(".")

                if transcription:
                    await send_transcription(transcription)
                    logger.info(f"Sent transcription: {transcription}")
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in consumer thread: {str(e)}")

        logger.info(f"Consumer thread iteration - Buffer size: {len(audio_buffer)}")


@app.post("/start")
async def start_transcription():
    global START, RESUMING, audio_buffer
    if not START:
        START = True
        RESUMING = False
        audio_buffer.clear()  # Clear the buffer when starting a new session
        threading.Thread(target=producer_thread).start()
        asyncio.create_task(consumer_thread())
    return {"status": "started"}

@app.post("/resume")
async def resume_transcription():
    global START, RESUMING
    if not START:
        START = True
        RESUMING = True
        threading.Thread(target=producer_thread).start()
        asyncio.create_task(consumer_thread())
    return {"status": "resumed"}

@app.post("/stop")
async def stop_transcription():
    global START
    START = False
    return {"status": "stopped"}

async def status_check():
    while True:
        logger.info(f"Thread status - START: {START}, Buffer size: {len(audio_buffer)}")
        await asyncio.sleep(5)  # Check every 5 seconds

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(status_check())

app.mount("/", StaticFiles(directory="static", html=True), name="static")
