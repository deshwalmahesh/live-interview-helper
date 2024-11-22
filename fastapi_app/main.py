from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pyaudio
import numpy as np
import logging
from pydantic import BaseModel
from helpers import Transcription, remove_blacklisted_words, diarize
from llm_helper import OpenAILLM
import json
import aiohttp
from base64 import b64encode
import time


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


app = FastAPI()
LLM = OpenAILLM()
TRNS = Transcription()

with open('config.json') as config_file:
    config = json.load(config_file)

# Audio settings
STEP_IN_SEC: int = 1
LENGTH_IN_SEC = config['audio']['length_in_sec']
RATE = config['audio']['rate']
NB_CHANNELS = config['audio']['num_channels']
NUM_SPEAKERS = config['transcription']['num_speakers']
CHUNK_SIZE = RATE * LENGTH_IN_SEC
API_ENDPOINT = config["transcription"]["API_ENDPOINT"]


if API_ENDPOINT: API_ENDPOINT = API_ENDPOINT.rstrip("/") + "/transcribe"
else: TRNS.load_pipeline() # Don't use cloud 


LLM.system_prompt = config["llm"]['system_prompt']
LLM.model_name = config["llm"]['model_name']


global audio_buffer, START, RESUMING
START = asyncio.Event()
RESUMING = False
active_connections = set()
audio_buffer = asyncio.Queue(maxsize = CHUNK_SIZE)
client_audio_buffer = asyncio.Queue(maxsize = CHUNK_SIZE)

thread_pool = ThreadPoolExecutor(max_workers=1)


class PreviousAnswersHistory(BaseModel):
    prev_transcriptions: List[str]
    prev_answers: List[str]

class TranscriptionRequest(BaseModel):
    transcription: List[str]
    previous_answers_history: PreviousAnswersHistory
    k_answer_history: int

class TranscriptionConfig(BaseModel):
    numSpeakers: int

class LengthInSecConfig(BaseModel):
    lengthInSec: int

class SystemPrompt(BaseModel):
    system_prompt: str


async def send_audio_to_cloud(audio_data: bytes) -> dict:
    """
    Send audio data to cloud API for transcription.
    """
    # Convert audio bytes to base64 for sending over HTTP
    audio_base64 = b64encode(audio_data).decode('utf-8')
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                API_ENDPOINT,  # API_ENDPOINT should be defined at module level
                json={'audio_data': audio_base64}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API request failed with status {response.status}")
        except Exception as e:
            logging.error(f"Error sending audio to cloud: {str(e)}")
            raise


# Base Config to set for Frontend
@app.get("/base-config")
async def get_config():
    return JSONResponse(content=config)

# ---- LLM End Points ----
@app.post("/openai-login")
async def login(request: dict):
    api_key = request.get("api_key")
    try:
        LLM.login_and_create_client(api_key)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/update_system_prompt")
async def update_system_prompt(prompt: SystemPrompt):
    try:
        LLM.system_prompt = prompt.system_prompt
        return {"message": "System prompt updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/get_answers")
async def get_answers(request: TranscriptionRequest):
    """
    Error Handling, Proper history usage remaining
    """
    transcription_history = str(request.transcription) # We need to properly format it
    previous_answers_history = request.previous_answers_history # Add functionality to use it in calls, if needed
    k_answer_history = request.k_answer_history # No of previous messages to use as answer history
    
    logger.info(f"Current Transcription History: {transcription_history}")
    logger.info(f"Previous Answers History: {previous_answers_history}")
    logger.info(f"K History: {k_answer_history}")

    markdown_content = LLM.hit_llm(transcription_history)
    
    return JSONResponse(content={"markdown": markdown_content})


@app.post("/update_length_in_sec")
async def update_length_in_sec(config: LengthInSecConfig):
    global LENGTH_IN_SEC, RATE, CHUNK_SIZE
    LENGTH_IN_SEC = config.lengthInSec
    CHUNK_SIZE = RATE * LENGTH_IN_SEC
    return {"status": "updated", "lengthInSec": LENGTH_IN_SEC}

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


async def producer_task():
    global RATE, CHUNK_SIZE
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=RATE,
    )

    while START.is_set():
        try:
            audio_data = await asyncio.to_thread(stream.read, RATE, exception_on_overflow=False)
            await audio_buffer.put(audio_data)
            logger.info(f"Producer added data. Buffer size: {audio_buffer.qsize()}")
        except Exception as e:
            logger.error(f"Error in producer task: {str(e)}")

    stream.stop_stream()
    stream.close()
    audio.terminate()

async def consumer_task():
    global NUM_SPEAKERS, LENGTH_IN_SEC, RATE, CHUNK_SIZE

    while START.is_set():
        try:
            if audio_buffer.qsize() >= LENGTH_IN_SEC:
                audio_data_to_process = b''.join([await audio_buffer.get() for _ in range(LENGTH_IN_SEC)])
                
                if API_ENDPOINT is not None:  # Send audio data to cloud API
                    try:
                        start = time.time()
                        transcription = await send_audio_to_cloud(audio_data_to_process)
                        TRNS.all_latency.append(time.time() - start)

                        if transcription:
                            transcription_text = remove_blacklisted_words(transcription["text"].rstrip(". ?!"))
                    except Exception as e:
                        logger.error(f"Cloud transcription failed: {str(e)}")
                        transcription_text = ""
                
                else:
                    audio_data_array = np.frombuffer(audio_data_to_process, np.int16).astype(np.float32) / 32768.0

                    logger.info(f"LENGTH_IN_SEC: {LENGTH_IN_SEC} || CHUNK_SIZE: {CHUNK_SIZE}")

                    transcription_task = asyncio.create_task(TRNS.transcribe(audio_data_array, RATE = RATE))

                    # Run diarization in a separate thread as it's giving error in async mode
                    logger.info(f"Current No of Speakers: {NUM_SPEAKERS}")
                    diarization_task = asyncio.create_task(
                        asyncio.to_thread(diarize, audio_data_array, NUM_SPEAKERS, RATE))

                    # Wait for both tasks to complete
                    transcription, diarization = await asyncio.gather(transcription_task, diarization_task)
                    logger.info(f"DIARIZATION:\n{diarization}")
                    
                
                    if transcription:
                        transcription_text = remove_blacklisted_words(transcription["text"]).rstrip(". ?!")


                await send_transcription(transcription_text)
                logger.info(f"Sent transcription: {transcription_text}")
                logging.info(TRNS.latency_stats())
                
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in consumer task: {str(e)}")

        logger.info(f"Consumer task iteration - Buffer size: {audio_buffer.qsize()}")


@app.post("/start")
async def start_transcription(config: TranscriptionConfig):
    global audio_buffer, RESUMING, NUM_SPEAKERS
    NUM_SPEAKERS = config.numSpeakers
    if not START.is_set():
        START.set()
        RESUMING = False
        audio_buffer = asyncio.Queue(maxsize=CHUNK_SIZE)
        asyncio.create_task(producer_task())
        asyncio.create_task(consumer_task())
    return {"status": "started"}

@app.post("/resume")
async def resume_transcription(config: TranscriptionConfig):
    global RESUMING, NUM_SPEAKERS
    NUM_SPEAKERS = config.numSpeakers
    if not START.is_set():
        START.set()
        RESUMING = True
        asyncio.create_task(producer_task())
        asyncio.create_task(consumer_task())
    return {"status": "resumed"}

@app.post("/stop")
async def stop_transcription():
    START.clear()
    return {"status": "stopped"}


# ----- Functionality Where Frontend Client sends audio -------
@app.websocket("/audio-stream")
async def websocket_audio_stream(websocket: WebSocket):
    await websocket.accept()
    buffer = b''
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer += data
            while len(buffer) >= CHUNK_SIZE:
                chunk_to_process = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]
                await client_audio_buffer.put(chunk_to_process)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket_audio_stream: {str(e)}")
    finally:
        if buffer:
            await client_audio_buffer.put(buffer)

async def client_consumer_task():
    while START.is_set():
        try:
            chunk = await client_audio_buffer.get()
            audio_data_array = np.frombuffer(chunk, np.int16).astype(np.float32) / 32768.0

            transcription = await TRNS.transcribe(audio_data_array)
            transcription_text = transcription["text"].rstrip(".")

            if transcription_text:
                await send_transcription(transcription_text)
                logger.info(f"Sent transcription: {transcription_text}")
        except Exception as e:
            logger.error(f"Error in client consumer task: {str(e)}")

        logger.info(f"Client consumer task iteration - Buffer size: {client_audio_buffer.qsize()}")


@app.post("/start-client")
async def start_client_transcription():
    global client_audio_buffer, RESUMING
    if not START.is_set():
        START.set()
        RESUMING = False
        # Clear the buffer by creating a new Queue
        client_audio_buffer = asyncio.Queue(maxsize = CHUNK_SIZE)
        asyncio.create_task(client_consumer_task())
    return {"status": "started"}

@app.post("/resume-client")
async def resume_client_transcription():
    if not START.is_set():
        START.set()
        global RESUMING
        RESUMING = True
        asyncio.create_task(client_consumer_task())
    return {"status": "resumed"}

@app.post("/stop-client")
async def stop_client_transcription():
    START.clear()
    return {"status": "stopped"}


# ------ Common Functions  --------
async def status_check():
    while True:
        logger.info(f"Task status - START: {START.is_set()}, Server Buffer size: {audio_buffer.qsize()}, Client Buffer size: {client_audio_buffer.qsize()}")
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(status_check())

app.mount("/", StaticFiles(directory="static", html=True), name="static")
