import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
import numpy as np
from audio_func import transcribe_audio, SAMPLE_RATE, LENGTH_IN_SEC

CHUNK_SIZE = SAMPLE_RATE * LENGTH_IN_SEC

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TranscriptionRequest(BaseModel):
    transcription_history: List[str]


async def process_audio_chunks(queue, websocket):
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:  # Signal to stop processing
                break
            try:
                transcription = await transcribe_audio(chunk)
                await websocket.send_text(transcription)
            except Exception as e:
                logging.error(f"Error in transcription: {e}")
                try:
                    await websocket.send_text(f"Error in transcription: {str(e)}")
                except RuntimeError:
                    logging.error("Failed to send error message, WebSocket might be closed")
                    break
            queue.task_done()
    except RuntimeError:
        logging.error("WebSocket closed during processing")
    except Exception as e:
        logging.error(f"Unexpected error in process_audio_chunks: {e}")

@app.websocket("/TranscribeStreaming")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = b''
    queue = asyncio.Queue()
    
    # Start the audio processing task
    process_task = asyncio.create_task(process_audio_chunks(queue, websocket))
    
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    audio_chunk = message["bytes"]
                    buffer += audio_chunk
                    while len(buffer) >= CHUNK_SIZE:
                        chunk_to_process = buffer[:CHUNK_SIZE]
                        buffer = buffer[CHUNK_SIZE:]
                        await queue.put(chunk_to_process)
                elif "text" in message and message["text"] == "submit_response":
                    break
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except RuntimeError:
        logging.info("WebSocket closed unexpectedly")
    finally:
        if buffer:  # Process any remaining data
            await queue.put(buffer)
        await queue.put(None)  # Signal to stop processing
        await process_task
        try:
            await websocket.close()
        except RuntimeError:
            logging.info("WebSocket already closed")

@app.post("/get_answers")
async def get_answers(request: TranscriptionRequest):

    # Dummy Answers
    full_transcription = " ".join(request.transcription_history)

    answer = f"Received transcription with {len(request.transcription_history)} entries. " \
             f"The full transcription is: {full_transcription}..."
    
    return {"answers": answer}