import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import logging
import numpy as np
from audio_func import transcribe_audio, SAMPLE_RATE, LENGTH_IN_SEC

CHUNK_SIZE = SAMPLE_RATE * LENGTH_IN_SEC

app = FastAPI()

async def process_audio_chunks(queue, websocket):
    while True:
        chunk = await queue.get()
        if chunk is None:  # Signal to stop processing
            break
        try:
            transcription = await transcribe_audio(chunk)
            await websocket.send_text(transcription)
        except Exception as e:
            logging.error(f"Error in transcription: {e}")
            await websocket.send_text(f"Error in transcription: {str(e)}")
        queue.task_done()

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
    finally:
        if buffer:  # Process any remaining data
            await queue.put(buffer)
        await queue.put(None)  # Signal to stop processing
        await process_task
        await websocket.close()