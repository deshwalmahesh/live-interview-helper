import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import logging
import numpy as np
# Import your PyTorch model
from audio_func import transcription_pipeline, SAMPLE_RATE, LENGTH_IN_SEC


CHUNK_SIZE = SAMPLE_RATE * LENGTH_IN_SEC # It is 16000 so make a chunk in the multiple of this much time in seconds

app = FastAPI()

def transcribe(audio_data_to_process):
    audio_data_array =  (np.frombuffer(audio_data_to_process, np.int16) / 32768.0).astype(np.float32) # MAYBE THIS IS THE ISSUE?

    print(audio_data_array.min(), audio_data_array.max(), np.mean(audio_data_array), np.std(audio_data_array))
    result = transcription_pipeline({"array": audio_data_array, "sampling_rate": SAMPLE_RATE}, 
                                  return_timestamps=True, 
                                  generate_kwargs={"language": "english", "return_timestamps": True, 
                                                   "max_new_tokens": 128})
    return result["text"]

@app.websocket("/TranscribeStreaming")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = b''
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    audio_chunk = message["bytes"]
                    buffer += audio_chunk
                    while len(buffer) >= CHUNK_SIZE:
                        chunk_to_process = buffer[:CHUNK_SIZE]
                        buffer = buffer[CHUNK_SIZE :]
                        try:
                            transcription = transcribe(chunk_to_process)
                            await websocket.send_text(transcription)
                        except Exception as e:
                            logging.error(f"Error in transcription: {e}")
                            await websocket.send_text(f"Error in transcription: {str(e)}")
                elif "text" in message:
                    if message["text"] == "submit_response":
                        break
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    finally:
        if len(buffer) >= CHUNK_SIZE:  # Process any remaining complete chunks
            try:
                transcription = transcribe(buffer[:CHUNK_SIZE])
                await websocket.send_text(transcription)
            except Exception as e:
                logging.error(f"Error in final transcription: {e}")
                await websocket.send_text(f"Error in final transcription: {str(e)}")
        await websocket.close()