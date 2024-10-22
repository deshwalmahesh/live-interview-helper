import queue
import re
import pyaudio
import asyncio
import torch 
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np


# Audio settings
LENGTH_IN_SEC: int = 5 #  Maximum time duration at which the audio data will pe processed together at once. Think of it as sliding window
SAMPLE_RATE = 16000 # Per second Sampling Rate

# Whisper settings
LANGUAGE = "english"
TRANSCRIPTION_MODEL_NAME = "openai/whisper-large-v3-turbo"

# Visualization
MAX_SENTENCE_CHARACTERS = 128

# Devices, dtypes
device_name = torch.device("cuda") if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
device = torch.device(device_name)
torch_dtype = torch.bfloat16

# --------------------- Transcription Pipeline -------------------------------

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

print(f"{TRANSCRIPTION_MODEL_NAME} loaded")

async def transcribe_audio(audio_data):
    audio_data_array = (np.frombuffer(audio_data, np.int16) / 32768.0).astype(np.float32)
    # np.frombuffer(audio_data_to_process, np.int16).astype(np.float32) / 255.0 Which one is the right way?

    result = await asyncio.to_thread(
        transcription_pipeline,
        {"array": audio_data_array, "sampling_rate": SAMPLE_RATE},
        return_timestamps=True,
        generate_kwargs={"language": "english", "return_timestamps": True, "max_new_tokens": 128})
    
    text = result["text"]
    
    # # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
    # text = re.sub(r"\[.*\]", "", text) 
    # text = re.sub(r"\(.*\)", "", text)

    return text
