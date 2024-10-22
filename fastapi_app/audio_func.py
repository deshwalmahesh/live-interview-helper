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

# Queues
audio_queue = asyncio.Queue()
length_queue = asyncio.Queue(maxsize=LENGTH_IN_SEC)

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
    chunk_length_s = LENGTH_IN_SEC,
    batch_size = 1,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device=device,
)

print(f"{TRANSCRIPTION_MODEL_NAME} loaded")