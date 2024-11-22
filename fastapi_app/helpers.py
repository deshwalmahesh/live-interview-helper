from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from pyannote.audio import Pipeline as DiarizationPipeline
import asyncio, re, torch
import numpy as np
import time

def remove_blacklisted_words(string:str, ignore_case:bool=False) -> str:
    """
    Given a string and a replacement map, it returns the replaced string.
    Fastest Way: Gistfrom best answer: https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
    args:
        string: string to execute replacements on
        replacements: replacement dictionary {value to find: value to replace}
        ignore_case: whether the match should be case insensitive
    return: Processed string
    """
    REPLACEMENTS = {"Okay": "", "Thank you": "", "Hmm": "", "I'm Sorry": "", "I'm going to go":""}
    
    if not REPLACEMENTS: # Edge case that'd produce a funny regex and cause a KeyError
        return string
    
    # If case insensitive, we need to normalize the old string so that later a replacement
    # can be found. For instance with {"HEY": "Greetings!"} we should match and find a replacement for "hey", "HEY", "hEy", etc.
    if ignore_case:
        def normalize_old(s): return s.lower()

        re_mode = re.IGNORECASE

    else:
        def normalize_old(s): return s
        re_mode = 0

    REPLACEMENTS = {normalize_old(key): val for key, val in REPLACEMENTS.items()}
    
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    rep_sorted = sorted(REPLACEMENTS, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)
    
    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped), re_mode)
    
    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: REPLACEMENTS[normalize_old(match.group(0))], string)



# Whisper settings
WHISPER_LANGUAGE = "english"
TRANSCRIPTION_MODEL_NAME = "openai/whisper-large-v3-turbo"
MAX_SENTENCE_CHARACTERS = 128

# Diarization settings
NUM_SPEAKERS = 2

device_name = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
device = torch.device(device_name)
torch_dtype = torch.bfloat16

# ------ Transcription Helpers ------
class Transcription:
    def __init__(self):
        self.all_latency = []

    def latency_stats(self):
        return np.mean(self.all_latency)

    def load_pipeline(self):

        transcription_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            TRANSCRIPTION_MODEL_NAME, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        
        transcription_model.to(device)

        processor = AutoProcessor.from_pretrained(TRANSCRIPTION_MODEL_NAME)

        self.transcription_pipeline = pipeline(
            "automatic-speech-recognition",
            model=transcription_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s = 30, #  min(LENGTH_IN_SEC, 30)
            torch_dtype=torch_dtype,
            device=device,
        )

        print(f"Loaded {TRANSCRIPTION_MODEL_NAME}")


    async def transcribe(self,audio_data_array, RATE):
        start = time.time()
        res = await asyncio.to_thread(
            self.transcription_pipeline,
            {"array": audio_data_array, "sampling_rate": RATE},
            return_timestamps=True,
            generate_kwargs={"language": "english", "return_timestamps": True, "max_new_tokens": MAX_SENTENCE_CHARACTERS}
        )
        self.all_latency.append(time.time() - start)
        return res


# ---- Speaker Diarization ---- Can't be Done for Live audio


# NOTE: If you lodd it first time, you need to have Accept all clauses foe the model and add a variable
# use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")
# diarization_pipeline = DiarizationPipeline.from_pretrained(
#   "pyannote/speaker-diarization-3.1")


# diarization_pipeline.to(device)

# def diarize(audio_numpy_array, num_speakers, RATE):
#     if num_speakers < 2: return None
#     diarization_result = diarization_pipeline({
#         "waveform": torch.from_numpy(audio_numpy_array).unsqueeze(0),  # Can use .to(device) but pipeline handles that
#         "sample_rate": RATE, "num_speakers": num_speakers})
#     return diarization_result

def diarize(audio_numpy_array, num_speakers, RATE): return None

