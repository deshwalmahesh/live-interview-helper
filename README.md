# Live Real Time Audio Transcription 
It uses:
1. OpenAI's [Whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
2. FastAPI + JavaScript

# What's new then?
It can transcribe your data, dump it and then as you can see query it (TO DO). This app has 2 modes of working:
1. You can use your Browser's mic to send audio stream to the backend to get it transcribed in real time. Audio from this approach is isolated.
2. You can use system's audio using `pyaudio`. What do I mean by this? It means that the frontend is just there to initiate the Websocket and show you the functionality. By using `pyaudio`, if you are on default mic + speaker, it uses both audios so if you are in a meeting, you can transcribe that. You can use a Speaker Diarization to detect who said what and when to transcribe your Google Meet, Zoom and any other meeting. You don't even need to make a bot of it.

As you can see there is `Get Answer` button which is meant for later greater purpose :)

# What's Next 🛺
- [ ] Implement Speaker Diarization in parallel async mode and merge with Transcription
- [ ] Add Flexibility to get Transcription Window, Number of Speakers, Language etc etc
- [ ] Add Silence Word, phrases blacklist
- [ ] Add functionality for `Get Answer` :bowtie:

   
### Fun Fact

I have no idea of how JavScript works still I managed to build the frontend somehow. All thanks to `Claude Sonnet 3.5` (Had I given someone the money I spent in prompts, it'd still be way lesser but hey! :feelsgood:)
