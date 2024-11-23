# Live Interview (~cheating~) Helper
Okay! I admit it's cheating in live interview more than helping but if you put conditions to artist's thought process, you better stop calling it an art :grin:

# What does it do?
It Captures Audio from your mic and Speaker in an interview and helps you get answer to your live interview question be it technical or non technical in ***Real Time*** with just a single click.


https://github.com/user-attachments/assets/aa053824-b0c3-46b8-9ca5-8067e8b85482


## Functionalities
- [x] *Free COLAB* GPU support OR On device Real Time Transcription using HF pipeline (uses`whisper-large-v3-turbo`) so you can use any any valid transcription model
- [x] Dedicated input for zero effort typing and getting answers in minimum words
- [x] Single Click Delayed  Screenshot + OCR of your screen. OCR Text becomes part of above box (useful for live coding)
- [x] Double click on any transcription to edit it
- [x] Option to select 0 or more transcription texts. Use `Shift + Click` to select messages from Text-A till Text-B
- [x] Smoothly renders `LaTex` equations and Markdown (Heading, subheading, Lists etc). Click on `Answers` panel to open a bigger popup
- [x] Option to set: OpenAI key, Custom Base Prompt, Audio Batch length, Transcription Language
- [x] Control to Start, Stop, Resume, Clear and Download Transcriptions
- [x] Add custom words and phrases to be removed from transcription (silence, noise, garbage words etc)

# Getting started

1. It is tested on `Python 3.12` so install that first
2. Install [`PyAudio`](https://pypi.org/project/PyAudio/) for your system (it is tested on Mac so should work for linux too. Not sure about Windows)
3. Get you [OpenAI API Key](https://medium.com/@lorenzozar/how-to-get-your-own-openai-api-key-f4d44e60c327)
4. Clone this repo or download the zip file
5. Open Terminal and navigate to the location which contains `requirements.txt`
6. Install requirements using `pip install -r requirements.txt`

## What if you don't have GPU?
1. Upload `Colab_Transcription.ipynb` to Colab
2. Select GPU Environment and follow every instruction
3. Come here and follow the below instructions 

It'll use Colab GPU to transcrive audio for you in real-time, on your local :)

## How to run App Locally?
1. Go base path in your terminal where `main.py` is present and run the command: `fastapi dev main.py` (you're running on your local so don't worry about `dev` or `uvicorn` etc)
2. Go to `127.0.0.1:8000` in your browser
3. Click on `Start` button to get the transcriptions. You can use `Stop`, `Resume`, `Clear` and `Download` for the transcriptions
4. Set `OpanAI` API key. Without this, you won't be able to use `Get Answers`

## Parameters
All of the useful parameters can be set using `config.json`. Also, there is a functionality to change those at the home page itself.

- Click on the left (collapsed) side panel to set Base Prompt for the LLM according to your interview style, topic etc.
- Add `OpenAI` key as without it you can't use Get Answers functionality
- Set `Audio Chunk Length` according to your system as well as your speech speed. Value of `1` will be process 1 second audio at a time thus it'll be faster but won't have much context. Value of 30 will be delayed but will have more context
- Set `Prev Ans Context` as needed. It means how many previous message history to use. This won't be needed usually but in case you have followup questions and want to give context to LLM


# How does it do it?
1. It captures live stream your default Speaker and default Mic Audio
2. Then it transcribes and Diarize (optional) the audio audio stream to get text stream
3. It displays the transcription with timestamps on the (left) panel
4. You can choose which texts to send to the LLM with clicks. You can also edit the transcriptions.
4. With a single click to the `Get Answers` button, it fetches the answers to the questions fro the conversation and displays answers on the (right) panel :bowtie:


# What's Next 🛺
- [ ] Improve Transcription Speed using further tricks like quantization and efficient frameworks (Ollama, TensorRT etc) and LLM speed using Streaming and option to automatically fetching answers in background after every `N` seconds
- [ ] Add the functionality to use attached headphones etc
- [ ] Find a way to record Mic and Speaker seperately (in parallel) so we can perfectly find who's interviewer and who's candidate (Virtual Sound Cards?)
- [ ] Support for Windows
- [ ] Prompt Optimization and Memory management for previous Que-Ans history
- [ ] Add support to choose Open Source and propriety LLMs (Anthropic)
- [ ] Integrate transcription APIs and other Open Source Models
- [ ] Use `KaTex` for better Equation Rendering
- [ ] Implement new functionalities like: use of previous Answer history and summarization, Question Extraction in background, Follow-up question mechanism, multi level prompts etc


# Known Limitations
1. It is tested on MacOS (M3) and will be good to go for Linux too. For Windows, you need a Windows patch of `pyaudio`
2. It use your default mic and speaker which means if you attach a headphone, it might not work. Haven't tested yet
3. Transcription speed can vary depending on your system. It gave me amazing results with M3 chip but for CPU only, it'll be very slow so you need a transcription API or a smaller model
4. There are some small issues with `MathJax` for equation rendering due to LLM output format
   
# Random Fact (that you definitely don't care)

I have no idea of how JavScript works still I managed to build the frontend somehow. All thanks to `Claude Sonnet 3.5` (Had I given someone the money I spent in prompts, it'd still be way lesser but hey! :feelsgood:)

# Help & Support
If you happen to know solutions for 2nd and 3rd `What's Next` Or you have free time to add the other functionalities, please feel free to add those and open a requet. Highly appreciated.

Please open any feature requests and issues which you face. Will try my best to resolve atlest backend ones.
