let websocket;
let audioContext;
let processor;
let chunksSent = 0;
let transcriptionHistory = [];
let currentChunk = [];
let lastTranscriptionTime = Date.now();

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const statusDiv = document.getElementById('status');
const errorDiv = document.getElementById('error');
const transcriptionDiv = document.getElementById('transcription');
const timeWindowInput = document.getElementById('timeWindow');
const getAnswersButton = document.getElementById('getAnswersButton');
const answersPanel = document.getElementById('answersPanel');

const showLastWindowButton = document.getElementById('showLastWindowButton');
const popup = document.getElementById('popup');
const popupContent = document.getElementById('popupContent');
const closePopupButton = document.querySelector('.close-popup');

showLastWindowButton.onclick = showLastWindow;
closePopupButton.onclick = closePopup;
getAnswersButton.onclick = getAnswers;

function showLastWindow() {
    if (transcriptionHistory.length > 0 && transcriptionHistory[0].length > 0) {
        popupContent.innerHTML = transcriptionHistory[0].join('<br>');
        popup.style.display = 'block';
    } else {
        popupContent.innerHTML = 'No transcription available.';
        popup.style.display = 'block';
    }
}

function closePopup() {
    popup.style.display = 'none';
}

async function getAnswers() {
    if (transcriptionHistory.length === 0) {
        answersPanel.innerHTML = 'No transcription data available.';
        return;
    }

    const lastWindow = transcriptionHistory[transcriptionHistory.length - 1];
    try {
        const response = await fetch('http://localhost:8000/get_answers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transcription: lastWindow }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        answersPanel.innerHTML = `<h3>Answers:</h3><p>${data.answers}</p>`;
    } catch (error) {
        console.error('Error:', error);
        answersPanel.innerHTML = `Error getting answers: ${error.message}`;
    }
}

startButton.onclick = startRecording;
stopButton.onclick = stopRecording;

function updateStatus(message) {
    statusDiv.textContent = message;
}

function showError(message) {
    errorDiv.textContent = message;
}

function startRecording() {
    chunksSent = 0;
    transcriptionHistory = [];
    currentChunk = [];
    lastTranscriptionTime = Date.now();

    showLastWindowButton.disabled = false;
    
    websocket = new WebSocket('ws://localhost:8000/TranscribeStreaming');
    
    websocket.onopen = () => {
        updateStatus('WebSocket connection established. Accessing microphone...');
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                updateStatus('Microphone accessed. Recording started.');
                audioContext = new AudioContext();
                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(16384, 1, 1);

                source.connect(processor);
                processor.connect(audioContext.destination);

                let sampleRate = audioContext.sampleRate;
                let resampleRatio = 16000 / sampleRate;

                processor.onaudioprocess = function(e) {
                    let inputData = e.inputBuffer.getChannelData(0);
                    let resampledBuffer = new Float32Array(Math.round(inputData.length * resampleRatio));
                    
                    for (let i = 0; i < resampledBuffer.length; i++) {
                        resampledBuffer[i] = inputData[Math.floor(i / resampleRatio)];
                    }

                    let int16Array = new Int16Array(resampledBuffer.length);
                    for (let i = 0; i < resampledBuffer.length; i++) {
                        int16Array[i] = Math.max(-32768, Math.min(32767, Math.round(resampledBuffer[i] * 32767)));
                    }

                    websocket.send(int16Array.buffer);
                    chunksSent++;
                    updateStatus(`Recording... (${chunksSent} chunks sent)`);
                };
            })
            .catch(err => {
                showError('Error accessing microphone: ' + err.message);
                console.error('Error accessing microphone:', err);
            });
    };

    websocket.onmessage = event => {
        console.log('Received message:', event.data);
        if (event.data === "Transcription complete.") {
            updateStatus('Transcription complete.');
        } else {
            const now = new Date();
            const timeString = now.toLocaleTimeString();
            const newText = `${timeString}: ${event.data}`;
            transcriptionDiv.innerHTML += newText + '<br>';
            transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
    
            currentChunk.push(newText);
            
            const currentTime = Date.now();
            const timeWindow = timeWindowInput.value * 1000; // Convert to milliseconds
            
            // Remove old entries from currentChunk
            while (currentChunk.length > 0 && currentTime - new Date(currentChunk[0].split(': ')[0]).getTime() > timeWindow) {
                currentChunk.shift();
            }
            
            // Update transcription history
            transcriptionHistory = [currentChunk];
            
            console.log('Current transcription history:', transcriptionHistory);
        }
    };

    websocket.onerror = error => {
        showError('WebSocket error: ' + error.message);
        console.error('WebSocket error:', error);
    };

    websocket.onclose = () => {
        updateStatus('WebSocket connection closed');
    };

    startButton.disabled = true;
    stopButton.disabled = false;
}

function stopRecording() {
    if (processor) {
        processor.disconnect();
    }
    if (audioContext) {
        audioContext.close();
    }
    updateStatus('Recording stopped. Sending final data...');
    websocket.send('submit_response');
    startButton.disabled = false;
    stopButton.disabled = true;
}