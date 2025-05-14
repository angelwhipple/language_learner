# Language Learning Companion
The **Language Learning Companion** is an adaptive language learning tool designed to enhance pronunciation and comprehension skills through personalized, adaptive feedback. By leveraging speech processing and affect recognition, the system dynamically adjusts exercises based on user performance and emotional state, creating a personalized and engaging learning experience.

### Key Features
- **Speech Analysis:** Real-time pronunciation evaluation using audio input.
- **Affect Recognition:** Detects user emotions (e.g., frustration, confidence) via webcam to adapt difficulty levels.
- **Adaptive Feedback:** Tailors future exercises and adjusts difficulty based on performance and emotional cues.

### How It Works
0. Run `python app.py` to start the program.
1. When ready, press "Record audio" and read the practice sentence clearly into the microphone.
2. The system evaluates pronunciation errors using audio analysis.
3. A webcam detects the user's emotional state (e.g., frustration or confidence).
4. Based on performance and emotions, the system tailors the content/difficulty of future training exercises, and offers pronunciation feedback.

### File Structure
- `audio/` contains audio (.mp3), text (.lab), and alignment (.TextGrid) files for recorded audio
- `resources/` contains resources the program needs to run like practice sentences, and reference vowel pronunciation data
- `spectrogram/` contains a spectrogram of the user's recorded audio frequencies
- `audio_processing.py` contains modules for audio sampling and phonetic analysis, including a `VocalSample`, `Phonemes`, `Words`, `Recorder`, and `Transcriber` class
- `video_processing.py` contains simple methods for recording and analyzing video frames
- `app.py` is the current main file. Run `python3 app.py` to start the UI and test the program
- `main.py` is the *former* main file. Run `python3 main.py` to see a terminal-driven version of the system
- `gpt.py` contains methods for interfacing with the OpenAI API 
- `calibrate.py` 
- `setup_mfa.sh`
- `results.ipynb` is a short Python notebook generating plots from accuracy test results

### Tech Stack
- Speech Processing: PyAudio, librosa, wav2vec
- Emotion Detection: OpenCV, DeepFace
- Programming Language: Python
- Platform: macOS (tested)

### Credits
- LibriSpeech dataset (960h of English speech)
- Meta wav2vec 2.0 ASR model
- Mayer, R. E. (2021). Multimedia learning (3rd ed.). Cambridge University Press
- D'Mello, S., Graesser, A., & Lehman, B. (2018). Affect-aware tutoring systems: Creating cyber-human learning partnerships. In R. Sottilare, A. Graesser, X. Hu, & G. Goodwin (Eds.), Design Recommendations for Intelligent Tutoring Systems (Vol. 6, pp. 131-152). U.S. Army Research Laboratory.
- Flynn, N. (2011). Comparing Vowel Formant Normalisation Procedures. York Papers in Linguistics, Issue 11, 1-28.