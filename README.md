# Language Learning Companion
The **Language Learning Companion** is an adaptive language learning tool designed to enhance pronunciation and comprehension skills through personalized, adaptive feedback. By leveraging speech processing and affect recognition, the system dynamically adjusts exercises based on user performance and emotional state, creating a personalized and engaging learning experience.

### Key Features
- **Speech Analysis:** Real-time pronunciation evaluation using audio input.
- **Affect Recognition:** Detects user emotions (e.g., frustration, confidence) via webcam to adapt difficulty levels.
- **Adaptive Feedback:** Provides tailored feedback and adjusts exercise complexity based on performance and emotional cues.

### How It Works
0. If the user wishes to base vowel normalization on their own vocals, run ```calibrate.py```
1. The user speaks into their microphone to complete language exercises.
2. The system evaluates pronunciation errors using audio analysis.
3. A webcam detects the user's emotional state (e.g., frustration or confidence).
4. Based on performance and emotions, the system adjusts the difficulty of subsequent exercises and provides personalized feedback.

### Tech Stack
- Speech Processing: PyAudio, librosa
- Emotion Detection: OpenCV, DeepFace
- Programming Language: Python
- Platform: macOS (tested)

### Current Status
The project is in its prototype phase, focusing on speech processing and affect recognition. Future iterations may integrate handwriting or type analysis.

### Credits
- LibriSpeech dataset (960h of English speech)
- Meta wav2vec 2.0 ASR model
- Mayer, R. E. (2021). Multimedia learning (3rd ed.). Cambridge University Press
- D'Mello, S., Graesser, A., & Lehman, B. (2018). Affect-aware tutoring systems: Creating cyber-human learning partnerships. In R. Sottilare, A. Graesser, X. Hu, & G. Goodwin (Eds.), Design Recommendations for Intelligent Tutoring Systems (Vol. 6, pp. 131-152). U.S. Army Research Laboratory.
- Flynn, N. (2011). Comparing Vowel Formant Normalisation Procedures. York Papers in Linguistics, Issue 11, 1-28.