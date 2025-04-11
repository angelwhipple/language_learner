# Language Learning Companion
The **Language Learning Companion** is an adaptive language learning tool designed to enhance pronunciation and comprehension skills through personalized, adaptive feedback. By leveraging speech processing and affect recognition, the system dynamically adjusts exercises based on user performance and emotional state, creating a personalized and engaging learning experience.

### Key Features
- **Speech Analysis:** Real-time pronunciation evaluation using audio input.
- **Affect Recognition:** Detects user emotions (e.g., frustration, confidence) via webcam to adapt difficulty levels.
- **Adaptive Feedback:** Provides tailored feedback and adjusts exercise complexity based on performance and emotional cues.

### Planned Modules
1. Core Modules (implemented in prototype):
   - Speech processing for pronunciation error detection.
   - Emotion recognition using facial expressions.
   - Adaptive feedback system for personalized learning.

2. Future Enhancements (not included in the prototype):
   - Handwriting analysis for grammar and spelling exercises.
   - Advanced multimodal integration of speech, writing, and emotion data.

### Tech Stack
- Speech Processing: PyAudio, librosa
- Emotion Detection: OpenCV, DeepFace
- Programming Language: Python
- Platform: macOS (tested)

### How It Works
1. The user speaks into a microphone to complete language exercises.
2. The system evaluates pronunciation errors using audio analysis. 
3. A webcam detects the user's emotional state (e.g., frustration or confidence).
4. Based on performance and emotions, the system adjusts the difficulty of subsequent exercises and provides real-time feedback.

### Current Status
The project is in its prototype phase, focusing on speech processing and affect recognition. Future iterations will integrate handwriting analysis and advanced multimodal features.

### Credits
- LibriSpeech dataset (960h of English speech)
- Meta wav2vec 2.0 ASR model

### Design Considerations
1. IPA vs ARPA phonemes
   - ARPA: Easier to compare programmatically, Montreal Forced Alignment (MRA) optimized
   - IPA: More user friendly, human readable
2. Normalization and thresholding of vowel formants
   - Vowel-extrinsic, formant-intrinsic, speaker-intrinsic methods work best
   - Lobanov (max grounding)
   - Gerstman (max/min grounding)