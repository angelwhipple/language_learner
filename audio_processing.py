import random
import string
import pyaudio
import wave
import librosa
import torch
from nltk.corpus import cmudict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from Levenshtein import distance as levenshtein_distance
import parselmouth
from textgrid import TextGrid
import subprocess

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
HERTZ = 16000  # 16 kHz

ARPA_VOWELS = {"IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "AH", "ER", "AW", "AY"}
PUNCTUATION = string.punctuation.replace("'", "")


def run_mfa_alignment():
    print("Running MFA alignment...")
    try:
        result = subprocess.run(
            ["bash", "./setup_mfa.sh"],
            check=True,  # Raise error if script fails
            capture_output=True,  # Capture stdout/stderr
            text=True  # Return string output
        )
        print("Alignment complete.")
    except subprocess.CalledProcessError as e:
        print("MFA Script failed with error:")
        print(e.stderr)


class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.in_stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,
                                     start=False)
        self.out_stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK,
                                      start=False)

    def record(self, seconds=5):
        print("Recording...")
        self.in_stream.start_stream()
        frames = []
        for _ in range(0, int(RATE / CHUNK * seconds)):
            data = self.in_stream.read(CHUNK)
            frames.append(data)
        self.in_stream.stop_stream()
        print("Finished recording.")

        return frames

    def playback(self, frames):
        print("Playing back...")
        self.out_stream.start_stream()
        for frame in frames:
            self.out_stream.write(frame)
        self.out_stream.stop_stream()
        print("Playback complete.")

    def save(self, frames, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

    def destroy(self):
        self.in_stream.close()
        self.out_stream.close()
        self.p.terminate()


class Transcriber:
    def __init__(self):
        self.dict = cmudict.dict()
        # Wav2Vec 2.0: ASR model pre-trained on LibriSpeech (960h of English speech)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # fine-tuned on LJSpeech Phonemes dataset, 94.4m param
        # self.processor = Wav2Vec2Processor.from_pretrained("bookbot/wav2vec2-ljspeech-gruut")
        # self.model = Wav2Vec2ForCTC.from_pretrained("bookbot/wav2vec2-ljspeech-gruut")

    def text_to_phonemes(self, text, filename):
        phonemes = [
            phoneme
            for word in text.split()
            for phoneme in self.dict[word.strip(PUNCTUATION).lower()][0]
        ]
        with open(filename, "w") as f:
            f.write(" ".join(phonemes))
        return Phonemes(phonemes)

    def transcribe(self, audio_file, filename):
        audio, sr = librosa.load(audio_file, sr=HERTZ)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            with open(filename, "w") as f:
                f.write(transcription.lower())


class VocalSample:
    def __init__(self, audio_file, textgrid_file):
        self.sound = parselmouth.Sound(audio_file)
        self.textgrid = TextGrid.fromFile(textgrid_file)
        vowels = []
        phonemes = []

        # Extract f1, f2, duration from MRA audio textgrid
        for interval in self.textgrid.getFirst("phones"):
            label = interval.mark.strip().upper()
            if label:
                phonemes.append(label)
                base_phoneme = ''.join([c for c in label if not c.isdigit()])
                if base_phoneme in ARPA_VOWELS:
                    duration = interval.maxTime - interval.minTime
                    midpoint = (interval.minTime + interval.maxTime) / 2
                    formants = self.sound.to_formant_burg(time_step=0.01)
                    f1 = formants.get_value_at_time(1, midpoint)
                    f2 = formants.get_value_at_time(2, midpoint)
                    vowels.append(Vowel(label, f1, f2, duration))

        self.phonemes = Phonemes(phonemes)
        self.vowels = Vowels(vowels)


class Phonemes:
    def __init__(self, phonemes):
        self.phonemes = phonemes

    def compare(self, other):
        return levenshtein_distance(self.phonemes, other.phonemes)

    def __str__(self):
        return " ".join(self.phonemes)


class Vowels:
    def __init__(self, vowel_sequence):
        self.vowels = vowel_sequence

    def normalize_vowel_formants(self):
        pass

    def evaluate_formants(self):
        pass

    def __str__(self):
        out = ''
        for vowel in self.vowels:
            out += f'{vowel.label}, F1: {vowel.f1}, F2: {vowel.f2}, DUR: {vowel.duration}\n'
        return


class Vowel:
    def __init__(self, label, f1, f2, duration):
        self.label = label
        self.f1 = f1
        self.f2 = f2
        self.duration = duration


if __name__ == "__main__":
    with open("resources/sample_sentences.txt", "r") as f:
        sentences = f.read().split("\n")
    recorder = AudioRecorder()
    transcriber = Transcriber()

    while True:
        sentence = sentences[random.randint(0, len(sentences) - 1)]
        print(f'\nYour sentence: {sentence}\n')

        user_input = input("Q to quit program. R to start/stop recording: ")
        if user_input == 'Q':
            recorder.destroy()
            break
        elif user_input == 'R':
            audio = recorder.record()
            recorder.playback(audio)
            recorder.save(audio, 'audio_sample.wav')
            expected = transcriber.text_to_phonemes(sentence, 'expected_phonemes.lab')
            transcriber.transcribe('audio_sample.wav', 'audio_sample.lab')
            run_mfa_alignment()
            sample = VocalSample('audio_sample.wav', 'audio_sample.TextGrid')
            print(sample.phonemes)
            print(sample.vowels)

            error = expected.compare(sample.phonemes)
            print(f'Phoneme error (Levenshtein distance): {error}')
