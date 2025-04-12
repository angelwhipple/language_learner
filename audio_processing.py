import csv
import random
import string
import json
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
import pyttsx3
import time
import pytermgui as ptg
from pytermgui import tim
import warnings
import logging

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
HERTZ = 16000  # 16 kHz

ARPA_VOWELS = {"IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "AH", "ER"}  # missing AW, AY
PUNCTUATION = string.punctuation.replace("'", "")


def run_mfa_alignment():
    print("Running MFA alignment...")
    start_time = time.time()
    try:
        result = subprocess.run(
            ["bash", "./setup_mfa.sh"],
            check=True,  # Raise error if script fails
            capture_output=True,  # Capture stdout/stderr
            text=True  # Return string output
        )
        elapsed = time.time() - start_time
        print(f"Alignment complete in {elapsed:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"MFA Script failed with error: {e.stderr}")


def load_csv(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data = [row for row in csvreader]
    return header, data


def load_reference_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def speak_feedback(feedback_list):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed: 150 words/min
    engine.setProperty('volume', 1)  # Volume 0-1

    for msg in feedback_list:
        print(f"OS: {msg}")
        engine.say(msg)
        engine.runAndWait()


def validate_vowel_coverage(vowels):
    provided = set([vowel.label for vowel in vowels.vowels])
    missing = []
    for vowel in ARPA_VOWELS:
        if vowel not in provided:
            missing.append(vowel)
    return missing


class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.in_stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,
                                     start=False)
        self.out_stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK,
                                      start=False)

    def record(self, seconds=5.0):
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
                    vowels.append(Vowel(base_phoneme, f1, f2, duration))

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

    def normalize_max_scaling(self, max_f1, max_f2):
        normalized = []
        for vowel in self.vowels:
            normalized.append(Vowel(
                vowel.label,
                round(vowel.f1 / max_f1, 2),
                round(vowel.f2 / max_f2, 2),
                vowel.duration
            ))
        return Vowels(normalized)

    def normalize_z_score(self, mean_f1, mean_f2, f1_std, f2_std):
        normalized = []
        for vowel in self.vowels:
            normalized.append(Vowel(
                vowel.label,
                round((vowel.f1 - mean_f1) / f1_std, 2),
                round((vowel.f2 - mean_f2) / f2_std, 2),
                vowel.duration
            ))
        return Vowels(normalized)

    def get_max_formant_values(self):
        return max([vowel.f1 for vowel in self.vowels]), max([vowel.f2 for vowel in self.vowels])

    def get_mean_formant_values(self):
        n = len(self.vowels)
        sum_f1 = sum(vowel.f1 for vowel in self.vowels)
        sum_f2 = sum(vowel.f2 for vowel in self.vowels)
        return sum_f1 / n, sum_f2 / n

    def get_formant_std(self):
        mean_f1, mean_f2 = self.get_mean_formant_values()
        var_f1 = sum((vowel.f1 - mean_f1) ** 2 for vowel in self.vowels) / len(self.vowels)
        var_f2 = sum((vowel.f2 - mean_f2) ** 2 for vowel in self.vowels) / len(self.vowels)
        return var_f1 ** 0.5, var_f2 ** 0.5

    def compare_to_ref(self, ref_mean, ref_std, threshold=2.5):
        # Normalize raw F1,F2 by the reference values before comparison
        normalized = self.normalize_z_score(ref_mean["F1"], ref_mean["F2"], ref_std["F1"], ref_std["F2"])
        fb = []
        for vowel in normalized.vowels:
            if abs(vowel.f1) > threshold or abs(vowel.f2) > threshold:
                fb.append(f"Your pronunciation of {vowel.label} needs work.")
        return fb if fb else ["Great pronunciation on that exercise!"]

    def gen_reference_data(self):
        mean_f1, mean_f2 = self.get_mean_formant_values()
        f1_std, f2_std = self.get_formant_std()
        std = {"F1": f1_std, "F2": f2_std}
        mean = {"F1": mean_f1, "F2": mean_f2}
        with open('resources/custom_std_data.json', 'w') as f:
            json.dump(std, f)
        with open('resources/custom_mean_data.json', 'w') as f:
            json.dump(mean, f)

    def gen_reference_map(self, filename):
        ref = {}
        for vowel in self.vowels:
            ref[f"{vowel.label}-F1"] = vowel.f1
            ref[f"{vowel.label}-F2"] = vowel.f2
        with open(filename, 'w') as f:
            json.dump(ref, f)

    def __str__(self):
        return ' '.join([vowel.label for vowel in self.vowels])


class Vowel:
    def __init__(self, label, f1, f2, duration):
        self.label = label
        self.f1 = f1
        self.f2 = f2
        self.duration = duration

    def __str__(self):
        return f"{self.label} - F1:{self.f1}, F2: {self.f2}, ms: {self.duration}"


class User:
    def __init__(self):
        self.sex = None

    def set_sex(self, sex):
        self.sex = sex


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    ref_std = load_reference_json("resources/custom_std_data.json")
    ref_mean = load_reference_json("resources/custom_mean_data.json")
    with open("resources/sample_sentences.txt", "r") as f:
        sentences = f.read().split("\n")

    recorder = AudioRecorder()
    transcriber = Transcriber()
    user = User()
    calibrating = True
    sentence = "He sits near red ants. Dogs saw two blue and grey boats. Her son could hum by the pier."
    while True:
        if calibrating:
            tim.print(f"To help the program better understand your speech, please read the following out loud:\n\n"
                      f"[bold green]{sentence}[/bold green]\n")
        else:
            sentence = sentences[random.randint(0, len(sentences) - 1)]
            tim.print(f"\n[bold green]Your sentence: {sentence}[/bold green]\n")

        tim.print(f"When you're ready, type [bold]R[/bold] to start recording.\n"
                  f"You can type [bold]Q[/bold] at any time to quit.\n")
        user_input = input(f"> ").strip().upper()
        if user_input == 'Q':
            recorder.destroy()
            break
        elif user_input == 'R':
            switch = lambda c: 10 if c else 5
            audio = recorder.record(switch(calibrating))
            recorder.playback(audio)
            recorder.save(audio, 'audio/sample.wav')
            transcriber.transcribe('audio/sample.wav', 'audio/sample.lab')
            run_mfa_alignment()
            sample = VocalSample('audio/sample.wav', 'audio/sample.TextGrid')
            if calibrating:
                if validate_vowel_coverage(sample.vowels):
                    msg = "Hmm, that wasn’t clear enough. Please read the sentence again slowly and carefully."
                    speak_feedback([msg])
                else:
                    msg = "That was good, thanks!"
                    speak_feedback([msg])
                    calibrating = False
            else:
                print(f"\nPhonemes: {sample.phonemes}")
                expected = transcriber.text_to_phonemes(sentence, 'expected_phonemes.lab')
                print(f"Expected phonemes: {expected}")
                error = expected.compare(sample.phonemes)
                print(f'Phoneme error (Levenshtein distance): {error}')
                print(f"\nVowels: {sample.vowels}")
                # feedback = evaluate_user_formants(user, ref, sample.vowels)
                feedback = sample.vowels.compare_to_ref(ref_mean, ref_std)
                speak_feedback(feedback)
        else:
            msg = "Input not recognized, please try again.\n"
            speak_feedback([msg])
