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

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
HERTZ = 16000  # 16 kHz

ARPA_VOWELS = {"IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "AH", "ER"}  # missing AW, AY
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
        print(f"MFA Script failed with error: {e.stderr}")


def load_csv(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data = [row for row in csvreader]
    return header, data


# def normalize_reference_formants():
#     """
#     Normalizes F1,F2 values for each vowel in the Hillenbrand dataset via linear compression and expansion
#     :returns: Dictionary of "vowel-formant-sex" : normalized value
#     """
#     header, data = load_csv("resources/hillenbrand_formant_data.csv")
#     ref = {}
#     for i in range(3, len(header)):  # for each formant F1,F2
#         formant_values = {data[v][0]: int(data[v][i]) for v in range(len(data))}  # mapping (vowel : F1/F2 val)
#         f_max = max(formant_values.values())
#         for vowel, freq in formant_values.items():
#             ref[f"{vowel}-{header[i]}"] = round(freq / f_max, 2)
#     return ref
#
#
# def evaluate_user_formants(user, ref, vowels, threshold=0.15):
#     normalized_vowels = vowels.normalize_formants(user.max_f1, user.max_f2)
#     fb = []
#     for vowel in normalized_vowels.vowels:
#         d1 = vowel.f1 - ref[f"{vowel.label}-F1-{user.sex}"]
#         d2 = vowel.f2 - ref[f"{vowel.label}-F2-{user.sex}"]
#         if abs(d1) > threshold and d1 > 0:
#             fb.append(f"Vowel too open on {vowel.label}. Try raising your tongue to reduce mouth openness.")
#         elif abs(d1) > threshold and d1 < 0:
#             fb.append(f"Vowel too closed on {vowel.label}. Try lowering your tongue to increase mouth openness.")
#         elif abs(d2) > threshold and d2 > 0:
#             fb.append(f"Vowel unrounded on {vowel.label}. Try retracting your tongue or rounding your lips slightly.")
#         elif abs(d2) > threshold and d2 < 0:
#             fb.append(f"Vowel over-rounded on {vowel.label}. Try advancing your tongue or rounding your lips less.")
#     return fb


def load_reference_mapping(filename):
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

    # def normalize_max_scaling(self, max_f1, max_f2):
    #     normalized = []
    #     for vowel in self.vowels:
    #         normalized.append(Vowel(
    #             vowel.label,
    #             round(vowel.f1 / max_f1, 2),
    #             round(vowel.f2 / max_f2, 2),
    #             vowel.duration
    #         ))
    #     return Vowels(normalized)

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
        var_f1 = sum((vowel.f1 - mean_f1)**2 for vowel in self.vowels) / len(self.vowels)
        var_f2 = sum((vowel.f2 - mean_f2)**2 for vowel in self.vowels) / len(self.vowels)
        return var_f1**0.5, var_f2**0.5

    def compare_to_ref(self, ref, threshold=0.15):
        fb = []
        for vowel in self.vowels:
            d1 = vowel.f1 - ref[f"{vowel.label}-F1"]
            d2 = vowel.f2 - ref[f"{vowel.label}-F2"]
            if abs(d1) > threshold and d1 > 0:
                fb.append(f"Vowel too open on {vowel.label}. Try raising your tongue to reduce mouth openness.")
            elif abs(d1) > threshold and d1 < 0:
                fb.append(f"Vowel too closed on {vowel.label}. Try lowering your tongue to increase mouth openness.")
            elif abs(d2) > threshold and d2 > 0:
                fb.append(f"Vowel unrounded on {vowel.label}. Try retracting your tongue or rounding your lips slightly.")
            elif abs(d2) > threshold and d2 < 0:
                fb.append(f"Vowel over-rounded on {vowel.label}. Try advancing your tongue or rounding your lips less.")
        return fb if fb else ["Great pronunciation on that exercise!"]

    def generate_reference_mapping(self, filename):
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
        self.max_f1, self.max_f2 = None, None
        self.mean_f1, self.mean_f2 = None, None
        self.f1_std, self.f2_std = None, None

    def set_sex(self, sex):
        self.sex = sex

    def set_max_formants(self, max_f1, max_f2):
        self.max_f1, self.max_f2 = max_f1, max_f2

    def set_mean_formants(self, mean_f1, mean_f2):
        self.mean_f1, self.mean_f2 = mean_f1, mean_f2

    def set_formant_std(self, f1_std, f2_std):
        self.f1_std, self.f2_std = f1_std, f2_std


if __name__ == "__main__":
    ref = load_reference_mapping("resources/ref_dict.json")
    with open("resources/sample_sentences.txt", "r") as f:
        sentences = f.read().split("\n")
    recorder = AudioRecorder()
    transcriber = Transcriber()

    user = User()
    calibrating = True
    sentence = "The red bird may see blue boats where good cats walk in the mall."
    while True:
        if calibrating and not user.sex:
            user_input = input(
                "Before we begin, please enter your sex (M/F): "
            ).strip().upper()
        elif calibrating and user.sex:
            user_input = input(
                f"To help the program better understand your speech, please read the following sentence out loud:\n\n"
                f"\"{sentence}\"\n\n"
                f"When you're ready, type R to start recording.\n"
                f"You can type Q at any time to quit.\n\n"
                f"Your input: "
            ).strip().upper()
        else:
            sentence = sentences[random.randint(0, len(sentences) - 1)]
            user_input = input(
                f"\nYour sentence: {sentence}\n\n"
                f"When you're ready, type R to start recording.\n"
                f"You can type Q at any time to quit.\n\n"
                f"Your input: "
            ).strip().upper()

        if user_input == 'Q':
            recorder.destroy()
            break
        elif user_input == 'M' or user_input == 'F':
            user.set_sex(user_input)
        elif user_input == 'R':
            switch = lambda c: 7.5 if c else 5
            audio = recorder.record(switch(calibrating))
            recorder.playback(audio)
            recorder.save(audio, 'audio/sample.wav')
            transcriber.transcribe('audio/sample.wav', 'audio/sample.lab')
            run_mfa_alignment()
            sample = VocalSample('audio/sample.wav', 'audio/sample.TextGrid')
            print(f"\nPhonemes: {sample.phonemes}")
            if calibrating:
                if validate_vowel_coverage(sample.vowels):
                    msg = "Hmm, that wasn’t clear enough. Please read the sentence again slowly and carefully."
                    speak_feedback([msg])
                else:
                    # user.set_max_formants(*sample.vowels.get_max_formant_values())
                    user.set_mean_formants(*sample.vowels.get_mean_formant_values())
                    user.set_formant_std(*sample.vowels.get_formant_std())
                    msg = "That was good, thanks!"
                    speak_feedback([msg])
                    calibrating = False
            else:
                expected = transcriber.text_to_phonemes(sentence, 'expected_phonemes.lab')
                print(f"Expected phonemes: {expected}")
                error = expected.compare(sample.phonemes)
                print(f'Phoneme error (Levenshtein distance): {error}')
                print(f"\nVowels: {sample.vowels}")
                # feedback = evaluate_user_formants(user, ref, sample.vowels)
                normalized_vowels = sample.vowels.normalize_z_score(user.mean_f1, user.mean_f2, user.f1_std, user.f2_std)
                feedback = normalized_vowels.compare_to_ref(ref)
                speak_feedback(feedback)
        else:
            print(f"Input not recognized, please try again.")
