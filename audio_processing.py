import csv
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


def normalize_reference_formants():
    """
    Normalizes F1,F2 values for each vowel in the reference dataset via linear compression and expansion
    :returns: Dictionary of "vowel-formant-sex" : normalized value
    """
    header, data = load_csv("resources/vowel-formant-data.csv")
    ref = {}
    for i in range(3, len(header)):  # for each formant F1,F2
        formant_values = {data[v][0]: int(data[v][i]) for v in range(len(data))}  # mapping (vowel : F1/F2 val)
        f_max = max(formant_values.values())
        for vowel, freq in formant_values.items():
            ref[f"{vowel}-{header[i]}"] = round(freq / f_max, 2)
    return ref


def evaluate_user_formants(user, ref, vowels, threshold=0.1):
    normalized_vowels = vowels.normalize_formants(user.max_f1, user.max_f2)
    for vowel in normalized_vowels:
        d1 = abs(vowel.f1 - ref[f"{vowel.label}-F1-{user.sex}"])
        d2 = abs(vowel.f2 - ref[f"{vowel.label}-F2-{user.sex}"])
        if d1 > threshold or d2 > threshold:
            print(f'Mispronounced vowel: {vowel}')


def validate_vowel_coverage(phonemes):
    phonemes = set([''.join([char for char in phoneme if not char.isdigit()]) for phoneme in phonemes])
    for vowel in ARPA_VOWELS:
        if vowel not in phonemes:
            return False
    return True


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

    def normalize_formants(self, max_f1, max_f2):
        normalized_vowels = []
        for vowel in self.vowels:
            normalized_vowels.append(Vowel(
                vowel.label,
                vowel.f1 / max_f1,
                vowel.f2 / max_f2,
                vowel.duration
            ))
        return Vowels(normalized_vowels)

    def get_max_formant_values(self):
        return max([vowel.f1 for vowel in self.vowels]), max([vowel.f2 for vowel in self.vowels])

    def get_min_formant_values(self):
        return min([vowel.f1 for vowel in self.vowels]), min([vowel.f2 for vowel in self.vowels])

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


class User:
    def __init__(self):
        self.sex = None
        self.max_f1 = None
        self.max_f2 = None

    def set_sex(self, sex):
        self.sex = sex

    def set_max_formants(self, max_f1, max_f2):
        self.max_f1, self.max_f2 = max_f1, max_f2


if __name__ == "__main__":
    ref = normalize_reference_formants()
    # print(ref)

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
                f"To get things set up, please read the following sentences out loud:\n\n"
                f"\"{sentence}\"\n\n"
                f"This will help the program better understand your speech.\n"
                f"When you're ready, type **R** to start recording.\n"
                f"You can type **Q** at any time to quit.\n\n"
                f"Your input: "
            )
        else:
            sentence = sentences[random.randint(0, len(sentences) - 1)]
            user_input = input(
                f"\nYour sentence: {sentence}\n"
                f"Q to quit program. R to start/stop recording: "
            )

        if user_input == 'Q':
            recorder.destroy()
            break
        elif user_input == 'M' or user_input == 'F':
            user.set_sex(user_input)
        elif user_input == 'R':
            switch = lambda c: 7.5 if c else 5
            audio = recorder.record(switch(calibrating))
            recorder.playback(audio)
            recorder.save(audio, 'audio_sample.wav')
            transcriber.transcribe('audio_sample.wav', 'audio_sample.lab')
            run_mfa_alignment()
            sample = VocalSample('audio_sample.wav', 'audio_sample.TextGrid')
            print(sample.phonemes)
            if calibrating:
                if not validate_vowel_coverage(sample.phonemes):
                    print("Hmm, that wasn’t clear enough. Please read the sentence again slowly and carefully.")
                else:
                    user.set_max_formants(*sample.vowels.get_max_formant_values())
                    calibrating = False
            else:
                expected = transcriber.text_to_phonemes(sentence, 'expected_phonemes.lab')
                error = expected.compare(sample.phonemes)
                print(f'Phoneme error (Levenshtein distance): {error}')
                evaluate_user_formants(user, ref, sample.vowels)
        else:
            print(f"Input not recognized, please try again.")
