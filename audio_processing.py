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
from scipy.io import wavfile
import matplotlib.pyplot as plt
from panphon import FeatureTable
from pathlib import Path
from panphon.distance import Distance
import csv
from typing import Literal
import os

import main

HEX_GREEN = "00ff00"
HEX_RED = "ff0000"
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100    # 44.1 kHz
HERTZ = 16000  # 16 kHz

ARPA_VOWELS = {"IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "AH", "ER"}  # missing AW, AY
ARPA_TO_IPA = {
    # Vowels
    "AA0": "ɑ", "AA1": "ɑ́", "AA2": "ɑ̀",
    "AE0": "æ", "AE1": "ǽ", "AE2": "æ̀",
    "AH0": "ə", "AH1": "ʌ́", "AH2": "ʌ̀",
    "AO0": "ɔ", "AO1": "ɔ́", "AO2": "ɔ̀",
    "AW0": "aʊ", "AW1": "aʊ́", "AW2": "aʊ̀",
    "AY0": "aɪ", "AY1": "aɪ́", "AY2": "aɪ̀",
    "EH0": "ɛ", "EH1": "ɛ́", "EH2": "ɛ̀",
    "ER0": "ɚ", "ER1": "ɝ́", "ER2": "ɝ̀",
    "EY0": "eɪ", "EY1": "eɪ́", "EY2": "eɪ̀",
    "IH0": "ɪ", "IH1": "ɪ́", "IH2": "ɪ̀",
    "IY0": "i", "IY1": "í", "IY2": "ì",
    "OW0": "oʊ", "OW1": "oʊ́", "OW2": "oʊ̀",
    "OY0": "ɔɪ", "OY1": "ɔɪ́", "OY2": "ɔɪ̀",
    "UH0": "ʊ", "UH1": "ʊ́", "UH2": "ʊ̀",
    "UW0": "u", "UW1": "ú", "UW2": "ù",

    # Consonants
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f",
    "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"
}

PUNCTUATION = string.punctuation.replace("'", "")
ft = FeatureTable()

def is_ipa_vowel(phone: str) -> bool:
    """
    Return True if `phone` carries the syllabic (vowel) feature.
    PanPhon’s fts() returns a dict of {feature_name: -1/0/+1}.
    """
    # 1) normalize to NFD so combining diacritics are separate
    phone = FeatureTable.normalize(phone).strip()
    # 2) split into PanPhon‑known segments
    segs = ft.ipa_segs(phone, normalize=False)
    # 3) any segment with syl=+1 is a vowel
    for seg in segs:
        feats = ft.fts(seg, normalize=False)
        if feats.get("syl", 0) == 1:
            return True
    return False


def arpa_to_ipa(arpa_phoneme):
    return ARPA_TO_IPA.get(arpa_phoneme, None)


def load_mfa_lexicon():
    mfa_root = Path(os.environ.get("MFA_ROOT_DIR", "~/Documents/MFA")).expanduser()
    dict_path = mfa_root / "pretrained_models" / "dictionary" / "english_mfa.dict"
    lex = {}
    with open(dict_path, encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word, phones = parts[0].lower(), parts[1:]
            lex[word] = phones
    return lex


def run_mfa_alignment():
    try:
        result = subprocess.run(
            ["bash", "./setup_mfa.sh"],
            check=True,  # Raise error if script fails
            capture_output=True,  # Capture stdout/stderr
            text=True  # Return string output
        )
    except subprocess.CalledProcessError as e:
        print(f"MFA Script failed with error: {e.stderr}")


def validate_vowel_coverage(vowels):
    provided = set([vowel.label for vowel in vowels.vowels])
    missing = []
    for vowel in ARPA_VOWELS:
        if vowel not in provided:
            missing.append(vowel)
    return missing


def generate_spectrogram(audio_path):
    sample_rate, samples = wavfile.read(f"audio/{audio_path}.wav")

    plt.figure(figsize=(10, 4))
    plt.specgram(samples, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='inferno')
    plt.xlabel(f"Time [s]")
    plt.ylabel(f"Frequency [Hz]")
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.savefig(f"spectrogram/{audio_path}")
    print("Spectrogram generated.")


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

        return frames

    def playback(self, frames):
        print("Playing back...")
        self.out_stream.start_stream()
        for frame in frames:
            self.out_stream.write(frame)
        self.out_stream.stop_stream()

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
        # self.dict = load_mfa_lexicon()
        # Wav2Vec 2.0: ASR model pre-trained on LibriSpeech (960h of English speech)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def text_to_phonemes(self, text):
        words, phonemes_by_word = [], []
        for word in text.split():
            key = word.strip(PUNCTUATION).lower()
            phonemes = self.dict[word.strip(PUNCTUATION).lower()][0]
            # phonemes = self.dict.get(key, [])
            words.append(word)
            phonemes_by_word.append(Phonemes(phonemes))
        return words, phonemes_by_word

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
        self.words = []
        word_intervals = []

        # Extract f1, f2, duration from MRA audio textgrid
        for interval in self.textgrid.getFirst("words"):
            word = interval.mark.strip().lower()
            if word:
                word_intervals.append((word, interval.minTime, interval.maxTime))

        for word, word_start, word_end in word_intervals:
            phonemes, vowels = [], []
            for interval in self.textgrid.getFirst("phones"):
                phoneme = interval.mark.strip().upper()
                if phoneme and interval.minTime >= word_start and interval.maxTime <= word_end:
                    phonemes.append(phoneme)
                    # if is_ipa_vowel(phoneme):
                    base_phoneme = ''.join([c for c in phoneme if not c.isdigit()])
                    if base_phoneme in ARPA_VOWELS:
                        duration = interval.maxTime - interval.minTime
                        midpoint = (interval.minTime + interval.maxTime) / 2
                        formants = self.sound.to_formant_burg(time_step=0.01)
                        f1 = formants.get_value_at_time(1, midpoint)
                        f2 = formants.get_value_at_time(2, midpoint)
                        vowels.append(Vowel(base_phoneme, phoneme, f1, f2, duration))
            self.words.append(Word(word, phonemes, vowels, word_start, word_end))

    def evaluate_pronunciation(self, ref_words, ref_phonemes):
        bit_errors, readable_fb, ssml_fb = [], [], []
        missed = []

        for i, (expected, spoken_word) in enumerate(zip(ref_words, self.words)):
            error = ref_phonemes[i].compare(spoken_word.phonemes)
            if error == 0:
                bit_errors.append(0)
            elif error < 0:
                missed.append(expected)
                bit_errors.append(1)
                ssml = f'''
                    <speak>
                    <prosody rate="medium">Your pronunciation of </prosody>
                    <break time="250ms"/>
                    <prosody rate="70%">{expected}</prosody>
                    <break time="250ms"/>
                    <prosody rate="medium"> was unclear.</prosody>
                    </speak>
                    '''
                ssml_fb.append(ssml)
                readable_fb.append(f"Your pronunciation of {expected} was unclear.")
            else:
                missed.append(expected)
                bit_errors.append(1)
                ipa_actual = "".join([arpa_to_ipa(ph) for ph in spoken_word.phonemes.phonemes])
                ipa_expected = "".join([arpa_to_ipa(ph) for ph in ref_phonemes[i].phonemes])
                ssml = f'''
                    <speak>
                    <prosody rate="medium">Instead of </prosody>
                    <break time="250ms"/>
                    <prosody rate="60%"><phoneme alphabet="ipa" ph="{ipa_actual}">{spoken_word.text}</phoneme></prosody>
                    <break time="250ms"/>
                    <prosody rate="medium"> try </prosody>
                    <break time="250ms"/>
                    <prosody rate="60%"><phoneme alphabet="ipa" ph="{ipa_expected}">{expected}</phoneme></prosody>
                    </speak>
                    '''
                ssml_fb.append(ssml)
                readable = (f"Instead of [b][color={HEX_RED}]{spoken_word.text.lower()}[/color][/b]"
                            f" try [b][color={HEX_GREEN}]{expected.lower()}[/color][/b]")
                readable_fb.append(readable)

        if sum(bit_errors):
            readable_fb.insert(0, f"You mispronounced {sum(bit_errors)} word{'s.' if sum(bit_errors) > 1 else '.'}")
            ssml_fb.insert(0, None)
        return bit_errors, readable_fb, ssml_fb, missed


class Word:
    def __init__(self, text, phoneme_arr, vowel_arr, start, end):
        self.text = text
        self.phonemes = Phonemes(phoneme_arr)
        self.vowels = Vowels(vowel_arr)
        self.start_time = start
        self.end_time = end


class Phonemes:
    def __init__(self, phonemes):
        self.phonemes = phonemes
        # self.dist = Distance()

    def compare(self, other):
        # distance = self.dist.fast_levenshtein_distance("".join(self.phonemes), "".join(other.phonemes))
        distance = levenshtein_distance(self.phonemes, other.phonemes)
        if 'SPN' in set(other.phonemes):    # unidentified noise
            return -1
        return distance

    def join_phonemes(self, phoneme_objects):
        combined = [ph for ph in self.phonemes]
        for phoneme_obj in phoneme_objects:
            combined.extend(phoneme_obj.phonemes)
        return Phonemes(combined)

    def __str__(self):
        return " ".join(self.phonemes)


def load_hillenbrand_vowels(sex):
    vowel_arr = []
    with open("resources/hillenbrand_formant_data.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['Vowel']    # e.g. IY
            duration = float(row[f'Duration-{sex}'])
            f1 = float(row[f'F1-{sex}'])
            f2 = float(row[f'F2-{sex}'])
            vowel_arr.append(Vowel(label, f'{label}0', f1, f2, duration))
    return Vowels(vowel_arr)


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
                vowel.phoneme,
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

    def evaluate_formants(self, sex='M', intrinsic=True, threshold=2.0):
        print(f"Running vowel formant analysis...")
        if intrinsic:
            ref_mean = main.load_reference_json("resources/custom_mean_data.json")
            ref_std = main.load_reference_json("resources/custom_std_data.json")
            ref_norms = main.load_reference_json('resources/custom_formant_data.json')
            # Normalize raw F1,F2 by the reference values before comparison
            normalized = self.normalize_z_score(ref_mean["F1"], ref_mean["F2"], ref_std["F1"], ref_std["F2"])
        else:
            hillenbrand = load_hillenbrand_vowels(sex)
            ref_mean, ref_std = hillenbrand.get_mean_formant_values(), hillenbrand.get_formant_std()
            hillenbrand_norms = hillenbrand.normalize_z_score(*ref_mean, *ref_mean)
            ref_norms = hillenbrand_norms.gen_reference_map("resources/hillenbrand_norms.json")
            normalized = self.normalize_z_score(*ref_mean, *ref_std)

        ssml_fb, readable_fb = [], []
        for i, vowel in enumerate(normalized.vowels):
            ipa = arpa_to_ipa(vowel.phoneme)
            diff_f1 = abs(vowel.f1 - ref_norms[f'{vowel.label}-F1'])
            if diff_f1 > threshold and vowel.f1 > 0:
                # High F1 -> Tongue too low/vowel too open
                critique = "Your vowel is too open on "
                feedback = "Try raising your tongue next time"
            elif diff_f1 > threshold and vowel.f1 < 0:
                # Low F1 -> Tongue too high/vowel too closed
                critique = "Your vowel is too closed in "
                feedback = "Try lowering your tongue next time"
            else:
                ssml = f'''
                <speak>
                <prosody rate="medium">Great vowel sound on </prosody>
                <break time="250ms"/>
                <prosody rate="60%"><phoneme alphabet="ipa" ph="{ipa}">{vowel.label}.</phoneme></prosody>
                </speak>
                '''
                ssml_fb.append(ssml)
                readable_fb.append(f"Great vowel sound on {vowel.label}")
                continue

            ssml = f'''
            <speak>
            <prosody rate="medium">{critique}</prosody>
            <break time="500ms"/>
            <prosody rate="x-slow"><phoneme alphabet="ipa" ph="{ipa}">{vowel.label}.</phoneme></prosody>
            <break time="500ms"/>
            <prosody rate="medium">{feedback}</prosody>
            </speak>
            '''
            ssml_fb.append(ssml)
            readable_fb.append(f"{critique} {vowel.label}. {feedback}")

        return readable_fb, ssml_fb

    def gen_reference_data(self):
        mean_f1, mean_f2 = self.get_mean_formant_values()
        f1_std, f2_std = self.get_formant_std()
        max_f1, max_f2 = self.get_max_formant_values()
        std = {"F1": f1_std, "F2": f2_std}
        mean = {"F1": mean_f1, "F2": mean_f2}
        max = {"F1": max_f1, "F2": max_f2}
        with open('resources/custom_std_data.json', 'w') as f:
            json.dump(std, f)
        with open('resources/custom_mean_data.json', 'w') as f:
            json.dump(mean, f)
        with open('resources/custom_max_formants.json', 'w') as f:
            json.dump(max, f)

    def gen_reference_map(self, filename):
        ref = {}
        for vowel in self.vowels:
            ref[f"{vowel.label}-F1"] = vowel.f1
            ref[f"{vowel.label}-F2"] = vowel.f2
        with open(filename, 'w') as f:
            json.dump(ref, f)
        return ref

    def merge_vowels(self, vowel_objects):
        hashmap = {vowel.label: vowel for vowel in self.vowels}
        for vowel_obj in vowel_objects:
            for vowel in vowel_obj.vowels:
                hashmap[vowel.label] = vowel
        return Vowels(hashmap.values())

    def __str__(self):
        return ' '.join([vowel.label for vowel in self.vowels])


class Vowel:
    def __init__(self, label, phoneme, f1, f2, duration):
        self.label = label
        self.phoneme = phoneme
        self.f1 = f1
        self.f2 = f2
        self.duration = duration

    def __str__(self):
        return f"{self.label} - F1:{self.f1}, F2: {self.f2}, ms: {self.duration}"
