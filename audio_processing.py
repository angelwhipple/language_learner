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
import time
from scipy.io import wavfile
import matplotlib.pyplot as plt


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
HERTZ = 16000  # 16 kHz

ARPA_VOWELS = {"IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "AH", "ER"}  # missing AW, AY
ARPA_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ"
}
PUNCTUATION = string.punctuation.replace("'", "")


def arpa_to_ipa(arpa_phoneme):
    base = ''.join([c for c in arpa_phoneme if not c.isdigit()])
    return ARPA_TO_IPA.get(base, None)


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

    def text_to_phonemes(self, text):
        words, phonemes_by_word = [], []
        for word in text.split():
            phonemes = self.dict[word.strip(PUNCTUATION).lower()][0]
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
            phonemes = []
            vowels = []
            for interval in self.textgrid.getFirst("phones"):
                phoneme = interval.mark.strip().upper()
                if phoneme and interval.minTime >= word_start and interval.maxTime <= word_end:
                    phonemes.append(phoneme)
                    base_phoneme = ''.join([c for c in phoneme if not c.isdigit()])
                    if base_phoneme in ARPA_VOWELS:
                        duration = interval.maxTime - interval.minTime
                        midpoint = (interval.minTime + interval.maxTime) / 2
                        formants = self.sound.to_formant_burg(time_step=0.01)
                        f1 = formants.get_value_at_time(1, midpoint)
                        f2 = formants.get_value_at_time(2, midpoint)
                        vowels.append(Vowel(base_phoneme, f1, f2, duration))
            self.words.append(Word(word, phonemes, vowels, word_start, word_end))

    def evaluate_pronunciation(self, ref_words, ref_phonemes):
        mispronounced, readable_fb, ssml_fb = [], [], []

        for i, (expected, spoken_word) in enumerate(zip(ref_words, self.words)):
            error, readable, ssml = ref_phonemes[i].compare(spoken_word.phonemes)
            if error >= 2:
                mispronounced.append((expected, error))  # tuple of (word, error)
                readable_fb.append(f"Let's work on your pronunciation of: {expected}")
                readable_fb.extend(readable)
                ssml_fb.append(None)
                ssml_fb.extend(ssml)

        if mispronounced:
            readable_fb.insert(0, f"You mispronounced {len(mispronounced)} words.")
            ssml_fb.insert(0, None)
        return mispronounced, readable_fb, ssml_fb


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

    def compare(self, other):
        readable_fb, ssml_fb = [], []
        distance = levenshtein_distance(self.phonemes, other.phonemes)
        for (expected, actual) in zip(self.get_base_phonemes(), other.get_base_phonemes()):
            ipa_expected, ipa_actual = arpa_to_ipa(expected), arpa_to_ipa(actual)
            if actual == 'SPN':
                ssml = f'''
                <speak>
                <prosody rate="medium">Trying pronouncing </prosody>
                <break time="500ms"/>
                <prosody rate="x-slow"><phoneme alphabet="ipa" ph="{ipa_expected}">{expected}</phoneme></prosody>
                <break time="500ms"/>
                <prosody rate="medium"> a bit clearer.</prosody>
                </speak>
                '''
                readable_fb.append(f"Try pronouncing {expected} a bit clearer.")
                ssml_fb.append(ssml)
            elif expected != actual:
                ssml = f'''
                <speak>
                <prosody rate="medium">Instead of </prosody>
                <break time="500ms"/>
                <prosody rate="x-slow"><phoneme alphabet="ipa" ph="{ipa_actual}">{actual}</phoneme></prosody>
                <break time="500ms"/>
                <prosody rate="medium"> try </prosody>
                <break time="500ms"/>
                <prosody rate="x-slow"><phoneme alphabet="ipa" ph="{ipa_expected}">{ipa_expected}</phoneme></prosody>
                </speak>
                '''
                readable_fb.append(f"Instead of {actual}, try {expected}.")
                ssml_fb.append(ssml)
        return distance, readable_fb, ssml_fb

    def get_base_phonemes(self):
        return [''.join([c for c in phoneme if not c.isdigit()]) for phoneme in self.phonemes]

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
        ssml_fb = []
        readable_fb = []
        for vowel in normalized.vowels:
            if abs(vowel.f1) > threshold or abs(vowel.f2) > threshold:
                ipa = arpa_to_ipa(vowel.label)
                ssml = f'''
                <speak>
                <prosody rate="medium">Your pronunciation of </prosody>
                <break time="500ms"/>
                <prosody rate="x-slow"><phoneme alphabet="ipa" ph="{ipa}">{vowel.label}</phoneme></prosody>
                <break time="500ms"/>
                <prosody rate="medium"> needs some work.</prosody>
                </speak>
                '''
                ssml_fb.append(ssml)
                readable_fb.append(f"Your pronunciation of {vowel.label} needs some work.")
        return readable_fb, ssml_fb

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
