import audio_processing as audio_proc
import video_processing as video_proc
import threading
import json
import csv
import warnings
import logging
import random
from pytermgui import tim
import boto3
from playsound import playsound
from dotenv import load_dotenv
import os
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def load_text_file(filename):
    with open(filename, "r") as f:
        return f.read()


def load_csv(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data = [row for row in csvreader]
    return header, data


def load_reference_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def load_sentences():
    sentence_dict = {difficulty.upper(): set(load_text_file(f"resources/{difficulty}_sentences.txt").split("\n"))
                     for difficulty in ["easy", "medium", "hard"]}
    return sentence_dict


def select_sentence(sentence_dict, difficulty):
    sentences = sentence_dict[difficulty]
    selected = list(sentences)[random.randint(0, len(sentences) - 1)]
    sentences.remove(selected)
    return selected


def set_recording_duration(difficulty):
    return 3.5 if difficulty == "EASY" else 6 if difficulty == "MEDIUM" else 8


def run_with_result(target, args, result_container, key):
    def wrapper():
        result_container[key] = target(*args)

    thread = threading.Thread(target=wrapper)
    thread.start()
    return thread


def speak_feedback(readable_feedback, ssml_feedback, text_type='text'):
    polly = boto3.client("polly",
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                         region_name="us-east-1"
                         )

    print(f"[b]Polly:[/b] {readable_feedback}")
    response = polly.synthesize_speech(
        Text=ssml_feedback if text_type == 'ssml' else readable_feedback,
        TextType=text_type,
        VoiceId='Joanna',  # You can change this to 'Matthew', 'Amy', etc.
        OutputFormat='mp3'
    )
    with open("audio/feedback.mp3", "wb") as file:
        file.write(response['AudioStream'].read())
    playsound("audio/feedback.mp3")


class User:
    pt_thresholds = {"EASY": 0, "MEDIUM": 3, "HARD": 7}

    def __init__(self):
        self.level = "EASY"  # level 1, 2, or 3
        self.points = 0

    def add_points(self, points):
        self.points = min(10, self.points + points)
        self.level = self.get_level_from_points()

    def subtract_points(self, points):
        self.points = max(0, self.points - points)
        self.level = self.get_level_from_points()

    def get_level(self):
        return self.level

    def get_points(self):
        return self.points

    def get_level_from_points(self):
        for level in reversed(self.pt_thresholds):  # Check HARD → EASY
            if self.points >= self.pt_thresholds[level]:
                return level
        return "EASY"  # Fallback


class IntegratedFeedbackModule:
    def get_video_frame_for_word(self, word, video_timestamps):
        midpoint = (word.start_time + word.end_time) / 2
        frame_idx = min(range(len(video_timestamps)), key=lambda i: abs(video_timestamps[i] - midpoint))
        return frame_idx

    def detect_emotion_from_video(self, cap, video_frames, frame_timestamps, transcribed_words):
        word_to_emotion = {}
        for word in transcribed_words:
            frame_idx = self.get_video_frame_for_word(word, frame_timestamps)
            frame = video_frames[frame_idx]
            emotion, confidence = cap.analyze_emotion(frame)
            if confidence > 0.75:
                word_to_emotion[word.text] = emotion
            else:
                word_to_emotion[word.text] = None
        return word_to_emotion

    def is_confused(self, emotion):
        return emotion == 'angry' or emotion == 'disgust' or emotion == 'surprise'

    def process_feedback(self, user, errors, readable_feedback, ssml_feedback, emotions):
        confusion_detected = any(self.is_confused(emote) for emote in emotions.values())
        if sum(errors):
            user.subtract_points(2 if confusion_detected else 1)
            speak_feedback(f"Your pronunciation could use some work.", None)
            for text, ssml in zip(readable_feedback, ssml_feedback):
                speak_feedback(text, ssml, 'ssml' if ssml else 'text')
        elif not sum(errors) and confusion_detected:
            speak_feedback(f"Perfect pronunciation, but you seemed a bit confused on that exercise. Let's "
                           f"do some more practice.", None)
        else:
            user.add_points(2)
            speak_feedback(f"Perfect pronunciation on that exercise! Let's move on.", None)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # ref_std = load_reference_json("resources/custom_std_data.json")
    # ref_mean = load_reference_json("resources/custom_mean_data.json")
    sentence_dict = load_sentences()

    user = User()
    recorder = audio_proc.AudioRecorder()
    transcriber = audio_proc.Transcriber()
    video_capture = video_proc.VideoCapture()
    feedback = IntegratedFeedbackModule()
    thread_results = {}

    while True:
        sentence = select_sentence(sentence_dict, user.get_level())
        tim.print(f"\n[bold green]Your sentence: {sentence}[/bold green]\n")
        tim.print(f"When you're ready, type [bold]R[/bold] to start/stop recording.\n"
                  f"You can type [bold]Q[/bold] at any time to quit.\n")
        user_input = input(f"> ").strip().upper()
        if user_input == 'Q':
            recorder.destroy()
            video_capture.destroy()
            break
        elif user_input == 'R':
            duration = set_recording_duration(user.get_level())
            audio_thread = run_with_result(
                target=recorder.record,
                args=(duration,),
                result_container=thread_results,
                key="audio"
            )
            video_thread = run_with_result(
                target=video_capture.record,
                args=(duration,),
                result_container=thread_results,
                key="video"
            )
            audio_thread.join()
            video_thread.join()
            audio = thread_results["audio"]
            v_frames, v_timestamps = thread_results["video"]
            recorder.playback(audio)
            recorder.save(audio, 'audio/sample.wav')
            audio_proc.generate_spectrogram("sample")

            transcriber.transcribe('audio/sample.wav', 'audio/sample.lab')
            audio_proc.run_mfa_alignment()
            sample = audio_proc.VocalSample('audio/sample.wav', 'audio/sample.TextGrid')

            expected_words, expected_phonemes = transcriber.text_to_phonemes(sentence)
            errors, readable_feedback, ssml_feedback = sample.evaluate_pronunciation(expected_words, expected_phonemes)
            emotion_map = feedback.detect_emotion_from_video(video_capture, v_frames, v_timestamps, sample.words)
            feedback.process_feedback(user, errors, readable_feedback, ssml_feedback, emotion_map)
        else:
            msg = "Input not recognized, please try again.\n"
            speak_feedback(msg, None)
