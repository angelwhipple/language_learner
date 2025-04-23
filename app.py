import io
from kivy.app import App
from kivy.properties import StringProperty, NumericProperty
import audio_processing as audio_proc
import video_processing as video_proc
import warnings
import logging
import main
from contextlib import redirect_stdout
import threading
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2


HEX_GREEN = "00ff00"
HEX_RED = "ff0000"
DEFAULT_PROMPT = "To practice your pronunciation, start recording whenever you're ready."


class ReactiveBuffer(io.StringIO):
    def __init__(self, on_write_callback):
        super().__init__()
        self.on_write_callback = on_write_callback

    def write(self, s):
        result = super().write(s)
        if self.on_write_callback:
            self.on_write_callback(self.getvalue())
        return result


class LiveFeed(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.playback_frames = None
        self.playback_index = 0
        self.playing_back = False
        Clock.schedule_interval(self.update, 1.0 / 30)  # ~30 fps

    def update(self, dt):
        if self.playing_back and self.playback_frames:
            if self.playback_index < len(self.playback_frames):
                frame = self.playback_frames[self.playback_index]
                self.playback_index += 1
            else:
                self.playing_back = False
                return
        else:
            ret, frame = self.capture.read()
            if not ret:
                return

        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture

    def start_playback(self, frames):
        self.playback_frames = frames
        self.playback_index = 0
        self.playing_back = True


class TestApp(App):
    user_name = StringProperty()
    sentence = StringProperty()
    evaluation = StringProperty()
    system_feedback = StringProperty(DEFAULT_PROMPT)
    score = NumericProperty()
    state = StringProperty("WAITING")
    b2_text = StringProperty("Record audio")

    def __init__(self, sentences, user, recorder, transcriber, video_cap, feedback, **kwargs):
        super().__init__(**kwargs)
        self.sentences = sentences
        self.user = user
        self.recorder = recorder
        self.transcriber = transcriber
        self.video_cap = video_cap
        self.feedback = feedback

    def on_quit(self):
        self.recorder.destroy()
        self.video_cap.destroy()
        self.stop()

    def on_b2_press(self):
        if self.state == "WAITING":
            # TODO: UI polish
            #   - gray out/disable record button while process runs
            #   - block until pipe finishes

            self.run_pipeline_async()
            self.state = "EVALUATE"
            self.b2_text = "Next example"
        elif self.state == "EVALUATE":
            self.sentence = main.select_sentence(self.sentences, self.user.get_level())
            self.state = "WAITING"
            self.system_feedback = DEFAULT_PROMPT
            self.b2_text = "Record audio"
            self.evaluation = ""

    def set_system_feedback(self, text):
        self.system_feedback = text

    def visualize_word_accuracy(self, words, errors):
        """
        Colors each word based on pronunciation accuracy.
        :param words:
        :param errors: a [0,1] array indicating error presence for each word
        :return:
        """
        colored_words = []
        for word, err in zip(words, errors):
            color = HEX_RED if err else HEX_GREEN
            colored_words.append(f"[b][color={color}]{word}[/color][/b]")
        self.evaluation = " ".join(colored_words)

    def run_pipeline_async(self):
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        buf = ReactiveBuffer(lambda output: Clock.schedule_once(lambda dt: self.set_system_feedback(output)))
        duration = main.set_recording_duration(user.get_level())
        thread_results = {}

        with redirect_stdout(buf):
            audio_thread = main.run_with_result(
                target=self.recorder.record,
                args=(duration,),
                result_container=thread_results,
                key="audio"
            )
            video_thread = main.run_with_result(
                target=self.video_cap.record,
                args=(duration,),
                result_container=thread_results,
                key="video"
            )
            audio_thread.join()
            video_thread.join()
            audio = thread_results["audio"]
            v_frames, v_timestamps = thread_results["video"]
            live_feed = App.get_running_app().root.ids.live_feed_widget
            audio_thread = main.run_with_result(
                target=self.recorder.playback,
                args=(audio,),
                result_container=thread_results,
                key="audio"
            )
            video_thread = main.run_with_result(
                target=live_feed.start_playback,
                args=(v_frames,),
                result_container=thread_results,
                key="video"
            )
            audio_thread.join()
            video_thread.join()
            self.recorder.save(audio, 'audio/sample.wav')
            self.transcriber.transcribe('audio/sample.wav', 'audio/sample.lab')
            audio_proc.run_mfa_alignment()
            sample = audio_proc.VocalSample('audio/sample.wav', 'audio/sample.TextGrid')

            expected_words, expected_phonemes = self.transcriber.text_to_phonemes(self.sentence)
            errors, readable_feedback, ssml_feedback = sample.evaluate_pronunciation(expected_words, expected_phonemes)

            self.visualize_word_accuracy(expected_words, errors)
            emotion_map = self.feedback.detect_emotion_from_video(self.video_cap, v_frames, v_timestamps, sample.words)
            self.feedback.process_feedback(self.user, errors, readable_feedback, ssml_feedback, emotion_map)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    sentences = main.load_sentences()
    user = main.User()
    recorder = audio_proc.AudioRecorder()
    transcriber = audio_proc.Transcriber()
    video_cap = video_proc.VideoCapture()
    feedback = main.IntegratedFeedbackModule()
    app = TestApp(sentences, user, recorder, transcriber, video_cap, feedback)

    app.user_name = "Whipple"
    app.sentence = main.select_sentence(sentences, user.get_level())
    app.run()

