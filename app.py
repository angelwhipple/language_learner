import io
from kivy.app import App
from kivy.lang import Builder
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
from kivy.uix.settings import SettingsWithSidebar
import cv2
import time


HEX_GREEN = "00ff00"
HEX_RED = "ff0000"
DEFAULT_PROMPT = "To practice your pronunciation, start recording whenever you're ready."


class ReactiveBuffer(io.StringIO):
    def __init__(self, on_write_callback):
        super().__init__()
        self.on_write_callback = on_write_callback
        self._temp_buffer = ""

    def write(self, s):
        self._temp_buffer += s
        if '\n' in self._temp_buffer:
            lines = self._temp_buffer.split('\n')
            for line in lines[:-1]:  # all complete lines
                if self.on_write_callback:
                    Clock.schedule_once(lambda dt, l=line: self.on_write_callback(l))
            self._temp_buffer = lines[-1]  # keep partial line in buffer
        return len(s)


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

        frame = cv2.flip(frame, -1)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture

    def start_playback(self, frames):
        self.playback_frames = frames
        self.playback_index = 0
        self.playing_back = True


class TestApp(App):
    username = StringProperty()
    sentence = StringProperty()
    evaluation = StringProperty()
    system_feedback = StringProperty(DEFAULT_PROMPT)
    state = StringProperty("WAITING")
    difficulty = StringProperty("EASY")
    score = NumericProperty(0)

    def __init__(self, sentences, user, recorder, transcriber, video_cap, feedback, **kwargs):
        super().__init__(**kwargs)
        self.sentences = sentences
        self.user = user
        self.recorder = recorder
        self.transcriber = transcriber
        self.video_cap = video_cap
        self.feedback = feedback

    def build(self):
        self.settings_cls = SettingsWithSidebar
        return Builder.load_file("test.kv")

    def build_config(self, config):
        config.setdefaults('general', {
            'username': 'Whipple',
            'normalization': 'Z-score'
        })
        self.username = self.config.get('general', 'username')

    def build_settings(self, settings):
        settings.add_json_panel('General', self.config, data="""
        [
            {
                "type": "string", 
                "title": "Username", 
                "desc": "Enter your username", 
                "section": "general", 
                "key": "username"
            },
            {
                "type": "options", 
                "title": "Normalization method", 
                "desc": "Choose a normalization method for vowel analysis", 
                "section": "general", 
                "key": "normalization",
                "options": ["Z-score", "Max-scaling"]
            }
        ]
        """)

    def on_config_change(self, config, section, key, value):
        print(f'Config changed: [{section}] {key} = {value}')
        if section == 'general' and key == 'username':
            self.username = value

    def open_settings_panel(self):
        self.open_settings()

    def on_quit(self):
        self.recorder.destroy()
        self.video_cap.destroy()
        self.stop()

    def on_b2_press(self):
        button_2 = App.get_running_app().root.ids.button_2
        if self.state == "WAITING":
            self.state = "EVALUATE"
            self.run_pipeline_async()
            button_2.text = "Next example"
            button_2.disabled = True
        elif self.state == "EVALUATE":
            self.state = "WAITING"
            self.system_feedback = DEFAULT_PROMPT
            self.sentence = main.select_sentence(self.sentences, self.user.get_level())
            self.evaluation = ""
            button_2.text = "Record audio"

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
        thread_results = {}

        with redirect_stdout(buf):
            duration = main.set_recording_duration(self.user.get_level())
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

            print('Running audiovisual analysis...')
            start_time = time.time()
            audio_proc.run_mfa_alignment()
            sample = audio_proc.VocalSample('audio/sample.wav', 'audio/sample.TextGrid')
            expected_words, expected_phonemes = self.transcriber.text_to_phonemes(self.sentence)
            errors, readable_feedback, ssml_feedback = sample.evaluate_pronunciation(expected_words, expected_phonemes)
            self.visualize_word_accuracy(expected_words, errors)
            emotion_map = self.feedback.detect_emotion_from_video(self.video_cap, v_frames, v_timestamps, sample.words)
            elapsed = time.time() - start_time
            print(f"Analysis complete in {elapsed:.2f} seconds.")
            self.feedback.process_feedback(self.user, errors, readable_feedback, ssml_feedback, emotion_map)
            self.score = self.user.get_points()
            self.difficulty = self.user.get_level()

            button_2 = App.get_running_app().root.ids.button_2
            button_2.disabled = False


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

    app.sentence = main.select_sentence(sentences, user.get_level())
    app.run()

