from kivy.app import App
from kivy.properties import StringProperty, NumericProperty


class TestApp(App):
    user_name = StringProperty()
    sentence = StringProperty()
    evaluation = StringProperty()
    score = NumericProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_sentence_evaluation(self):
        # TODO: color words based on [0,1] pronunciation correctness
        pass


if __name__ == '__main__':
    app = TestApp()
    app.user_name = "Whipple"
    app.sentence = "The orchestra performed a flawless rendition of the piece."
    app.run()
