from openai import OpenAI
import os
from main import User


class ChatGPT:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system = (
            "You are a pronunciation tutor. You generate single English sentences targeted to the learner's current "
            "difficulty level. Sentence content should be casual topics like work, school, food, shopping, etc. "
            "Sentence length should vary by difficulty: EASY sentences must be read within 4.5s, MEDIUM within 6s, "
            "and HARD within 8s."
        )

    def build_user(self, user_obj):
        user = (
            f"Learner level: {user_obj.get_level()}\n"
            f"Recently missed words: {user_obj.recently_missed}\n\n"
            f"Current performance (correctly pronounced/total words spoken): {user_obj.accuracy}\n\n"
            "Please output **one**  sentence at this difficulty that contains at least 2-3 occurrences of vowels from "
            "missed words (or similar words) that are appropriate for their level."
        )
        return user

    def request_sentence(self, user):
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.build_user(user)}
            ],
            temperature=0.7,
            max_tokens=60
        )
        sentence = resp.choices[0].message.content.strip()
        return sentence


if __name__ == '__main__':
    chat_gpt = ChatGPT()
    user = User()
    user.update_stats(['The cat jumped out the hat'], ['jumped', 'out'])
    sentence = chat_gpt.request_sentence(user)
    print(sentence)