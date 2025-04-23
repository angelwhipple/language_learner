import cv2
import time
from deepface import DeepFace


class VideoCapture:
    def __init__(self, index=0):
        self.camera = cv2.VideoCapture(index)
        if not self.camera.isOpened():
            raise SystemError('Failed to open camera. Try granting the program access to your camera, then try again.')

    def record(self, seconds=5.0):
        frames, timestamps = [], []
        start_time = time.time()

        while time.time() - start_time < seconds:
            ret, frame = self.camera.read()
            frame_time = time.time() - start_time
            if not ret:
                raise SystemError('Failed to capture video frame.')

            frames.append(frame)
            timestamps.append(frame_time)

        return frames, timestamps

    def analyze_emotion(self, frame):
        # DeepFace expects RGB format, OpenCV gives BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        top_emotion = result[0]['dominant_emotion']
        confidence = result[0]['face_confidence']

        return top_emotion, confidence

    def destroy(self):
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = VideoCapture()
    cap.destroy()

    while True:
        frames, timestamps = cap.record(5)
        cv2.imshow('Camera Feed', frames[-1])
        cap.analyze_emotion(frames[-1])
        cv2.waitKey(0)

