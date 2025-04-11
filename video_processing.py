import cv2


class VideoCapture:
    def __init__(self, index=0):
        self.camera = cv2.VideoCapture(index)
        if not self.camera.isOpened():
            raise SystemError('Failed to open camera. Try allowing the program to access your camera, then try again.')

    def destroy(self):
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = VideoCapture()

    while True:
        ret, frame = cap.camera.read()
        if not ret:
            print("Failed to capture frame.")
        else:
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.destroy()
                break

