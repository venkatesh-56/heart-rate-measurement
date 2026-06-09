import cv2

class EyeDetector:

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_eye.xml"
        )

    def detect(self, face_roi):

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        return eyes