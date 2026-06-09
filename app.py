import cv2
import numpy as np
import matplotlib.pyplot as plt

from detectors.face_detector import FaceDetector
from detectors.eye_detector import EyeDetector

from processing.signal_extractor import SignalExtractor
from processing.signal_filter import SignalFilter
from processing.fft_processor import FFTProcessor
from processing.bpm_calculator import BPMCalculator

from visualization.dashboard import Dashboard

# -----------------------------
# Initialize Modules
# -----------------------------
face_detector = FaceDetector()
eye_detector = EyeDetector()

extractor = SignalExtractor()
filter_obj = SignalFilter()
fft_obj = FFTProcessor()
bpm_obj = BPMCalculator()

dashboard = Dashboard()

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

signal = []

fps = 30
bpm = 0

# -----------------------------
# Matplotlib Setup
# -----------------------------
plt.ion()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = face_detector.detect(frame)

    for (x, y, w, h) in faces:

        # Face Box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        face_roi = frame[
    y:y+int(h*0.4),
    x:x+w
]
        # Eye Detection
        eyes = eye_detector.detect(face_roi)

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                face_roi,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

        # Extract Signal
        value = extractor.extract(face_roi)
        print("Signal:", value)
        signal.append(value)

        if len(signal) > 300:
            signal.pop(0)

        # BPM Calculation
        if len(signal) > 100:

            filtered = filter_obj.filter_signal(signal)

            freqs, mags = fft_obj.process(
                filtered,
                fps
            )

            bpm = bpm_obj.calculate(
                freqs,
                mags
            )
            print("BPM:", bpm)
            # -----------------------------
            # Pulse Graph
            # -----------------------------
            ax1.clear()
            ax1.plot(filtered)
            ax1.set_title(
                f"Pulse Signal | BPM: {bpm}"
            )
            ax1.set_xlabel("Samples")
            ax1.set_ylabel("Amplitude")

            # -----------------------------
            # Frequency Graph
            # -----------------------------
            ax2.clear()
            ax2.plot(freqs, mags)
            ax2.set_title(
                "Frequency Spectrum"
            )
            ax2.set_xlabel(
                "Frequency (Hz)"
            )
            ax2.set_ylabel(
                "Magnitude"
            )

            plt.tight_layout()
            plt.pause(0.001)

        # BPM Text
        frame = dashboard.draw(
            frame,
            bpm
        )

    cv2.imshow(
        "Heart Rate Measurement",
        frame
    )

    key = cv2.waitKey(1)

    # ESC Key
    if key == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
plt.close('all')