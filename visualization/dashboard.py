import cv2

class Dashboard:

    def draw(self, frame, bpm):

        cv2.putText(
            frame,
            f"BPM: {bpm}",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        return frame