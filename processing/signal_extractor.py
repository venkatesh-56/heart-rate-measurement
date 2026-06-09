import numpy as np

class SignalExtractor:

    def extract(self, roi):

        if roi.size == 0:
            return 0

        # Use only upper face region
        h, w, _ = roi.shape

        forehead = roi[
            int(h*0.1):int(h*0.3),
            int(w*0.3):int(w*0.7)
        ]

        green_signal = np.mean(
            forehead[:, :, 1]
        )

        return green_signal