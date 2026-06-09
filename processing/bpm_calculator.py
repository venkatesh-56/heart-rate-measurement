import numpy as np

class BPMCalculator:

    def calculate(
        self,
        freqs,
        magnitudes
    ):

        mask = (
            (freqs >= 0.7)
            &
            (freqs <= 3.5)
        )

        if not np.any(mask):
            return 0

        peak_freq = freqs[mask][
            np.argmax(
                magnitudes[mask]
            )
        ]

        bpm = peak_freq * 60

        return round(bpm)