from scipy.signal import butter, filtfilt
import numpy as np

class SignalFilter:

    def filter_signal(self, signal):

        signal = np.array(signal)

        if len(signal) < 60:
            return signal

        signal = signal - np.mean(signal)

        b, a = butter(
            3,
            [0.7/15, 3.5/15],
            btype='bandpass'
        )

        filtered = filtfilt(
            b,
            a,
            signal
        )

        return filtered