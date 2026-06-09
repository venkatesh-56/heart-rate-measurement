import numpy as np

class FFTProcessor:

    def process(self, signal, fps):

        n = len(signal)

        fft = np.fft.rfft(signal)

        freqs = np.fft.rfftfreq(n, d=1/fps)

        return freqs, abs(fft)