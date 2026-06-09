import matplotlib.pyplot as plt

class FrequencyGraph:

    def show(self, freqs, magnitudes):

        plt.clf()

        plt.title("Frequency Spectrum")

        plt.plot(freqs, magnitudes)

        plt.pause(0.001)