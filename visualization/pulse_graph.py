import matplotlib.pyplot as plt

class PulseGraph:

    def show(self, signal):

        plt.clf()

        plt.title("Pulse Signal")

        plt.plot(signal)

        plt.pause(0.001)