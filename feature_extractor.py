import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from python_speech_features import mfcc, logfbank, delta
from sklearn.preprocessing import minmax_scale

from reader import DbReader

WINLEN = 0.025  # length of the analysis window in seconds
WINSTEP = 0.015  # the step between successive windows in seconds
NFFT = 1700  # the FFT size
TIPS = 1000  # number of elements to cut from the signal
N = 2  # number of preceding and following frames


class Extractor:

    def extract_features(self, signal, rate, winlen=WINLEN, winstep=WINSTEP, nfft=NFFT, n=N):
        # signal = self._scale_signal(signal)  # obniza o 4% !?
        # signal = self._cut_the_tips(signal) # obniza o 1%

        mfcc_values = mfcc(signal=signal, samplerate=rate, winlen=winlen, winstep=winstep, nfft=nfft)
        dmfcc_values = delta(mfcc_values, n)
        result = np.append(mfcc_values, dmfcc_values, axis=1)

        return result

    @staticmethod
    def _scale_signal(signal):
        return minmax_scale(signal, feature_range=(-1, 1), axis=0).astype(np.float32)

    @staticmethod
    def _cut_the_tips(signal):
        return signal[TIPS:-TIPS]


def plot_coefficients(sample, coef):
    for i in range(12):
        command = sample[i][0]
        sig = sample[i][1]
        srate = sample[i][2]
        print(f"command = {command}, sig = {sig}, sig.shape = {sig.shape}, srate = {srate}")

        if coef == 'mfcc':
            feats = mfcc(sig, srate, nfft=NFFT)
        elif coef == 'fbank':
            feats = logfbank(sig, srate, nfft=NFFT)
        elif coef == 'dmfcc':
            mfcc_feat = mfcc(sig, srate, nfft=NFFT)
            feats = delta(mfcc_feat, 2)
        else:
            raise Exception(f"Bad value for argument coef: {coef}")

        plt.title(i + 1)
        plt.subplot(3, 4, i + 1)
        sns.heatmap(feats)
    plt.show()


if __name__ == "__main__":
    reader = DbReader('../Pliki/258118', mode='one', train_size=0.5)

    sample = reader.get_train()
    print(f"len(sample) = {len(sample)}, type = {type(sample)}")

    plot_coefficients(sample, 'mfcc')
    plot_coefficients(sample, 'fbank')
    plot_coefficients(sample, 'dmfcc')
