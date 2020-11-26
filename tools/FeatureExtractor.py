import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn import preprocessing

def VisualizeWaveForm(x, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.title("Waveform")
    plt.show()


def VisualizeSpectrogram(x, sr):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()


def Extract(x,sr):
    zero_crossings = np.mean(librosa.zero_crossings(x, pad=False))
    chroma_stft = np.mean(librosa.feature.chroma_stft(x, sr=sr))
    spec_cent = np.mean(librosa.feature.spectral_centroid(x, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(x, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(x, sr=sr))
    mfcc = librosa.feature.mfcc(x, sr=sr)
    features = [zero_crossings, chroma_stft, spec_cent, spec_bw, rolloff]

    for entry in mfcc:
        features.append(np.mean(entry))

    return np.array(features)

# accepts a dataframe and applies column wise scaling
def PreprocessData(data):
   scaler = preprocessing.StandardScaler()
   return scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
