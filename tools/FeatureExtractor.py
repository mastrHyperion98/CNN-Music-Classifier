import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn import preprocessing

def visualize_waveform(x, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.title("Waveform")
    plt.show()


def visualize_spectrogram(x, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(x, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()


def extract(x,sr, genre):
    zero_crossings = np.mean(librosa.zero_crossings(x, pad=False))
    chroma_stft = np.mean(librosa.feature.chroma_stft(x, sr=sr))
    spec_cent = np.mean(librosa.feature.spectral_centroid(x, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(x, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(x, sr=sr))
    tempo, beats = librosa.beat.beat_track(y=x, sr=sr)
    mfcc = librosa.feature.mfcc(x, sr=sr)
    features = [zero_crossings, chroma_stft, spec_cent, spec_bw, rolloff, tempo]

    for entry in mfcc:
        features.append(np.mean(entry))

    features.append(genre)
    return features
