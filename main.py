import librosa
from tools import FeatureExtractor
import numpy as np
import pandas as pd


def Sample():
    audio_path = 'samples/track_2.ogg'
    x, sr = librosa.load(audio_path)
    genre = 'rock'
    row1 = FeatureExtractor.Extract(x, sr)
    audio_file = 'samples/track_1.wav'
    x1, sr1 = librosa.load(audio_file)
    row2 = FeatureExtractor.Extract(x1, sr1)
    audio_file = 'samples/track_3.wav'
    x1, sr1 = librosa.load(audio_file)
    row3 = FeatureExtractor.Extract(x1, sr1)
    audio_file = 'samples/track_4.wav'
    x1, sr1 = librosa.load(audio_file)
    row4 = FeatureExtractor.Extract(x1, sr1)
    data = np.vstack((row1, row2))
    data = np.vstack((data, row3))
    data = np.vstack((data, row4))
    dataframe = pd.DataFrame(data)
    X = FeatureExtractor.PreprocessData(dataframe)
    print(X)


def main():
    Sample()


if __name__ == '__main__':
    main()


