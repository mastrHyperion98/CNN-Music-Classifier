# DataExtraction.py is the script used to read all our music files and extract the required information into a csv file
# The csv file will be fed into the CNN and it will be used to speed up testing with the neural network.
# Reading all the music files and extracting the data is expected to take a long time.
import csv
import os
import librosa
import numpy as np
from tools import FeatureExtractor
import matplotlib.pyplot as plt

def main():
    FetchDataFeatures()
    print('Extraction Completed')


def FetchDataFeatures():
    header = 'zero_crossing_rate chroma_stft spectral_centroid spectral_bandwidth rolloff tempo'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    dataset_dir = 'dataset'
    genres = os.listdir(path=dataset_dir)
    for genre in genres:
        genre_dir = dataset_dir+"/"+genre
        filenames = os.listdir(genre_dir)
        for filename in filenames:
            audio_path = genre_dir+'/'+filename
            x, sr = librosa.load(audio_path, mono=True, duration=30)
            features = FeatureExtractor.Extract(x, sr, genre)
            to_append = ''
            for feature in features:
                to_append += f' {feature}'

            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        print(f'{genre} has been completed')


def FetchGraphData():
    numRows=128
    numCols=431
    header = ''
    for i in range(1, numRows*numCols+1):
        header += f' db{i}'
    header += ' label'
    header = header.split()

    file = open('dataGraph.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    dataset_dir = 'dataset'
    genres = os.listdir(path=dataset_dir)
    for genre in genres:
        genre_dir = dataset_dir+"/"+genre
        filenames = os.listdir(genre_dir)
        for filename in filenames:
            audio_path = genre_dir+'/'+filename
            x, sr = librosa.load(audio_path, mono=True, duration=10, offset=15)
            mels_spectrogram = librosa.feature.melspectrogram(x, sr, n_mels=128)
            Xdb = librosa.power_to_db(mels_spectrogram, ref=np.max)
            features = np.reshape(Xdb,(numCols*numRows,))
            to_append = ''
            for feature in features:
                to_append += f' {feature}'

            file = open('dataGraph.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        print(f'{genre} has been completed')


if __name__ == '__main__':
    main()