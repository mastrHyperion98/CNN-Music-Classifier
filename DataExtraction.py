# DataExtraction.py is the script used to read all our music files and extract the required information into a csv file
# The csv file will be fed into the CNN and it will be used to speed up testing with the neural network.
# Reading all the music files and extracting the data is expected to take a long time.
import csv
import os
from random import random

import librosa
import numpy as np
from tools import FeatureExtractor

def main():
    fetch_data_features('Data_800_10sec', 10)
    print('Extraction Completed')


def fetch_data_features(output, duration):
    """Writes the features the specified csv file"""
    header = 'zero_crossing_rate chroma_stft spectral_centroid spectral_bandwidth rolloff tempo'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open(f'{output}.csv', 'w', newline='')
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
            x, sr = librosa.load(audio_path, mono=True, duration=duration)
            features = FeatureExtractor.extract(x, sr, genre)
            to_append = ''
            for feature in features:
                to_append += f' {feature}'

            file = open(output, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        print(f'{genre} has been completed')


def fetch_graph_data(output, duration):
    """Writes the specified offset and duration of the spectrogram into a csv file to be read and imported into Pandas"""
    numRows = 128
    numCols = round(431/10 * duration)
    header = ''
    for i in range(1, numRows*numCols+1):
        header += f' db{i}'
    header += ' label'
    header = header.split()

    file = open(f'{output}.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # Traverse through the directory and recursively extract songs data from each folder.
    # Folder names are the genres
    # for copy right reasons we avoid adding the name of the tracks to the csv file.
    dataset_dir = 'dataset'
    genres = os.listdir(path=dataset_dir)
    for genre in genres:
        genre_dir = dataset_dir+"/"+genre
        if os.path.isdir(genre_dir):
            filenames = os.listdir(genre_dir)

            for filename in filenames:
                if ".ogg" in filename:
                    audio_path = genre_dir+'/'+filename
                    offset_int = random.randint(30, 90)
                    print('Audio path: ' + audio_path)
                    x, sr = librosa.load(audio_path, mono=True, duration=duration, offset=offset_int)
                    mels_spectrogram = librosa.feature.melspectrogram(x, sr, n_mels=128)
                    Xdb = librosa.power_to_db(mels_spectrogram, ref=np.max)
                    # reshape so that it fits on one row in the  csv
                    features = np.reshape(Xdb, (numCols*numRows,))
                    to_append = ''
                    for feature in features:
                        to_append += f' {feature}'

                    to_append += f' {genre}'
                    file = open(f'{output}.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())

            print(f'{genre} has been completed')


if __name__ == '__main__':
    main()
