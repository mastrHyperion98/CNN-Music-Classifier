"""
data_extraction.py is the script used to read all our music files and extract the
required information into a csv file. The csv file will be fed into the CNN and it
will be used to speed up testing with the neural network.
Reading all the music files and extracting the data is expected to take a long time.
"""
import csv
import random
import os
import warnings
import librosa
import numpy as np
# import matplotlib.pyplot as plt
from tools import FeatureExtractor
warnings.filterwarnings('ignore')


def main():
    FetchGraphData()
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
    num_rows = 128
    num_cols = 1292
    header = ''
    for i in range(1, num_rows*num_cols+1):
        header += f' feature {i}'
    header += ' label'
    header = header.split()

    csv_name = 'dataGraph-10r.csv'
    file = open(csv_name, 'w', newline='')

    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    dataset_dir = 'dataset'
    genres = os.listdir(path=dataset_dir)
    for genre in genres:
        genre_dir = dataset_dir+"/"+genre
        if os.path.isdir(genre_dir):
            filenames = os.listdir(genre_dir)
            index = 0
            counter = 10
            for filename in filenames:
                if ".ogg" in filename:
                    index += 1
                    offset_int = random.randint(30, 90)
                    audio_path = os.path.dirname(os.path.abspath(
                        __file__)) + '/' + genre_dir+'/'+filename
                    print(str(index) + ' Audio path: ' + audio_path + ', offset: ' + str(offset_int))
                    output, sample_rate = librosa.load(audio_path, sr=44100, mono=True,
                                                       duration=30, offset=offset_int)
                    mels_spectrogram = librosa.feature.melspectrogram(output, sample_rate,
                                                                      n_fft=2048, hop_length=1024,
                                                                      n_mels=128)
                    x_db = librosa.power_to_db(mels_spectrogram, ref=np.max)
                    features = np.reshape(x_db, (num_cols*num_rows,))
                    to_append = ''

                    for feature in features:
                        to_append += f' {feature}'
                    genre_to_int = {'Classical': 0, 'Country': 1, 'Dance': 2, 'HipHop': 3, 'Jazz': 4,
                                    'Pop': 5, 'R&B': 6, 'Rock': 7}
                    to_append += f' {genre_to_int[genre]}'
                    file = open(csv_name, 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
                        counter = counter - 1
                    if counter == 0:
                        break
            print(f'{genre} has been completed')


if __name__ == '__main__':
    main()
