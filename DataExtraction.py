# DataExtraction.py is the script used to read all our music files and extract the required information into a csv file
# The csv file will be fed into the CNN and it will be used to speed up testing with the neural network.
# Reading all the music files and extracting the data is expected to take a long time.
import csv
import os
import librosa
import numpy as np
from tools import FeatureExtractor
def main():
    FetchData()


def FetchData():
    dataset_dir = 'dataset'
    genres = os.listdir(path=dataset_dir)
    for genre in genres:
        genre_dir = dataset_dir+"/"+genre
        filenames = os.listdir(genre_dir)
        for filename in filenames:
            audio_path = genre_dir+'/'+filename
            x, sr = librosa.load(audio_path)
            features = FeatureExtractor.Extract(filename, x, sr, genre)
            to_append = ''
            for feature in features:
                to_append += f' {feature}'

            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())


if __name__ == '__main__':
    main()