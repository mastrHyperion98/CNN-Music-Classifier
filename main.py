import librosa
from sklearn.preprocessing import LabelEncoder

from tools import FeatureExtractor
import numpy as np
import pandas as pd


def LoadData(filename):
    # Read our extracted data into a pandas dataframe
    dataframe = pd.read_csv(filename)
    # Next up we want to use sklearn preprocessing Label Encoder to turn our labels into numerics
    genres = dataframe.iloc[:, -1]
    # we will create an encoder
    labelEncoder = LabelEncoder()
    # create our y dataset
    y = labelEncoder.fit_transform(genres)
    # Extract and preprocess our features from the dataframe
    X = FeatureExtractor.PreprocessData(dataframe)

    return X, y


def main():
   X, y = LoadData('data.csv')


if __name__ == '__main__':
    main()


