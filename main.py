import librosa
import sklearn.neural_network

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tools import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_channels(x, **kwargs):
    """Plot each channel in z as an image, where z is in (C,H,W) format."""
    vmax = max(-x.min(), x.max())  # Range of values
    for i, channel in enumerate(x):
        plt.figure(**kwargs);
        plt.imshow(channel.detach(), vmin=-vmax, vmax=vmax, cmap='gray');
        plt.colorbar();
        plt.title("channel %d" % i)
        plt.show()


def main():
    sr = 22050
    X, y = LoadData('dataGraph.csv')
    #Split into training and testing data
    train_x, val_x, train_y, val_y=  train_test_split(X,y, test_size=0.1)
    #clf = RandomForestClassifier(max_depth=10, random_state=0)
    #clf.fit(train_x, train_y)
    #print(clf.score(val_x, val_y))
    # converting training images into torch format

    # For the dataGraph there is a single channel'
    img = train_x
    print(img.shape)
    img = img.reshape(72, 128, 431)
    print(img.shape)
    tensor = torch.from_numpy(img).float()
    tensor = tensor[:, None, None, :, :]
    print(tensor)

    for i in range(3):
        dimg = tensor[i]
        torch.manual_seed(0);  # Ensure PyTorch uses same random initial weights
        conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=0)
        a = conv(dimg)

        plot_channels(a[0])



if __name__ == '__main__':
    main()


