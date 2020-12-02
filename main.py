import librosa
import sklearn.neural_network

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from pytorch_lightning.metrics import Accuracy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tools import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
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


def ExperimentOne():
    #Experiment One and two uses our extrapolated features to try and create a fast and efficient trainer.
    #Here we will be using a RandomForestClassifier of max_depth 20 and 200 estimators to estimate our resuts.
    print("#########\tExperiment One: RandomForest Classifier\t#########")
    max_depth = 26
    n_estimators = 100
    # load our data
    X, y, genres = LoadData('data.csv')
    train_x, val_x, train_y, val_y=  train_test_split(X,y, test_size=0.1)
    clf = RandomForestClassifier(max_depth=max_depth, bootstrap=True, n_estimators=n_estimators, random_state=0)
    clf.fit(train_x, train_y)
    print("Shape of training data: " + str(train_x.shape))
    print("Training Accuracy: " + str(clf.score(train_x, train_y)*100))
    print("Testing Accuracy: " + str(clf.score(val_x, val_y)*100))

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, val_x, val_y,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


def ExperimentTwo():
    #Experiment two is very similar to Experiment One but using a Neural Network instead
    print("#########\tExperiment Two: Neural Network Classifier\t#########")
    # Your code here. Aim for 2-4 lines.
    X, y = LoadData('data.csv')
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_train_torch = torch.from_numpy(train_x).float()
    y_train_torch = torch.from_numpy(train_y).long()
    X_test_torch = torch.from_numpy(val_x).float()
    y_test_torch = torch.from_numpy(val_y).long()

    torch.manual_seed(0) # Ensure model weights initialized with same random numbers

    # Create an object that holds a sequence of layers and activation functions
    model = torch.nn.Sequential(
        torch.nn.Linear(26, 20),   # Applies Wx+b from 784 dimensions down to 10
        torch.nn.ReLU(),
        torch.nn.Linear(20, 14),  # Applies Wx+b from 784 dimensions down to 10
        torch.nn.ReLU(),
        torch.nn.Linear(14, 8),  # Applies Wx+b from 784 dimensions down to 10
    )

    # Create an object that can compute "negative log likelihood of a softmax"
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Make 10 passes over the training data, each time using batch_size samples to compute gradient
    num_epoch = 1000
    batch_size = 200
    model.train()
    for epoch in range(num_epoch):
        for i in range(0, len(train_x), batch_size):
            X = X_train_torch[i:i + batch_size]  # Slice out a mini-batch of features
            y = y_train_torch[i:i + batch_size]  # Slice out a mini-batch of targets

            y_pred = model(X)  # Make predictions (final-layer activations)
            l = loss(y_pred, y)  # Compute loss with respect to predictions

            model.zero_grad()  # Reset all gradient accumulators to zero (PyTorch thing)
            l.backward()  # Compute gradient of loss wrt all parameters (backprop!)
            optimizer.step()  # Use the gradients to take a step with SGD.

        print("Epoch %d final minibatch had loss %.4f" % (epoch + 1, l.item()))

    model.eval()
    accuracy = Accuracy()
    predictions = model(X_test_torch)
    y_predictions = []
    for prediction in predictions:
        values, indices = prediction.max(0)
        y_predictions.append(indices)

    y_predictions = torch.from_numpy(np.array(y_predictions))

    print("Testing Accuracy: " + str(accuracy(y_predictions, y_test_torch)))

    predictions = model(X_train_torch)
    y_predictions = []
    for prediction in predictions:
        values, indices = prediction.max(0)
        y_predictions.append(indices)

    y_predictions = torch.from_numpy(np.array(y_predictions))

    print("Training Accuracy: " + str(accuracy(y_predictions, y_train_torch)))

def ExperimentThree():
    #Using CNN and spectrogram images directly for classification
    X, y = LoadData('dataGraph.csv')
    # Split into training and testing data
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1)
    # For the dataGraph there is a single channel'
    N = 128
    M = 431
    train_x = train_x.reshape(train_x.shape[0], N, M)
    train_x_torch = torch.from_numpy(train_x).float()
    train_x_torch = train_x_torch[:, None, :, :]
    #TO-DO: Reshape torch to be 3D and match form (train_x.shape[0],N,M) where each index of N and M are the target values
    train_y_torch = torch.from_numpy(train_y).long()[:, None, None]

    torch.manual_seed(0)  # Ensure model weights initialized with same random numbers

    # Create an object that holds a sequence of layers and activation functions
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=0)
    )

    # Create an object that can compute "negative log likelihood of a softmax"
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Make 10 passes over the training data, each time using batch_size samples to compute gradient
    num_epoch = 1000
    batch_size = 25 # batch size 25
    model.train()
    for epoch in range(num_epoch):
        for i in range(0, len(train_x), batch_size):
            X = train_x_torch[i:i + batch_size]  # Slice out a mini-batch of features
            y = train_y_torch[i:i + batch_size]  # Slice out a mini-batch of targets

            y_pred = model(X)  # Make predictions (final-layer activations)
            l = loss(y_pred, y)  # Compute loss with respect to predictions

            model.zero_grad()  # Reset all gradient accumulators to zero (PyTorch thing)
            l.backward()  # Compute gradient of loss wrt all parameters (backprop!)
            optimizer.step()  # Use the gradients to take a step with SGD.

        print("Epoch %d final minibatch had loss %.4f" % (epoch + 1, l.item()))


def main():
   ExperimentThree()



if __name__ == '__main__':
    main()


