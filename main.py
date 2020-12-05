import librosa
import sklearn.neural_network

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from pytorch_lightning.metrics import Accuracy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tools import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
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
        plt.figure(**kwargs)
        plt.imshow(channel.detach(), vmin=-vmax, vmax=vmax, cmap='gray')
        plt.colorbar()
        plt.title("channel %d" % i)
        plt.show()


def test_torch(X_test, y_test, model):

    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    print('X_test size', str(X_test.size()))
    print('y_test size', str(y_test.size()))
    print('1dim shape', str(X_test.shape[0]))

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            input_data = X_test[i, :]
            print('X_test cut size', str(X_test[i, :].size()))
            label = y_test[i]
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == label).sum().item()

    return 100 * correct / total


def ExperimentOne():
    # Experiment One and two uses our extrapolated features to try and create a fast and efficient trainer.
    # Here we will be using a RandomForestClassifier of max_depth 20 and 200 estimators to estimate our resuts.
    print("#########\tExperiment One: RandomForest Classifier\t#########")
    max_depth = 30
    n_estimators = 100
    # load our data
    X, y, genres = LoadData('data.csv')
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1)
    clf = RandomForestClassifier(max_depth=max_depth, bootstrap=True,
                                 n_estimators=n_estimators, random_state=0)
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
    X, y, genres = LoadData('data.csv')
    # Experiment two is very similar to Experiment One but using a Neural Network instead
    print("#########\tExperiment Two: Neural Network Classifier\t#########")

    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1, shuffle=True)
    x_train_torch = torch.from_numpy(train_x).float()
    y_train_torch = torch.from_numpy(train_y).long()
    x_test_torch = torch.from_numpy(val_x).float()
    y_test_torch = torch.from_numpy(val_y).long()

    torch.manual_seed(0)  # Ensure model weights initialized with same random numbers

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
    num_epoch = 200
    batch_size = 200
    model.train()
    for epoch in range(num_epoch):
        for i in range(0, len(train_x), batch_size):
            X = x_train_torch[i:i + batch_size]  # Slice out a mini-batch of features
            y = y_train_torch[i:i + batch_size]  # Slice out a mini-batch of targets

            y_pred = model(X)  # Make predictions (final-layer activations)
            l = loss(y_pred, y)  # Compute loss with respect to predictions

            model.zero_grad()  # Reset all gradient accumulators to zero (PyTorch thing)
            l.backward()  # Compute gradient of loss wrt all parameters (backprop!)
            optimizer.step()  # Use the gradients to take a step with SGD.

        print("Epoch %d final minibatch had loss %.4f" % (epoch + 1, l.item()))

    model.eval()
    accuracy = Accuracy()
    predictions = model(x_test_torch)
    y_predictions = []
    for prediction in predictions:
        values, indices = prediction.max(0)
        y_predictions.append(indices)

    y_predictions = torch.from_numpy(np.array(y_predictions))

    print("Testing Accuracy: " + str(accuracy(y_predictions, y_test_torch)))

    predictions = model(x_train_torch)
    y_predictions = []
    for prediction in predictions:
        values, indices = prediction.max(0)
        y_predictions.append(indices)

    y_predictions = torch.from_numpy(np.array(y_predictions))

    print("Training Accuracy: " + str(accuracy(y_predictions, y_train_torch)))


def test_model(X_test, y_test, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        size = X_test.shape[0]
        print('size: ', size)
        for i in range(size):
            image = X_test[i:i+1, :, :, :]
            label = y_test[i]
            output = model(image)
            print('label: ', label.shape, ', output:', output.shape)
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == label).sum().item()

    print('corr: ', correct, ', total:', total)
    return 100 * correct / total


def ExperimentThree():
    x_data, y_data = LoadData('dataGraph-10r.csv')
    # Experiment three Using CNN and spectrogram images directly for classification
    print("#########\tExperiment Three: Convolutional Neural Network Classifier\t#########")

    # Split into training and testing data
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.1)
    # For the dataGraph there is a single channel'
    N = 128
    M = 1292
    x_train_torch = train_x.reshape(train_x.shape[0], N, M)
    train_x_torch = torch.from_numpy(train_x).float()
    train_x_torch = train_x_torch[:, None, :, :]
    # TO-DO: Reshape torch to be 3D and match form (train_x.shape[0],N,M) where each index of N and M are the target values
    y_train_torch = torch.from_numpy(train_y).long()

    torch.manual_seed(0)  # Ensure model weights initialized with same random numbers

    print('train_x: ', x_train_torch.shape, ', train_y:', y_train_torch.shape)
    # Create an object that holds a sequence of layers and activation functions
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
        torch.nn.Linear(12*4*4, 120),
        torch.nn.Linear(120, 60),
        torch.nn.Linear(60, 8),
    )

    # Create an object that can compute "negative log likelihood of a softmax"
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Make 10 passes over the training data, each time using batch_size samples to compute gradient
    num_epoch = 2
    batch_size = 4  # batch size 25
    model.train()
    for epoch in range(num_epoch):
        for i in range(0, len(train_x), batch_size):
            x_sample = x_train_torch[i:i + batch_size]  # Slice out a mini-batch of features
            y_sample = y_train_torch[i:i + batch_size]  # Slice out a mini-batch of targets
            y_pred = model(x_sample)  # Make predictions (final-layer activations)
            l = loss(y_pred, y_sample)  # Compute loss with respect to predictions

            model.zero_grad()  # Reset all gradient accumulators to zero (PyTorch thing)
            l.backward()  # Compute gradient of loss wrt all parameters (backprop!)
            optimizer.step()  # Use the gradients to take a step with SGD.

        print("Epoch %d final minibatch had loss %.4f" % (epoch + 1, l.item()))

    x_test_torch = torch.from_numpy(test_x).float()
    y_test_torch = torch.from_numpy(test_y).long()
    model.eval()

    print("Testing Accuracy: " + str(test_model(x_test_torch.clone().detach().requires_grad_(False),
                                                y_test_torch.clone().detach().requires_grad_(False), model)))


def main():
    ExperimentThree()


if __name__ == '__main__':
    main()
