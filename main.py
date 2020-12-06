import csv
import math

import librosa
import sklearn.neural_network

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1, MeanSquaredError, MeanSquaredLogError
from pytorch_lightning.metrics.functional.classification import confusion_matrix, auroc, precision_recall_curve, \
    stat_scores_multiple_classes
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tools import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename, labelEncoder):
    # Read our extracted data into a pandas dataframe
    dataframe = pd.read_csv(filename)
    # Next up we want to use sklearn preprocessing Label Encoder to turn our labels into numerics
    genres = dataframe.iloc[:, -1]
    # create our y dataset
    y = labelEncoder.fit_transform(genres)
    # Extract and preprocess our features from the dataframe
    X = FeatureExtractor.PreprocessData(dataframe)

    return X, y


# Plot the epoch vs loss
def plot_loss(X, y, title, outputFile):
    # We will print stuff here
    plt.plot(X, y)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(outputFile)
    plt.show()


# Function to get and plot the confusion matrix
def plot_confusion_matrix(pred, target, label, title, outputFile):
    matrix = confusion_matrix(pred, target, normalize=True)
    plt.figure(figsize=(150,150))
    im = plt.matshow(matrix, cmap='gist_gray')
    plt.title(title)
    plt.xticks(np.arange(0,8,1), labels=label, rotation=90)
    plt.yticks(np.arange(0,8,1), labels=label)
    for (i, j), z in np.ndenumerate(matrix):
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    plt.colorbar()
    plt.savefig(outputFile)
    plt.show()


def compute_precision(tp, fp):
    # prediction = tp / (tp + fp)
    return tp / (tp+fp)


def compute_recall(tp,fn):
    # recall = tp / (tp + fn)
    return tp / (tp+fn)


def output_analytics(predictions, target, label, filename):
    # Define the objects we will need

    header = 'Accuracy'
    for i in range(8):
        header += f' Precision_{label[i]}'

    for i in range(8):
        header += f' Recall_{label[i]}'

    header += ' means_squared_error'

    header = header.split()
    file = open(f'{filename}.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # Get the indices of the best predictions
    predict = []
    for prediction in predictions:
        values, indices = prediction.max(0)
        predict.append(indices)

    #turn it into a torch
    predict = torch.from_numpy(np.array(predict))

    plot_confusion_matrix(predict, target, label,
                          "CNN Confusion Matrix (" + f'{predict.shape[0]} test samples)', filename+'_confusionMatrix.png')
    plot_precision_recall_curve(predict, target, "Recall vs Precision", filename+"_RP_Graph.png")

    acc = Accuracy()
    mse = MeanSquaredError()
    accuracy = acc(predict, target)
    tps, fps, tns, fns, sups = stat_scores_multiple_classes(predict, target, num_classes=8)
    precision = compute_precision(tps,fps)
    recall = compute_recall(tps, fns)
    means_squared_error = mse(predict, target)

    # save in a list and write to a csv file
    list = f'{accuracy}'

    for prec in precision:
        list += f' {prec}'

    for rec in recall:
        list += f' {rec}'

    list += f' {means_squared_error}'
    file = open(f'{filename}.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(list.split())
    file.close()


def plot_precision_recall_curve(pred, target, title, outputFile):
    precision, recall, thresholds = precision_recall_curve(pred, target)
    plt.plot(recall,precision)
    plt.title(title)
    plt.xlabel("Recall (R)")
    plt.ylabel("Precision (P)")
    plt.savefig(outputFile)
    plt.show()


def experiment_one(X, y, labels):
    #Experiment One and two uses our extrapolated features to try and create a fast and efficient trainer.
    #Here we will be using a RandomForestClassifier of max_depth 20 and 200 estimators to estimate our resuts.
    print("#########\tExperiment One: RandomForest Classifier\t#########")
    max_depth = 26
    n_estimators = 100
    # load our data
    train_x, val_x, train_y, val_y=  train_test_split(X,y, test_size=0.1)
    # Create a RandomForestClassifier
    clf = RandomForestClassifier(max_depth=max_depth, bootstrap=True, n_estimators=n_estimators, random_state=0)
    # Fit the data
    clf.fit(train_x, train_y)
    # Display accuracy of the model
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
    plt.savefig('DecisionTree_ConfusionMatrix.png')
    plt.show()


def experiment_two(X, y, labels):
    #Experiment two is very similar to Experiment One but using a Neural Network instead
    print("#########\tExperiment Two: Neural Network Classifier\t#########")
    # Your code here. Aim for 2-4 lines.
    #X, y = LoadData('data.csv')
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_train_torch = torch.from_numpy(train_x).float()
    y_train_torch = torch.from_numpy(train_y).long()
    X_test_torch = torch.from_numpy(val_x).float()
    y_test_torch = torch.from_numpy(val_y).long()

    torch.manual_seed(0) # Ensure model weights initialized with same random numbers

    # Create an object that holds a sequence of layers and activation functions
    model = torch.nn.Sequential(
        torch.nn.Linear(26, 20),   # Applies Wx+b from 26 dimensions down to 20
        torch.nn.ReLU(),
        torch.nn.Linear(20, 14),  # Applies Wx+b from 20 dimensions down to 14
        torch.nn.ReLU(),
        torch.nn.Linear(14, 8),  # Applies Wx+b from 14 dimensions down to 8
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

    # Set model in evaluation mode
    model.eval()
    output_analytics(model(X_train_torch), y_train_torch, labels, 'FNN_Train_Analytics')
    output_analytics(model(X_test_torch), y_test_torch, labels, 'FNN_Test_Analytics')


def experiment_three():
    #Using CNN and spectrogram images directly for classification
    labelEncoder = LabelEncoder()
    X, y = load_data('dataGraphFull.csv',labelEncoder)
    labels = labelEncoder.inverse_transform(np.arange(0, 8, 1))
    # Split into training and testing data
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2)
    del X
    del y
    # For the dataGraph there is a single channel'
    TRACK_LENGTH = 15
    N = 128

    M = round(646/15 * TRACK_LENGTH)
    REDUCTION_KERNEL_SIZE = round((42/646 * M)/3.5)
    train_x = train_x.reshape(train_x.shape[0], N, M)
    val_x = val_x.reshape(val_x.shape[0], N, M)
    train_x_torch = torch.from_numpy(train_x).float()
    train_x_torch = train_x_torch[:, None, :, :]
    test_x_torch = torch.from_numpy(val_x).float()[:, None, :, :]

    # Transform/Reshape our targets
    train_y_torch = torch.from_numpy(train_y).long()
    test_y_torch = torch.from_numpy(val_y).long()


    torch.manual_seed(0)  # Ensure model weights initialized with same random numbers

    # Create an object that holds a sequence of layers and activation functions
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(kernel_size=(2, 4)),
        torch.nn.Dropout2d(p=0.2),

        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(kernel_size=(2, 4)),

        torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(kernel_size=(2, 4)),
        torch.nn.Dropout2d(p=0.2),

        torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(18, REDUCTION_KERNEL_SIZE), stride=1,padding=1),
        torch.nn.ReLU(),

        torch.nn.Conv2d(in_channels=256, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
    )

    # Create an object that can compute "negative log likelihood of a softmax"
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Make 10 passes over the training data, each time using batch_size samples to compute gradient
    num_epoch = 50
    batch_size = 80 # batch size 25
    model.train()

    loss_list = []
    epoch_list = np.arange(1,num_epoch+1,1, dtype='int64')
    for epoch in range(num_epoch):
        for i in range(0, len(train_x), batch_size):
            X = train_x_torch[i:i + batch_size]  # Slice out a mini-batch of features
            y = train_y_torch[i:i + batch_size]  # Slice out a mini-batch of targets\
            y_pred = model(X)  # Make predictions (final-layer activations)
            l = loss(y_pred, y)  # Compute loss with respect to predictions

            model.zero_grad()  # Reset all gradient accumulators to zero (PyTorch thing)
            l.backward()  # Compute gradient of loss wrt all parameters (backprop!)
            optimizer.step()  # Use the gradients to take a step with SGD.
        # Add loss to the list
        loss_list.append(l.item())
        print("Epoch %d final minibatch had loss %.4f" % (epoch + 1, l.item()))

    model.eval()

    plot_loss(epoch_list, loss_list, "CNN of Spectrogram Loss vs Epoch ("+f'{train_x.shape[0]} samples)','LossPltCNN.png')
    output_analytics(model(test_x_torch), test_y_torch, labels, "CNN_Test_Analytics")   # Testing Analytics
    #output_analytics(model(train_x_torch), train_y_torch, labels, "CNN_Train_Analytics")    #Training Analytics


def main():
  experiment_three()


if __name__ == '__main__':
    main()


