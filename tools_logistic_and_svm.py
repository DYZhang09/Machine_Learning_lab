from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def getData(csv_file_path, keep_dim=True, visualize=False):
    """
    preprocess data from csv file.\n
    :param csv_file_path: csv file path
    :param keep_dim: if True then label will be reshaped to (1, N),
                      else will be reshaped to 1-dimension array with length N
    :return: (train_data, train_label), (test_data, test_label)
    """
    if os.path.exists(csv_file_path) is False:
        print(csv_file_path + "Not found.")
        return None

    features_all = []
    labels_all = []

    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            features_all.append([float(feature) for feature in row[1:58]])  # get features
            labels_all.append(int(row[-1]))  # get labels

    data = np.array(features_all)  # transform list into numpy array
    label = np.array(labels_all)

    data_mean = np.mean(data, axis=0, keepdims=True)  # standardize data
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean) / data_std

    train_data = data[:3000]  # split data into two parts
    test_data = data[3000:]
    train_label = label[:3000]
    test_label = label[3000:]
    if keep_dim:
        train_label = train_label.reshape((1, -1))
        test_label = test_label.reshape((1, -1))
    if visualize:
        t_sne(data, label, 'visualization of dataset')

    return (train_data, train_label), (test_data, test_label)


def drawLossForDiffParams(model, train_data, train_label, test_data, test_label,
                          learning_rates, reg_strengths, epoch, optimize='sgd'):
    """
    try different combination of hyper-params to train model and
    draw loss for each combination.\n
    :param model: classifier model. must be a Python class with function member:
                    __init__(learning_rate, reg_strength, *args),
                    train(train_data, train_label, epoch, optimize, *args),
                    predict(test_data, *args),
                    calcAccuracy(test_label, *args)
    :param train_data: data for classifier training
    :param train_label: ground-truth label of train data
    :param test_data: data for prediction and test model
    :param test_label: ground-truth label of test data
    :param learning_rates: a list of learning rates
    :param reg_strengths: a list of regularization strengths
    :param epoch: number of training iteration
    :param optimize: optimize method, can be 'sgd','adagrad','rmsprop'. default 'sgd'.
    :return: best accuracy in hyper-params' combinations and
                a Python dictionary of hyper-params combination that performs best
    """
    best_accu = .0
    best_param = {'learning_rate': .0, 'regularization': .0}
    index = [i * 10 for i in range(int(epoch / 10))]

    for reg_strength in reg_strengths:
        for learning_rate in learning_rates:
            lr = model(learning_rate=learning_rate, reg_strength=reg_strength)
            history_train_loss = lr.train(train_data, train_label, epoch=epoch, optimize=optimize)
            lr.predict(test_data)
            accu = lr.calcAccuracy(test_label)
            # draw loss
            plt.plot(index, history_train_loss,
                     marker='o', label='train_loss ' + 'lr: ' + str(learning_rate) + ' reg: ' + str(reg_strength))
            # select best param combination
            if accu > best_accu:
                best_accu = accu
                best_param['learning_rate'] = learning_rate
                best_param['regularization'] = reg_strength
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train loss of different hyper-params')
    plt.grid()
    plt.show()
    return best_accu, best_param


def drawAccuOfBestForDiffEpoch(model, train_data, train_label, test_data, test_label,
                               best_param, epochs, optimize='sgd'):
    """
    try different epoch to train classifier with the best params and draw accuracy.\n
    :param model: classifier model. must be a Python class with function member:
                    __init__(learning_rate, reg_strength, *args),
                    train(train_data, train_label, epoch, optimize, *args),
                    predict(test_data, *args),
                    calcAccuracy(test_label, *args)
    :param train_data: data for classifier training
    :param train_label: ground-truth label of train data
    :param test_data: data for prediction and test model
    :param test_label: ground-truth label of test data
    :param best_param: a Python dictionary of best params
    :param epochs: a list of epochs
    :param optimize: optimize method, can be 'sgd','adagrad','rmsprop'. default 'sgd'.
    :return: a list of accuracies for each epoch
    """
    learning_rate = best_param['learning_rate']
    reg_strength = best_param['regularization']
    accuracies = []
    lr = model(learning_rate=learning_rate, reg_strength=reg_strength)
    for epoch in epochs:
        lr.train(train_data, train_label, epoch=epoch, optimize=optimize)
        lr.predict(test_data)
        accu = lr.calcAccuracy(test_label)
        accuracies.append(accu)
    plt.plot(epochs, accuracies, 'ro-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy of best classifier by different epoch')
    plt.grid()
    plt.show()
    return accuracies


def t_sne(x, labels, fig_title=None):
    tsne = manifold.TSNE(n_components=3, init='pca')
    x_tsne = tsne.fit_transform(x)
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['red', 'blue']
    for i in range(int(x.shape[0]*0.4)):
        ax.scatter(x_tsne[i, 0], x_tsne[i, 1],x_tsne[i, 2],
                   color=colors[labels[i]])
    if fig_title is not None:
        plt.title(fig_title)
    plt.show()

