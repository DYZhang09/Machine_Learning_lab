import numpy as np
import csv
import os
import matplotlib.pyplot as plt


class LogisticRegression(object):
    def __init__(self, dimension, learning_rate=1e-3, epoch=1, reg_strength=0.5):
        """
        init Logistic Regression classifier.\n
        :param dimension: the dimension of features
        :param learning_rate: learning rate , default 1e-3
        :param epoch: epoch for training, default 1
        :param reg_strength: regularization strength, default 0.5
        """
        self.theta = np.random.randn(1, dimension + 1)
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.reg_strength = reg_strength
        self.predict_result = None

    def __sigmoid(self, x, backward=False, dy=.0):
        """
        sigmoid function g(x) = 1 / (1 + exp(-X)).\n
        :param x: input
        :param backward: if calculating gradient then True, else False. default False
        :param dy: value of dL / d(g(x)), default 0.0
        :return: results of function with shape of x
        """
        y = 1 / (1 + np.exp(-x) + 1e-7)
        dx = dy * y * (1 - y)
        if backward is False:
            return y
        else:
            return dx

    def __crossEntropyLoss(self, x, y, regu_lambda):
        """
        calculate loss and gradient through cross entropy function.\n
        :param x: data with shape (N, D)
        :param y: ground-truth labels matrix of x with shape (N, 1)
        :param regu_lambda: coefficient of regularization loss
        :return: loss, d_theta(d(Loss) / d(theta))
        """
        N, D = x.shape
        theta = self.theta
        z = x @ theta.T
        sigmoid_z = self.__sigmoid(z)
        loss = -(y * np.log(sigmoid_z) + (1 - y) * np.log(1 - sigmoid_z))  # cross entropy
        loss = (1 / N) * (np.sum(loss)) + regu_lambda * np.sum(theta * theta)  # add regularization loss

        d_sigmoid_z = (1 - y) * (1 / (1 - sigmoid_z)) - y * (1 / sigmoid_z)  # calculate gradient
        d_z = self.__sigmoid(z, True, d_sigmoid_z)
        d_theta = (x.T @ d_z).T / N
        d_theta += 2 * regu_lambda * theta
        return loss, d_theta

    def train(self, x, y):
        """
        train for classifier.\n
        :param x: train data with shape (N, D)
        :param y: label of train data with shape (N, 1)
        :return: history loss(every 100 epochs)
        """
        history_loss = []
        train_x = np.insert(x, 57, values=1, axis=1)  # expend one more dimension to calculate "+b"
        for i in range(self.epoch):
            if (i + 1) % 100 == 0:
                print("Processing epoch: %d" % (i + 1))
            loss, d_theta = self.__crossEntropyLoss(train_x, y, self.reg_strength)
            if (i + 1) % 100 == 0 or i == self.epoch - 1:
                print("Loss: %.8f" % loss)
                history_loss.append(loss)
            self.theta -= self.learning_rate * d_theta  # update theta
        return history_loss

    def predict(self, test_data):
        """
        predict label of test data.\n
        :param test_data: test data with shape (N, D)
        :return: predict result with shape (N, 1)
        """
        test_x = np.insert(test_data, 57, values=1, axis=1)
        self.predict_result = self.__sigmoid(test_x @ self.theta.T)
        self.predict_result[self.predict_result < 0.5] = 0
        self.predict_result[self.predict_result >= 0.5] = 1
        return self.predict_result

    def calcAccuracy(self, test_label):
        """
        calculate accuracy on test data.\n
        :param test_label: ground-truth label of test data with shape (N, 1)
        :return: accuracy
        """
        diff = self.predict_result - test_label
        correct = diff[diff == 0.]
        return correct.size / test_label.shape[0]


def getData(csv_file_path):
    """
    get data from csv file.\n
    :param csv_file_path: csv file path
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
            labels_all.append([float(label) for label in row[-1:]])  # get labels
    data = np.array(features_all)  # transform list into numpy array
    label = np.array(labels_all)
    data_mean = np.mean(data, axis=0, keepdims=True)  # standardize data
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean) / data_std
    train_data = data[:3000]  # split data into two parts
    test_data = data[3000:]
    train_label = label[:3000]
    test_label = label[3000:]
    return (train_data, train_label), (test_data, test_label)


(train_data, train_label), (test_data, test_label) = getData(r"H:\机器学习\结课实验\income.csv")
epoch = 5000
lr = LogisticRegression(dimension=57, epoch=epoch)
history_loss = lr.train(train_data, train_label)
lr.predict(test_data)
acc = lr.calcAccuracy(test_label)
print("Accuracy: %.5f" % acc)

index = [(i + 1) * 100 for i in range(int(epoch / 100))]
plt.plot(index, history_loss)
plt.show()
