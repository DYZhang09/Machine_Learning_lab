import numpy as np
import csv
import os
import matplotlib.pyplot as plt


class LogisticRegression(object):
    def __init__(self, learning_rate=1e-3, reg_strength=0.5):
        """
        init Logistic Regression classifier.\n
        :param learning_rate: learning rate , default 1e-3
        :param reg_strength: regularization strength, default 0.5
        """
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.weight = None
        self.beta = None
        self.ada_h_w = None
        self.ada_h_b = None
        self.predict_result = None

    def __initParams(self, dimension):
        """
        initialize params of logistic regression classifier.\n
        :param dimension: the dimension of feature
        :return: None
        """
        self.weight = np.random.randn(dimension, 1)
        self.beta = np.random.randn()
        self.ada_h_w = np.zeros_like(self.weight)
        self.ada_h_b = 0

    def __hypothesis(self, x, backward=False, d_h=None):
        """
        the hypothesis of logistic regression is h(x) = W.T @ x + B. if backward is False, this function
        will calculate h(x), otherwise will calculate dh(x) / dW and dh(x) / dB.\n
        :param x: input with shape(D, N)
        :param backward: if this function is used for back propagation then True. default False
        :param d_h: dLoss / dh(x), default None
        :return: if backward is False, this will return: h(x), else will return: dh(x) / dW ,dh(x) / dB.
        """
        h = self.weight.T @ x + self.beta
        if backward is False:
            return h
        else:
            if d_h is None:  # calculate gradient
                d_h = np.zeros_like(h)
            d_weight = (d_h @ x.T).T
            d_weight += 2 * self.reg_strength * self.weight
            d_beta = np.sum(d_h) / x.shape[1]
            return d_weight, d_beta

    def __sigmoid(self, x, backward=False, dy=.0):
        """
        sigmoid function g(x) = 1 / (1 + exp(-X)).\n
        :param x: input
        :param backward: if calculating gradient then True, else False. default False
        :param dy: value of dLoss / d(g(x)), default 0.0
        :return: if backward is False this will return: g(x), else will return: dLoss / dg(x).
        """
        y = 1 / (1 + np.exp(-x) + 1e-7)
        dx = dy * y * (1 - y)  # calculate gradient
        if backward is False:
            return y
        else:
            return dx

    def __crossEntropyLoss(self, pred, y):
        """
        calculate loss and gradient through cross entropy function.\n
        :param pred: the result of prediction with shape (1, N)
        :param y: ground-truth labels matrix of x with shape (1, N)
        :return: loss, dLoss / dpred
        """
        _, N = pred.shape
        reg_lambda = self.reg_strength
        loss = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))  # cross entropy
        loss = (1 / N) * (np.sum(loss)) + reg_lambda * np.sum(self.weight * self.weight)  # add regularization loss
        d_pred = (1 / N) * ((1 - y) * (1. / (1 - pred)) - y * (1. / pred))  # calculate gradient
        return loss, d_pred

    def __propagation(self, x, y):
        """
        forward propagation and backward propagation.\n
        :param x: train data with shape (D, N)
        :param y: ground-truth label of train data with shape (1, N)
        :return: (prediction, loss), (dLoss / dW, dLoss / dB)
        """
        h = self.__hypothesis(x)
        pred = self.__sigmoid(h)  # predict
        loss, d_pred = self.__crossEntropyLoss(pred, y)
        d_h = self.__sigmoid(h, backward=True, dy=d_pred)  # calculate gradient
        d_weight, d_beta = self.__hypothesis(x, backward=True, d_h=d_h)
        return (pred, loss), (d_weight, d_beta)

    def __updateParams(self, d_weight, d_beta, optimize='sgd'):
        """
        update weights and beta of classifier.\n
        :param d_weight: the gradient of weight
        :param d_beta: the gradient of beta
        :param optimize: optimize method
        :return: None
        """
        if optimize is 'sgd':
            self.weight -= self.learning_rate * d_weight
            self.beta -= self.learning_rate * d_beta
        else:
            if optimize is 'adagrad':
                self.ada_h_w += d_weight * d_weight
                self.ada_h_b += d_beta * d_beta
            else:
                self.ada_h_w = 0.9 * self.ada_h_w + 0.1 * d_weight * d_weight
                self.ada_h_b = 0.9 * self.ada_h_b + 0.1 * d_beta * d_beta
            self.weight -= self.learning_rate * d_weight / (np.sqrt(self.ada_h_w) + 1e-7)
            self.beta -= self.learning_rate * d_beta / (np.sqrt(self.ada_h_b) + 1e-7)

    def train(self, x, y, epoch=1, optimize='sgd'):
        """
        train for classifier.\n
        :param x: train data with shape (N, D)
        :param y: label of train data with shape (1, N)
        :param epoch: the number of iterations of training
        :param optimize: optimize method, can be 'sgd', 'adagrad' and 'rmsprop'. default 'sgd'.
        :return: history loss(every 5 epochs), history_accu(every 5 epochs)
        """
        history_loss = []
        history_accu = []
        self.__initParams(x.shape[1])  # initialize params
        train_x = x.T.copy()

        for i in range(epoch):
            if (i + 1) % 100 == 0:
                print("Processing epoch: %d" % (i + 1))
            (pred, loss), (d_weight, d_beta) = self.__propagation(train_x, y)  # calculate loss and gradients
            self.__updateParams(d_weight, d_beta, optimize)  # update params
            self.predict(x)  # calculate accuracy on train data
            train_accu = self.calcAccuracy(y)
            if i % 5 == 0:
                history_loss.append(loss)
                history_accu.append(train_accu)
        return history_loss, history_accu

    def predict(self, test_data):
        """
        predict label of test data.\n
        :param test_data: test data with shape (N, D)
        :return: predict result with shape (1, N)
        """
        test_x = test_data.T.copy()
        self.predict_result = self.__sigmoid(self.weight.T @ test_x + self.beta)
        self.predict_result[self.predict_result < 0.5] = 0
        self.predict_result[self.predict_result >= 0.5] = 1
        return self.predict_result.copy()

    def calcAccuracy(self, test_label):
        """
        calculate accuracy on test data.\n
        :param test_label: ground-truth label of test data with shape (1, N)
        :return: accuracy
        """
        diff = self.predict_result - test_label
        correct = diff[diff == 0.]
        return correct.size / test_label.shape[1]


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
    labels_all = [[]]
    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            features_all.append([float(feature) for feature in row[1:58]])  # get features
            labels_all[0].append(float(row[-1]))  # get labels
    data = np.array(features_all)  # transform list into numpy array
    label = np.array(labels_all)
    data_mean = np.mean(data, axis=0, keepdims=True)  # standardize data
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean) / data_std
    train_data = data[:3000]  # split data into two parts
    test_data = data[3000:]
    train_label = label[:, :3000]
    test_label = label[:, 3000:]
    return (train_data, train_label), (test_data, test_label)


(train_data, train_label), (test_data, test_label) = getData(r"H:\机器学习\结课实验\income.csv")
epoch = 500
learning_rates = [5e-1, 1e-1, 5e-2, 1e-2]
reg_strengths = [1e-1, 1e-2, 1e-3, 1e-4]
history_losses = []
history_accuracies = []
index = [i * 5 for i in range(int(epoch / 5))]
best_accu = .0
best_param = {'learning_rate': .0, 'regularization': .0}

for learning_rate in learning_rates:
    for reg_strength in reg_strengths:
        lr = LogisticRegression(learning_rate=learning_rate, reg_strength=reg_strength)
        history_loss, history_accu = lr.train(train_data, train_label, epoch=epoch, optimize='sgd')
        history_losses.append(history_loss)
        history_accuracies.append(history_accu)
        plt.plot(index, history_losses[-1], label='lr=' + str(learning_rate) + ' reg=' + str(reg_strength))
        lr.predict(test_data)
        acc = lr.calcAccuracy(test_label)
        print("Accuracy: %.5f" % acc)
        if acc > best_accu:
            best_accu = acc
            best_param['learning_rate'] = learning_rate
            best_param['regularization'] = reg_strength
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc="best")
plt.show()
print("best_accu: ", best_accu)
print(best_param)

for i in range(len(learning_rates)):
    for j in range(len(reg_strengths)):
        plt.plot(index, history_accuracies[i * len(reg_strengths) + j],
                 label='lr=' + str(learning_rates[i]) + ' reg=' + str(reg_strengths[j]))
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.show()
