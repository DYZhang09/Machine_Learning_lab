import numpy as np
from tools_logistic_and_svm \
    import getData, drawAccuOfBestForDiffEpoch, drawLossForDiffParams


class LogisticRegression(object):
    def __init__(self, learning_rate=1e-3, reg_strength=1e-2):
        """
        init Logistic Regression classifier.\n
        :param learning_rate: learning rate , default 1e-3
        :param reg_strength: regularization strength, default 1e-2
        """
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.weight = None
        self.beta = None
        self.ada_h_w = None
        self.ada_h_b = None
        self.predict_label = None

    def __initParams(self, dimension):
        """
        initialize params of logistic regression classifier.\n
        :param dimension: the dimension of feature
        :return: None
        """
        self.weight = np.random.randn(dimension, 1)
        self.beta = 0
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

    def train(self, train_data, train_label, epoch=1, optimize='sgd'):
        """
        train for classifier.\n
        :param train_data: data for classifier training with shape (N, D)
        :param train_label: ground-truth label of train data with shape (1, N)
        :param epoch: the number of iterations of training
        :param optimize: optimize method, can be 'sgd', 'adagrad' and 'rmsprop'. default 'sgd'.
        :return: history train loss(every 10 epochs)
        """
        history_train_loss = []
        self.__initParams(train_data.shape[1])  # initialize params
        train_x = train_data.T.copy()

        for i in range(epoch):
            if (i + 1) % 100 == 0:
                print("Processing epoch: %d" % (i + 1))
            (pred, loss), (d_weight, d_beta) = self.__propagation(train_x,
                                                                  train_label)  # calculate loss and gradients
            self.__updateParams(d_weight, d_beta, optimize)  # update params
            if i % 10 == 0:
                history_train_loss.append(loss)
        return history_train_loss

    def predict(self, test_data, return_label=True):
        """
        use the hypothesis of classifier to predict.\n
        :param test_data: data to predict with shape (N, D)
        :param return_label: if True this function will return label(0 or 1) of each sample
                                else return probability(between 0 and 1)) of each sample
        :return: labels if return_label is True else probabilities
        """
        test_x = test_data.T.copy()
        pred = self.__sigmoid(self.weight.T @ test_x + self.beta)
        pred_label = pred.copy()
        pred_label[pred_label < 0.5] = 0
        pred_label[pred_label >= 0.5] = 1
        self.predict_label = pred_label
        if return_label:
            return pred_label
        else:
            return pred

    def calcAccuracy(self, test_label):
        """
        calculate accuracy .\n
        :param test_label: ground-truth label of test data with shape (1, N)
        :return: accuracy
        """
        label = test_label
        diff = self.predict_label - label
        correct = diff[diff == 0.]
        return correct.size / label.shape[1]


learning_rates = [5e-1, 1e-1, 5e-2, 1e-2]
reg_strengths1 = [1e-1, 1e-2]
reg_strengths2 = [1e-3, 1e-4]
epoch = 300
(train_data, train_label), (test_data, test_label) = getData(r"H:\机器学习\结课实验\income.csv")

# draw loss for different hyper-params
optimize = 'rmsprop'
best_accu_1, best_param_1 = drawLossForDiffParams(LogisticRegression,
                                                  train_data, train_label, test_data, test_label,
                                                  learning_rates, reg_strengths1, epoch, optimize=optimize)
best_accu_2, best_param_2 = drawLossForDiffParams(LogisticRegression,
                                                  train_data, train_label, test_data, test_label,
                                                  learning_rates, reg_strengths2, epoch, optimize=optimize)
best_accu = best_accu_1 if best_accu_1 > best_accu_2 else best_accu_2
best_param = best_param_1 if best_accu_1 > best_accu_2 else best_param_2
print(best_accu)
print(best_param)

best_param['learning_rate'] = 1e-2

# draw accuracy of best classifier for different epoch
epochs = [30, 50, 100, 150, 300, 500, 750, 1000, 1500]
accuracies = drawAccuOfBestForDiffEpoch(LogisticRegression,
                                        train_data, train_label, test_data, test_label,
                                        best_param, epochs, optimize=optimize)
print(accuracies)
