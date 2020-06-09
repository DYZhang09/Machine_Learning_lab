import numpy as np
from tools_logistic_and_svm \
    import getData, drawLossForDiffParams, drawAccuOfBestForDiffEpoch


class LinearSVM(object):
    def __init__(self, class_num=2, delta=1, learning_rate=1e-3, reg_strength=1e-2):
        """
        init the SVM classifier.\n
        :param class_num: number of classes
        :param delta: delta of SVM margin
        :param learning_rate: learning rate for training. default 1e-3
        :param reg_strength: regularization strength for training. default 1e-2.
        """
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.delta = delta
        self.class_num = class_num
        self.weight = None
        self.beta = None
        self.ada_h_w = None
        self.ada_h_b = None
        self.predict_label = None

    def __initParams(self, dimension):
        """
        initialize params of classifier.\n
        :param dimension: dimension of features
        :return: None
        """
        self.weight = np.random.randn(dimension, self.class_num)
        self.beta = np.random.randn(1, self.class_num)
        self.ada_h_w = np.zeros_like(self.weight)
        self.ada_h_b = np.zeros_like(self.beta)

    def __hypothesis(self, x, backward=False, d_h=None):
        """
        the hypothesis of linear SVM is h(x) = x @ W + B. if backward is False, this function
        will calculate h(x), otherwise will calculate dh(x) / dW and dh(x) / dB.\n
        :param x: input with shape(N, D)
        :param backward: if this function is used for back propagation then True. default False
        :param d_h: dLoss / dh(x), default None
        :return: if backward is False, this will return: h(x), else will return: dh(x) / dW ,dh(x) / dB.
        """
        h = x @ self.weight + self.beta
        if backward is False:
            return h
        else:
            if d_h is None:  # calculate gradient
                d_h = np.zeros_like(h)
            d_weight = x.T @ d_h
            d_weight += 2 * self.reg_strength * self.weight
            d_beta = np.sum(d_h) / x.shape[1]
            return d_weight, d_beta

    def __hingeLoss(self, scores, y):
        """
        calculate loss and gradient through hinge loss function.\n
        :param scores: scores from hypothesis
        :param y: ground-truth label of train data with shape (N, )
        :return: loss, dLoss / dscores
        """
        N = scores.shape[0]

        correct_label_scores = scores[np.arange(N), y].reshape((-1, 1))
        margin = np.clip(scores - correct_label_scores + self.delta, 0, None)
        margin[np.arange(N), y] = 0
        loss = np.sum(margin) / N
        loss += self.reg_strength * np.sum(self.weight * self.weight)

        d_scores = np.zeros_like(scores)
        d_scores[margin != 0] = 1
        d_scores[np.arange(N), y] = -np.sum(d_scores, axis=1)
        return loss, d_scores

    def __propagation(self, x, y):
        """
        forward propagation and backward propagation.\n
        :param x: train data with shape (N, D)
        :param y: ground-truth label of train data with shape (N, )
        :return: (scores, loss), (dLoss / dW, dLoss / dB)
        """
        scores = self.__hypothesis(x)
        loss, d_scores = self.__hingeLoss(scores, y)
        d_weight, d_beta = self.__hypothesis(x, True, d_scores)
        return (scores, loss), (d_weight, d_beta)

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
        train for SVM classifier.\n
        :param train_data: data for training with shape (N, D)
        :param train_label: ground-truth label of train data with shape (N,)
        :param epoch: number of train iteration
        :param optimize: optimize method, can be 'sgd','adagrad' or 'rmsprop'. default 'sgd'
        :return: history train loss
        """
        history_train_loss = []
        self.__initParams(train_data.shape[1])

        for i in range(epoch):
            if (i + 1) % 100 == 0:
                print("Processing epoch: %d" % (i + 1))
            (pred, loss), (d_weight, d_beta) = self.__propagation(train_data,
                                                                  train_label)  # calculate loss and gradients
            self.__updateParams(d_weight, d_beta, optimize)  # update params
            if i % 10 == 0:
                history_train_loss.append(loss)

        return history_train_loss

    def predict(self, test_data):
        """
        use hypothesis of classifier to predict label.\n
        :param test_data: data to predict with shape (N, D)
        :return: predict labels with shape (N, )
        """
        scores = test_data @ self.weight + self.beta
        self.predict_label = np.argmax(scores, axis=1)
        return self.predict_label.copy()

    def calcAccuracy(self, test_label):
        """
        calculate accuracy.\n
        :param test_label: ground-truth label of test data with shape (N,)
        :return: accuracy
        """
        diff = self.predict_label - test_label
        correct_num = diff[diff == 0.].size
        return correct_num / test_label.shape[0]


learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
reg_strengths1 = [1e-1, 1e-2]
reg_strengths2 = [1e-3, 1e-4]
epoch = 300
(train_data, train_label), (test_data, test_label) = getData(r"H:\机器学习\结课实验\income.csv",
                                                             keep_dim=False)
# draw loss for different hyper-params
optimize = 'rmsprop'
best_accu_1, best_param_1 = drawLossForDiffParams(LinearSVM,
                                                  train_data, train_label, test_data, test_label,
                                                  learning_rates, reg_strengths1, epoch, optimize=optimize)
best_accu_2, best_param_2 = drawLossForDiffParams(LinearSVM,
                                                  train_data, train_label, test_data, test_label,
                                                  learning_rates, reg_strengths2, epoch, optimize=optimize)
best_accu = best_accu_1 if best_accu_1 > best_accu_2 else best_accu_2
best_param = best_param_1 if best_accu_1 > best_accu_2 else best_param_2
print(best_accu)
print(best_param)

# draw accuracy of best classifier for different epoch
epochs = [30, 50, 100, 150, 300, 500, 750, 1000, 1500]
accuracies = drawAccuOfBestForDiffEpoch(LinearSVM,
                                        train_data, train_label, test_data, test_label,
                                        best_param, epochs, optimize=optimize)
print(accuracies)
