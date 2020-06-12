from tools_logistic_and_svm import *
from kernel import *
import cvxopt.solvers


class KernelSVM(object):
    def __init__(self, kernel, C):
        """
        initialize kernel svm classifier.\n
        :param kernel: Python class of kernel.
                        the class must have a data member can be called as 'kernel.linear'
                        (if kernel is linear kernel then kernel.linear should be True else False)
                        and a function member can be called as 'kernel.calculate(xi, xj, *args)'
        :param C: relaxation variable
        """
        self.kernel = kernel
        self.C = C
        self.weight = []
        self.bias = 0
        self.a = None
        self.support_vec = []
        self.support_vec_label = []
        self.predict_label = None

    def __getKernelMatrix(self, x):
        """
        get kernel matrix of input.\n
        :param x: input
        :return: kernel matrix of x
        """
        N, _ = x.shape
        K = np.zeros((N, N))
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                K[i, j] = self.kernel.calculate(xi, xj)
        return K

    def __getLagrangeMulti(self, input, label, K):
        """
        get lagrange multiplier α.\n
        :param input: input data with shape (N, D)
        :param label: ground-truth label of input data with shape (N, )
        :param K: kernel matrix of input
        :return: lagrange multiplier α
        """
        N = input.shape[0]
        P = cvxopt.matrix(np.outer(label, label) * K)  # get P, q, A, b for cvxopt.solvers.qp
        q = cvxopt.matrix(np.ones(N) * -1)
        A = label.reshape((1, N))
        A = A.astype('float')
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(0.0)

        if self.C is None:  # hard margin
            G = cvxopt.matrix(np.diag(np.ones(N) * -1))
            h = cvxopt.matrix(np.zeros(N))
        else:  # soft margin
            temp1 = np.diag(np.ones(N) * -1)
            temp2 = np.identity(N)
            G = cvxopt.matrix(np.vstack((temp1, temp2)))
            temp1 = np.zeros(N)
            temp2 = np.ones(N) * self.C
            h = cvxopt.matrix(np.hstack((temp1, temp2)))

        result = cvxopt.solvers.qp(P, q, G, h, A, b)  # solve
        a = np.ravel(result['x'])
        return a

    def __getBias(self, a, support_vec_Label, support_vec, K, indexis):
        """
        get bias.\n
        :param a: lagrange multiplier α
        :param support_vec_Label: ground-truth label of support vector
        :param support_vec: support vector
        :param K: kernel matrix
        :param indexis: indexis of support vector
        :return: bias
        """
        b = 0
        for i in range(len(a)):
            b += support_vec_Label[i]
            b -= np.sum(a * support_vec_Label * K[indexis[i], support_vec])
        b /= len(a)
        return b

    def __getWeight(self, features, a, support_vec, support_vec_label):
        """
        get weight(if kernel is linear kernel).\n
        :param features: dimension of features
        :param a: lagrange multiplier α
        :param support_vec: support vector
        :param support_vec_label: ground-truth label of support vector
        :return: if kernel is linear kernel then return weight else return None
        """
        if self.kernel.linear is True:
            w = np.zeros(features)
            for i in range(len(a)):
                w += a[i] * support_vec_label[i] * support_vec[i]
        else:
            w = None
        return w

    def __transformLabel(self, label, back=False):
        """
        transform labels. label[label == 0] = -1 or label[label == -1] = 0.\n
        :param label: origin labels
        :param back: if back is True then label[label == -1] = 0. default False
        :return: transformed labels
        """
        _label = label.copy()
        if back is False:
            _label[_label == 0] = -1
        else:
            _label[_label == -1] = 0
        return _label

    def train(self, train_data, train_label):
        """
        train for kernel svm classifier.\n
        :param train_data: data for training with shape (N, D)
        :param train_label: ground-truth label of train data with shape (N, )
        :return: None
        """
        label = self.__transformLabel(train_label)

        N, D = train_data.shape
        K = self.__getKernelMatrix(train_data)
        a = self.__getLagrangeMulti(train_data, label, K)
        support_vec = a > 1e-5
        indexis = np.arange(len(a))[support_vec]
        self.a = a[support_vec]
        self.support_vec = train_data[support_vec]
        self.support_vec_label = label[support_vec]

        self.bias = self.__getBias(self.a, self.support_vec_label, support_vec, K, indexis)
        self.weight = self.__getWeight(D, self.a, self.support_vec, self.support_vec_label)

    def predict(self, test_data):
        """
        predict labels of data.\n
        :param test_data: data that needs to predict with shape(N, D)
        :return: predict labels
        """
        if self.weight is not None:
            label = np.sign(test_data @ self.weight + self.bias)
            self.predict_label = self.__transformLabel(label, True)
        else:
            predict_label = np.zeros(len(test_data))
            for i in range(len(test_data)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.support_vec_label, self.support_vec):
                    s += a * sv_y * self.kernel.calculate(test_data[i], sv)  # model of kernel svm f(x)
                predict_label[i] = s
            label = np.sign(predict_label + self.bias)
            self.predict_label = self.__transformLabel(label, True)
        return self.predict_label.copy()

    def calcAccuracy(self, test_label):
        """
        calculate accuracy of predict labels.\n
        :param test_label: ground-truth label of test data with shape (N, )
        :return: accuracy
        """
        diff = test_label - self.predict_label
        corr_num = diff[diff == 0].size
        return corr_num / test_label.shape[0]


(train_data, train_label), (test_data, test_label) = getData(r"H:\机器学习\结课实验\income.csv",
                                                             keep_dim=False)
sigmas = [1, 3, 5, 7, 9, 11, 13]
C_list = [1e-2, 1e-1, 1, 10, 100]
for C in C_list:
    accuracies = []
    for sigma in sigmas:
        svm = KernelSVM(GaussianKernel(sigma=sigma), C=C)
        svm.train(train_data, train_label)
        label = svm.predict(test_data)
        accuracies.append(svm.calcAccuracy(test_label))
        print(accuracies[-1])
    plt.plot(sigmas, accuracies, label='C: ' +str(C))
plt.title('Accuracy of different C and sigma of RBF kernel')
plt.legend(loc='best')
plt.xlabel('sigma')
plt.ylabel('accuracy')
plt.grid()
plt.show()
