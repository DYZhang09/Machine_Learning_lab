from tools_logistic_and_svm import *
from kernel import *
import cvxopt.solvers


class KernelSVM(object):
    def __init__(self, kernel, C):
        self.kernel = kernel
        self.C = C
        self.weight = []
        self.bias = 0
        self.a = None
        self.support_vec = []
        self.support_vec_label = []
        self.predict_label = None

    def __getKernelMatrix(self, x):
        N, _ = x.shape
        K = np.zeros((N, N))
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                K[i, j] = self.kernel.calculate(xi, xj)
        return K

    def __getLagrangeMuil(self, input, label, C, K):
        N = input.shape[0]
        P = cvxopt.matrix(np.outer(label, label) * K)
        q = cvxopt.matrix(np.ones(N) * -1)
        A = label.reshape((1, N))
        A = A.astype('float')
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(0.0)

        if C is None:
            G = cvxopt.matrix(np.diag(np.ones(N) * -1))
            h = cvxopt.matrix(np.zeros(N))
        else:
            temp1 = np.diag(np.ones(N) * -1)
            temp2 = np.identity(N)
            G = cvxopt.matrix(np.vstack((temp1, temp2)))
            temp1 = np.zeros(N)
            temp2 = np.ones(N) * self.C
            h = cvxopt.matrix(np.hstack((temp1, temp2)))

        result = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(result['x'])
        return a

    def __getBias(self, a, supprot_vec_Label, support_vec, K, indexis):
        b = 0
        for i in range(len(a)):
            b += supprot_vec_Label[i]
            b -= np.sum(a * supprot_vec_Label * K[indexis[i], support_vec])
        b /= len(a)
        return b

    def __getWeight(self, features, a, support_vec, support_vec_label):
        if self.kernel.linear is True:
            w = np.zeros(features)
            for i in range(len(a)):
                w += a[i] * support_vec_label[i] * support_vec[i]
        else:
            w = None
        return w

    def __transformLabel(self, label, back=False):
        _label = label.copy()
        if back is False:
            _label[_label == 0] = -1
        else:
            _label[_label == -1] = 0
        return _label

    def train(self, train_data, train_label):
        label = self.__transformLabel(train_label)

        N, D = train_data.shape
        K = self.__getKernelMatrix(train_data)
        a = self.__getLagrangeMuil(train_data, label, self.C, K)
        support_vec = a > 1e-5
        indexis = np.arange(len(a))[support_vec]
        self.a = a[support_vec]
        self.support_vec = train_data[support_vec]
        self.support_vec_label = label[support_vec]

        self.bias = self.__getBias(self.a, self.support_vec_label, support_vec, K, indexis)
        self.weight = self.__getWeight(D, self.a, self.support_vec, self.support_vec_label)

    def predict(self, test_data):
        if self.weight is not None:
            label = np.sign(test_data @ self.weight + self.bias)
            self.predict_label = self.__transformLabel(label, True)
        else:
            predict_label = np.zeros(len(test_data))
            for i in range(len(test_data)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.support_vec_label, self.support_vec):
                    s += a * sv_y * self.kernel.calculate(test_data[i], sv)
                predict_label[i] = s
            label = np.sign(predict_label + self.bias)
            self.predict_label = self.__transformLabel(label, True)
        return self.predict_label.copy()

    def calcAccuracy(self, test_label):
        diff = test_label - self.predict_label
        corr_num = diff[diff == 0].size
        return corr_num / test_label.shape[0]


(train_data, train_label), (test_data, test_label) = getData(r"H:\机器学习\结课实验\income.csv",
                                                             keep_dim=False)
print(train_data.shape, test_data.shape)
svm = KernelSVM(LinearKernel(), 1)
svm.train(train_data, train_label)
label = svm.predict(test_data)
print(svm.calcAccuracy(test_label))
