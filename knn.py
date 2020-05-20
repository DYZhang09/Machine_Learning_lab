import numpy as np
import collections
import tensorflow.keras as keras
import matplotlib.pyplot as plt


class KnnClassifier(object):
    def __init__(self):
        self.X = None
        self.Y = None
        self.y_pred = None

    def train(self, X, Y):
        '''
        train for knnclassifier(just save pictures and ground-truth labels)
        :param X: a matrix(N*D) of pictures in train set
        :param Y: a matrix(1*N) of ground-truth labels
        :return: None
        '''
        self.X = X
        self.Y = Y

    def predict(self, x, k=1, L2=True):
        '''
        predict labels for pictures in test set
        :param x: a matrix(M*D) of pictures in test set
        :param k: k of knn(default 1)
        :param L2: method of calculating distance, if true use L2 else L1
        :return: a matrix(1*M) of predicted labels
        '''
        print("K: %d" % k)
        num_test = x.shape[0]  # get the size of test set
        self.y_pred = np.zeros(num_test)  # initialize result

        for i in range(num_test):
            if L2:  # calculate distance
                distance = np.sum((self.X - x[i, :]) ** 2, axis=1) ** 0.5
            else:
                distance = np.sum(np.abs(self.X - x[i, :]), axis=1)

            # get k nearest neighbor labels
            label_indexs = distance.argsort()[0:k]
            knn_label = [self.Y[label_index] for label_index in label_indexs]
            self.y_pred[i] = collections.Counter(knn_label).most_common(1)[0][0]
            if i % 10 == 0:
                print("Processing : %d" % i)

        return self.y_pred, label_indexs

    def calc_accuracy(self, y_truth):
        '''
        calculate accuracy of trained knnclassifier
        :param y_truth: a matrix(1*M) of labels of test set
        :return: accuracy
        '''
        if self.y_pred is None:  # not trained
            print("No predict labels\n")
            return None

        diff = y_truth - self.y_pred  # get difference
        num_correct = diff[diff == 0].size
        num_test = self.y_pred.size
        return num_correct / num_test


# load data
mnist = keras.datasets.mnist
(x_train_ori, y_train), (x_test_ori, y_test) = mnist.load_data()
x_train = x_train_ori.astype(np.int32)  # to avoid overflow in distance calculating
x_test = x_test_ori.astype(np.int32)
x_train = x_train.reshape(x_train.shape[0], -1)  # (60000, 28, 28) to (60000, 784)
x_test = x_test.reshape(x_test.shape[0], -1)

classifier = KnnClassifier()
classifier.train(x_train, y_train)  # train

# try different k and distance calculating methods
k_list = [1, 3, 5, 7, 9, 11, 15, 21]
L2_accu = []
L1_accu = []
for i in range(len(k_list)):  # L2
    classifier.predict(x_test, k_list[i])
    accuracy = classifier.calc_accuracy(y_test)
    L2_accu.append(accuracy)
    print("accuracy(k:%d, L2):%.2f" % (k_list[i], accuracy))

for i in range(len(k_list)):  # L1
    classifier.predict(x_test, k_list[i], False)
    accuracy = classifier.calc_accuracy(y_test)
    L1_accu.append(accuracy)
    print("accuracy(k:%d, L1):%.2f" % (k_list[i], accuracy))

print(L1_accu)
print(L2_accu)

# visualize the k nearest picture
'''title = ['1st', '2nd', '3rd']
for i in range(5):
    index = np.random.randint(0, 10000)
    pred, label_indexs = classifier.predict(x_test[index:index + 1, :], 3)

    plt.subplot(5, 4, i * 4 + 1)
    plt.imshow(x_test_ori[index])
    plt.title("Test Fig", fontsize=8)
    plt.axis('off')

    for j in range(3):
        plt.subplot(5, 4, i * 4 + 2 + j)
        plt.imshow(x_train_ori[label_indexs[j]])
        plt.title(title[j], fontsize=8)
        plt.axis('off')

plt.show()'''
