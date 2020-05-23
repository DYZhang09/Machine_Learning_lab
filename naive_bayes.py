import numpy as np
import string
import re
import os
import math
import random
import jieba


class NaiveBayesClassifier(object):
    def __init__(self, cn=False, stop_word_file=None):
        self.words_cnt_all = {}  # count words in all files
        self.words_cnt_category = {"ham": {}, "spam": {}}  # count words desperately in files of two categories
        self.samples_cnt_category = {"ham": 0.0, "spam": 0.0}  # count samples of two categories
        self.words_prob_category = {"ham": {}, "spam": {}}
        self.prior_p_spam = 0.0  # prior probability
        self.prior_p_ham = 0.0
        self.predict_file_label_dict = {}
        self.cn = cn
        self.stop_words_file = stop_word_file
        self.stop_words = []

    def __removePunctuation(self, s):
        """
        remove punctuations from a string\n
        :param s: string
        :return: string without punctuations
        """
        exclude = set(string.punctuation)
        return ''.join(c for c in s if c not in exclude)

    def __tokenize(self, s):
        """
        remove punctuations and then split a string\n
        :param s: string that needs to split
        :return: a list of split string
        """
        text = self.__removePunctuation(s)
        text = text.lower()
        return re.split("\W+", text)

    def __removeStopWords(self, s_cn):
        """
        remove stop words from a Chinese string\n
        :param s_cn: a Chinese string
        :param stop_words_file: the path of stop words file
        :return: a Chinese string without stop words
        """
        if len(self.stop_words) == 0:
            self.stop_words = [open(self.stop_words_file, encoding='gbk').read().split()]
            print("Get stop words successfully!")
            print(self.stop_words)
        return ''.join(c for c in s_cn if c not in self.stop_words)

    def __tokenize_cn(self, s_cn):
        """
        remove stop words and then split a Chinese string\n
        :param s_cn: a Chinese string that needs to split
        :param stop_words_file: the path of stop words file
        :return: a list of split Chinese string
        """
        text = self.__removeStopWords(s_cn)
        return jieba.lcut(text)

    def __countWords(self, words):
        """
        count words number\n
        :param words: string that needs to count words
        :return: a dictionary that contains information about words and their number
        """
        words_count = {}
        for word in words:
            words_count[word] = words_count.get(word, 0.0) + 1.0
        return words_count

    def train(self, sample_path_list, sample_label_list, stop_words_file=None):
        """
        train for NaiveBayes Classifier\n
        :param sample_path_list: a list of sample emails'path for training
        :param sample_label_list: a list of sample emails'labels
        :param cn: if mails are Chinese emails then True, default False
        :return: None
        """
        for i in range(len(sample_path_list)):
            text_file = sample_path_list[i]
            category = sample_label_list[i]
            self.samples_cnt_category[category] += 1

            if self.cn is False:
                text = open(text_file).read()
                words = self.__tokenize(text)
            else:
                text = open(text_file, encoding='gb2312').read()
                words = self.__tokenize_cn(text)
            words_cnt = self.__countWords(words)
            for word, cnt in list(words_cnt.items()):
                if len(word) <= 3:
                    continue
                if word not in self.words_cnt_all:
                    self.words_cnt_all[word] = 0.0
                if word not in self.words_cnt_category[category]:
                    self.words_cnt_category[category][word] = 0.0
                self.words_cnt_all[word] += cnt
                self.words_cnt_category[category][word] += cnt
            if ((i + 1) % 1000 == 0):
                print("Processing(train): ", i + 1)

        self.prior_p_ham = \
            self.samples_cnt_category["ham"] / sum(self.samples_cnt_category.values())  # get prior probability
        self.prior_p_spam = \
            self.samples_cnt_category["spam"] / sum(self.samples_cnt_category.values())
        for word, cnt in self.words_cnt_category["ham"].items():  # get words probability in two categories, P(X|Y)
            self.words_prob_category["ham"][word] = cnt / sum(self.words_cnt_category["ham"].values())
        for word, cnt in self.words_cnt_category["spam"].items():
            self.words_prob_category["spam"][word] = cnt / sum(self.words_cnt_category["spam"].values())

    def predict(self, data_path_list):
        """
        use Naive Bayes classifier to predict emails\n
        :param data_folder_path: the root folder that contains mails need to predict
        :return: a python dictionary that contains information of file paths and their classes
        """
        for i in range(len(data_path_list)):
            data_path = data_path_list[i]
            if self.cn is False:
                text = open(data_path).read()
                words = self.__tokenize(text)
            else:
                text = open(data_path, encoding='gb2312').read()
                words = self.__tokenize_cn(text)
            words_cnt = self.__countWords(words)

            log_p_ham = 0.0  # log of ham/spam probability
            log_p_spam = 0.0

            words_cnt_ham = sum(self.words_cnt_category["ham"].values())
            words_cnt_spam = sum(self.words_cnt_category["spam"].values())
            words_kinds_ham = len(self.words_cnt_category["ham"].values())
            words_kinds_spam = len(self.words_cnt_category["spam"].values())
            for word, cnt in words_cnt.items():
                if len(word) <= 3:
                    continue

                word_cnt_ham = self.words_cnt_category["ham"].get(word, 0.)
                word_cnt_spam = self.words_cnt_category["spam"].get(word, 0.)

                theta_ham = (self.words_prob_category["ham"][word]
                             if word_cnt_ham else 1 / (words_kinds_ham + words_cnt_ham))
                theta_spam = (self.words_prob_category["spam"][word]
                              if word_cnt_spam else 1 / (words_kinds_spam + words_cnt_spam))

                log_p_ham += cnt * math.log(theta_ham)
                log_p_spam += cnt * math.log(theta_spam)

            log_p_ham += self.prior_p_ham
            log_p_spam += self.prior_p_spam
            category = "ham" if log_p_ham >= log_p_spam else "spam"
            self.predict_file_label_dict[data_path] = category
            if ((i + 1) % 1000 == 0):
                print("Processing(test): ", i + 1)

        return self.predict_file_label_dict

    def calcAccuracy(self, test_label_list):
        """
        calculate the accuracy of prediction\n
        :param label_list: the list of ground-truth labels of predict data
        :return: accuracy
        """
        index = 0
        file_num = len(self.predict_file_label_dict.values())
        correct_num = 0.
        for mail_path, category in self.predict_file_label_dict.items():
            correct_num += (1. if category == test_label_list[index] else 0.)
            index += 1
        return correct_num / file_num


def getData(path, train_ratio=0.7, cn=False):
    """
    get all emails'path and their labels\n
    :param path: if cn is False, path is the root folder that contains English emails else
                path is the index file for CN emails
    :param train_ratio: (number of emails for train) / (number of total emails) (default 0.7)
    :param cn: if mails are Chinese emails then True, default False
    :return: file list(full path) , label list
    """
    file_list = []
    label_list = []
    if cn is False:
        spam_files = os.listdir(os.path.join(path, "spam"))
        ham_files = os.listdir(os.path.join(path, "ham"))
        for spam_file in spam_files:
            if spam_file.endswith(".txt"):
                file_list.append(os.path.join(os.path.join(path, "spam"), spam_file))
                label_list.append("spam")
        for ham_file in ham_files:
            if ham_file.endswith(".txt"):
                file_list.append(os.path.join(os.path.join(path, "ham"), ham_file))
                label_list.append("ham")
    else:
        index_file = open(path)
        root_path = os.path.dirname(path)
        for line in open(path):
            info = line.split()
            file_list.append(os.path.join(root_path, info[1]))
            label_list.append(info[0])

    mail_num = len(file_list)
    index = [i for i in range(mail_num)]
    random.shuffle(index)
    index_train = index[:(int)(mail_num * train_ratio)]
    index_test = index[(int)(mail_num * train_ratio):]
    train_mails = [file_list[i] for i in index_train]
    test_mails = [file_list[i] for i in index_test]
    train_labels = [label_list[i] for i in index_train]
    test_labels = [label_list[i] for i in index_test]

    return (train_mails, train_labels), (test_mails, test_labels)


(train_mails, train_labels), (test_mails, test_labels) = getData(r"H:/机器学习/实验二/Bayes数据集/english_email")

nb_classifier = NaiveBayesClassifier()
nb_classifier.train(train_mails, train_labels)
nb_classifier.predict(test_mails)
print("English emails accuracy: ", nb_classifier.calcAccuracy(test_labels))

(train_mails, train_labels), (test_mails, test_labels) \
    = getData(r"H:\机器学习\实验二\Bayes数据集\trec06c\data\newindex.txt", train_ratio=0.0001, cn=True)
nb_classifier_cn = NaiveBayesClassifier(cn=True,
                                        stop_word_file=r"H:\机器学习\实验二\Bayes数据集\trec06c\data\中文停用词表GBK.txt")
nb_classifier_cn.train(train_mails, train_labels)
print(nb_classifier_cn.words_cnt_category)
#nb_classifier_cn.predict(test_mails)
#print("Chinese emails accuracy: ", nb_classifier_cn.calcAccuracy(test_labels))
