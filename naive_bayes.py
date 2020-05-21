import numpy as np
import string
import re
import os
import math
import random


class NaiveBayesClassifier(object):
    def __init__(self):
        self.words_cnt_all = {}  # count words in all files
        self.words_cnt_category = {"ham": {}, "spam": {}}  # count words desperately in files of two categories
        self.samples_cnt_category = {"ham": 0.0, "spam": 0.0}  # count samples of two categories
        self.words_prob_category = {"ham": {}, "spam": {}}
        self.prior_p_spam = 0.0  # prior probability
        self.prior_p_ham = 0.0
        self.predict_file_label_dict = {}

    def __removePunctuation(self, s):
        """
        to remove punctuations from a string\n
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

    def train(self, sample_path_list):
        """
        train for NaiveBayes Classifier\n
        :param data_folder_path: the folder that contains all sample datas.
                structure:  [folder_path]\\[spam/ham]\\[filename]
        :return: 0 for success else return -1 if folder has something wrong
        """
        for text_file in sample_path_list:
            if "spam" in text_file:
                category = "spam"
            else:
                category = "ham"
            self.samples_cnt_category[category] += 1

            text = open(text_file).read()
            words = self.__tokenize(text)
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

        self.prior_p_ham = \
            self.samples_cnt_category["ham"] / sum(self.samples_cnt_category.values())  # get prior probability
        self.prior_p_spam = \
            self.samples_cnt_category["spam"] / sum(self.samples_cnt_category.values())
        for word, cnt in self.words_cnt_category["ham"].items():
            self.words_prob_category["ham"][word] = cnt / sum(self.words_cnt_category["ham"].values())
        for word, cnt in self.words_cnt_category["spam"].items():
            self.words_prob_category["spam"][word] = cnt / sum(self.words_cnt_category["spam"].values())

    def predict(self, data_path_list):
        """
        use Naive Bayes classifier to predict emails\n
        :param data_folder_path: the root folder that contains mails need to predict
        :return: a python dictionary that contains information of file paths and their classes
        """
        for data_path in data_path_list:
            text = open(data_path).read()
            words = self.__tokenize(text)
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

        return self.predict_file_label_dict

    def calcAccuracy(self, label_list):
        """
        calculate the accuracy of prediction\n
        :param label_list: the list of ground-truth labels of predict data
        :return: accuracy
        """
        index = 0
        file_num = len(self.predict_file_label_dict.values())
        correct_num = 0.
        for mail_path, category in self.predict_file_label_dict.items():
            correct_num += (1. if category == label_list[index] else 0.)
            index += 1
        return correct_num / file_num


def getData(dir):
    """
    get all txt files'path under the dir\n
    :param dir: root folder that contains all mails
    :return: file list(full path) , label list
    """
    file_list = []
    label_list = []
    spam_files = os.listdir(os.path.join(dir, "spam"))
    ham_files = os.listdir(os.path.join(dir, "ham"))
    for spam_file in spam_files:
        if spam_file.endswith(".txt"):
            file_list.append(os.path.join(os.path.join(dir, "spam"), spam_file))
            label_list.append("spam")
    for ham_file in ham_files:
        if ham_file.endswith(".txt"):
            file_list.append(os.path.join(os.path.join(dir, "ham"), ham_file))
            label_list.append("ham")
    return file_list, label_list


mail_paths, label_list = getData(r"H:\机器学习\实验二\Bayes数据集\english_email")
mail_num = len(mail_paths)
index = [i for i in range(mail_num)]
random.shuffle(index)
index_train = index[:(int)(mail_num * 0.7)]
index_test = index[(int)(mail_num * 0.7):]

train_mails = [mail_paths[i] for i in index_train]
test_mails = [mail_paths[i] for i in index_test]
test_labels = [label_list[i] for i in index_test]

nb_classifier = NaiveBayesClassifier()
nb_classifier.train(train_mails)
nb_classifier.predict(test_mails)
print(nb_classifier.calcAccuracy(test_labels))
