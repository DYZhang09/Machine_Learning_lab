import numpy as np
import string
import re
import os


class NaiveBayesClassifier(object):
    def __init__(self):
        self.words_cnt_all = {}  # count words in all files
        self.words_cnt_category = {"ham": {}, "spam": {}}  # count words desperately in files of two categories
        self.samples_cnt_category = {"ham": 0.0, "spam": 0.0}  # count samples of two categories
        self.prior_p_spam = 0.0  # prior probability
        self.prior_p_ham = 0.0

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

    def __getSampleFilePath(self, dir, file_list=None):
        """
        get all txt files' path under the dir for training\n
        :param dir: root folder that contains all txt files
        :param file_list: the list of file path, default None. if not None, the file path
                            will be appended to the list
        :return: file_list(full path)
        """
        if file_list is None:
            file_list = []
        spam_files = os.listdir(os.path.join(dir, "spam"))
        ham_files = os.listdir(os.path.join(dir, "ham"))
        for spam_file in spam_files:
            if spam_file.endswith(".txt"):
                file_list.append(os.path.join(os.path.join(dir, "spam"), spam_file))
        for ham_file in ham_files:
            if ham_file.endswith(".txt"):
                file_list.append(os.path.join(os.path.join(dir, "ham"), ham_file))
        return file_list

    def train(self, data_folder_path):
        """
        train for NaiveBayes Classifier\n
        :param data_folder_path: the folder that contains all sample datas.
                structure:  [folder_path]\\[spam/ham]\\[filename]
        :return: 0 for success else return -1 if folder has something wrong
        """
        txt_file_paths = self.__getSampleFilePath(data_folder_path)
        for text_file in txt_file_paths:
            if "spam" in text_file:
                category = "spam"
            else:
                category = "ham"
            self.samples_cnt_category[category] += 1

            text = open(text_file).read()
            words = self.__tokenize(text)
            words_cnt = self.__countWords(words)
            for word, cnt in list(words_cnt.items()):
                if word not in self.words_cnt_all:
                    self.words_cnt_all[word] = 0.0
                if word not in self.words_cnt_category[category]:
                    self.words_cnt_category[category][word] = 0.0
                self.words_cnt_all[word] += cnt
                self.words_cnt_category[category][word] += cnt

        self.prior_p_ham = \
            self.samples_cnt_category["ham"] / sum(self.samples_cnt_category.values())   # get prior probability
        self.prior_p_spam = \
            self.samples_cnt_category["spam"] / sum(self.samples_cnt_category.values())



nb_classifier = NaiveBayesClassifier()
nb_classifier.train(r"H:\机器学习\实验二\Bayes数据集\english_email")