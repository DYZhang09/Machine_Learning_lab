import string
import re
import os
import math
import random
import jieba
import time
import matplotlib.pyplot as plt


class NaiveBayesClassifier(object):
    def __init__(self, stop_word_file, polynomial=True, cn=False):
        """
        initialize the Naive Bayes classifiers\n
        :param stop_word_file: the .txt file that contains stop words
        :param polynomial: if polynomial is True, the classifier will use polynomial model,
                            else will use bernoulli model. Default True
        :param cn: if the model is trained for Chinese emails then True. Default False
        """
        self.words_cnt_all = {}  # count words in all files
        self.words_cnt_category = {"ham": {}, "spam": {}}  # count words desperately in files of two categories
        self.samples_cnt_category = {"ham": 0.0, "spam": 0.0}  # count samples of two categories
        self.words_prob_category = {"ham": {}, "spam": {}}  # the probabilities of words' appearance in two categories
        self.prior_p_spam = 0.0  # prior probability
        self.prior_p_ham = 0.0
        self.predict_file_label_dict = {}  # store the result of prediction
        self.cn = cn  # whether the emails are Chinese emails or not
        self.stop_words_file = stop_word_file  # stop words file path for Chinese emails
        self.stop_words = []  # store the stop words
        self.polynomial = polynomial

    def __removePunctuation(self, s):
        """
        remove punctuations from an English string\n
        :param s: string
        :return: string without punctuations
        """
        exclude = set(string.punctuation)
        return ''.join(c for c in s if c not in exclude)

    def __tokenize(self, s):
        """
        remove punctuations and then split an English string\n
        :param s: string that needs to split
        :return: a list of split string
        """
        text = self.__removePunctuation(s)
        text = text.lower()
        text = re.sub("\d+", " ", text)
        return re.split("\W+", text)

    def __countWords(self, words):
        """
        count words number for an English string\n
        :param words: string that needs to count words
        :return: a Python dictionary that contains information about words and their number
        """
        if len(self.stop_words) == 0:
            self.stop_words = [line.strip() for line in open(self.stop_words_file).readlines()]
            print("Get stop words(EN) successfully!")

        words_count = {}
        words_en = self.__tokenize(words)
        for word in words_en:
            if word not in self.stop_words:
                words_count[word] = words_count.get(word, 0.0) + 1.0
        return words_count

    def __countWordsCN(self, words):
        """
        count words number for a Chinese string(exclude stop words)\n
        :param words: string that needs to count words
        :return: a Python dictionary that contains information about words and their number
        """
        if len(self.stop_words) == 0:  # get stop words list
            self.stop_words = [line.strip() for line in open(self.stop_words_file, encoding='gb2312').readlines()]
            print("Get stop words(CN) successfully!")

        words_count = {}
        words_cn = re.sub(u"([^\u4e00-\u9fa5])", "", words)  # ignore numbers and alphas
        text = jieba.lcut(words_cn)  # split Chinese words
        for word in text:
            if word not in self.stop_words:
                words_count[word] = words_count.get(word, 0.0) + 1.0
        return words_count

    def train(self, sample_path_list, sample_label_list):
        """
        train for Naive Bayes Classifier\n
        :param sample_path_list: a list of sample emails'path for training
        :param sample_label_list: a list of sample emails'labels
        :return: None
        """
        time_start = time.time()
        for i in range(len(sample_path_list)):
            if (i + 1) % 1000 == 0:
                print("Processing(train): ", i + 1)
            text_file = sample_path_list[i]
            category = sample_label_list[i]
            self.samples_cnt_category[category] += 1

            if self.cn is False:  # read file and count words
                text = open(text_file).read()
                words_cnt = self.__countWords(text)
            else:
                text = open(text_file, encoding='gb2312').read()
                words_cnt = self.__countWordsCN(text)

            for word, cnt in words_cnt.items():
                if word not in self.words_cnt_all:
                    self.words_cnt_all[word] = 0.0
                if word not in self.words_cnt_category[category]:
                    self.words_cnt_category[category][word] = 0.0
                self.words_cnt_all[word] += cnt
                if self.polynomial:
                    self.words_cnt_category[category][word] += cnt
                else:
                    self.words_cnt_category[category][word] += 1

        print("Calculating Probabilities...")
        self.prior_p_ham = \
            self.samples_cnt_category["ham"] / sum(self.samples_cnt_category.values())  # get prior probability
        self.prior_p_spam = \
            self.samples_cnt_category["spam"] / sum(self.samples_cnt_category.values())
        sum_word_cnt_ham = sum(self.words_cnt_category["ham"].values())  # get total number of words of two categories
        sum_word_cnt_spam = sum(self.words_cnt_category["spam"].values())
        for word, cnt in self.words_cnt_category["ham"].items():  # get words probability in two categories, P(X|Y)
            self.words_prob_category["ham"][word] = cnt / sum_word_cnt_ham
        for word, cnt in self.words_cnt_category["spam"].items():
            self.words_prob_category["spam"][word] = cnt / sum_word_cnt_spam

        time_end = time.time()
        print("Time cost(train): %d s\n" % (time_end - time_start))

    def predict(self, data_path_list):
        """
        use Naive Bayes classifier to predict emails\n
        :param data_path_list: a list of sample emails'path for testing
        :return: a python dictionary that contains information of file paths and their classes
        """
        time_start = time.time()
        words_cnt_ham = sum(self.words_cnt_category["ham"].values())  # total number of words in two categories
        words_cnt_spam = sum(self.words_cnt_category["spam"].values())
        words_kinds_ham = len(self.words_cnt_category["ham"].values())  # number of different words in two categories
        words_kinds_spam = len(self.words_cnt_category["spam"].values())  # used for smooth while estimating P(X|Y)

        for i in range(len(data_path_list)):
            if (i + 1) % 1000 == 0:
                print("Processing(test): ", i + 1)

            log_p_ham = 0.0  # log of ham/spam probability
            log_p_spam = 0.0

            data_path = data_path_list[i]
            if self.cn is False:  # real file and count words
                text = open(data_path).read()
                words_cnt = self.__countWords(text)
            else:
                text = open(data_path, encoding='gb2312').read()
                words_cnt = self.__countWordsCN(text)

            for word, cnt in words_cnt.items():
                word_cnt_ham = self.words_cnt_category["ham"].get(word, 0.)
                word_cnt_spam = self.words_cnt_category["spam"].get(word, 0.)

                theta_ham = (self.words_prob_category["ham"][word]  # get the estimate of P(X|Y)
                             if word_cnt_ham else 1 / (words_kinds_ham + words_cnt_ham))
                theta_spam = (self.words_prob_category["spam"][word]
                              if word_cnt_spam else 1 / (words_kinds_spam + words_cnt_spam))

                if self.polynomial:
                    log_p_ham += cnt * math.log(theta_ham)
                    log_p_spam += cnt * math.log(theta_spam)
                else:
                    log_p_ham += math.log(theta_ham)
                    log_p_spam += math.log(theta_spam)

            log_p_ham += math.log(self.prior_p_ham)  # get the probability of two categories
            log_p_spam += math.log(self.prior_p_spam)
            category = "ham" if log_p_ham >= log_p_spam else "spam"
            self.predict_file_label_dict[data_path] = category

        time_end = time.time()
        print("Time cost(test): %d s\n" % (time_end - time_start))
        return self.predict_file_label_dict

    def calcAccuracy(self, test_label_list):
        """
        calculate the accuracy of prediction\n
        :param test_label_list: the list of ground-truth labels of predict data
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
    :return: (train_mails, train_labels), (test_mails, test_labels)
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

    mail_num = len(file_list)  # split all emails into two parts, one for train and the other for test
    index = [i for i in range(mail_num)]
    random.shuffle(index)
    index_train = index[:(int)(mail_num * train_ratio)]
    index_test = index[(int)(mail_num * train_ratio):]
    train_mails = [file_list[i] for i in index_train]
    test_mails = [file_list[i] for i in index_test]
    train_labels = [label_list[i] for i in index_train]
    test_labels = [label_list[i] for i in index_test]

    return (train_mails, train_labels), (test_mails, test_labels)


def visualizeSpamWordsProb(dict_spam_words_prob, cn=False):
    """
    visualize the probability of top 10 words in spams\n
    :param dict_spam_words_prob: a python dictionary, elements are (word, count)
    :return: None
    """
    sort_dict_spam_words = sorted(dict_spam_words_prob.items(), key=lambda item: item[1])[-10:]
    sort_list_words = []
    sort_list_cnts = []
    for word, cnt in sort_dict_spam_words:
        sort_list_words.append(word)
        sort_list_cnts.append(cnt)

    plt.rcParams["font.family"] = "STSong"
    plt.barh(range(len(sort_list_cnts)), sort_list_cnts, color='steelblue', tick_label=sort_list_words)
    for x, y in enumerate(sort_list_cnts):
        plt.text(y + 0.0001, x, '%.4f' % y)
    plt.title("Probability of the 10 most frequent words in %s spam" % ('CN' if cn else 'EN'))
    plt.xlabel('Probability')
    plt.ylabel('Word')
    plt.show()


(train_mails, train_labels), (test_mails, test_labels) = getData(r"H:/机器学习/实验二/Bayes数据集/english_email")
nb_classifier = NaiveBayesClassifier(stop_word_file=r"H:\机器学习\实验二\Bayes数据集\english_email\stopwords.txt")
nb_classifier.train(train_mails, train_labels)
nb_classifier.predict(test_mails)
print("English emails accuracy: %.5f\n" % nb_classifier.calcAccuracy(test_labels))
spam_words_prob = nb_classifier.words_prob_category["spam"]
visualizeSpamWordsProb(spam_words_prob)

(train_mails, train_labels), (test_mails, test_labels) \
    = getData(r"H:\机器学习\实验二\Bayes数据集\trec06c\data\newindex.txt", train_ratio=0.7, cn=True)
train_nums = [100, 500, 1000, 2000, 5000, 10000, 15000, 25000, 35000]
accuracy_list_poly = []
accuracy_list_bernoulli = []
for num in train_nums:
    nb_classifier_cn = NaiveBayesClassifier(cn=True,
                                            stop_word_file=r"H:\机器学习\实验二\Bayes数据集\trec06c\data\中文停用词表GBK.txt")
    nb_classifier_cn.train(train_mails[:num], train_labels[:num])
    nb_classifier_cn.predict(test_mails)
    accuracy = nb_classifier_cn.calcAccuracy(test_labels)
    accuracy_list_poly.append(accuracy)
    print("Chinese emails accuracy: %.5f\n" % accuracy)
    # spam_words_prob_cn = nb_classifier_cn.words_prob_category["spam"]
    # visualizeSpamWordsProb(spam_words_prob_cn, cn=True)

    nb_classifier_cn_b = NaiveBayesClassifier(cn=True,
                                              stop_word_file=r"H:\机器学习\实验二\Bayes数据集\trec06c\data\中文停用词表GBK.txt",
                                              polynomial=False)
    nb_classifier_cn_b.train(train_mails[:num], train_labels[:num])
    nb_classifier_cn_b.predict(test_mails)
    accuracy = nb_classifier_cn_b.calcAccuracy(test_labels)
    accuracy_list_bernoulli.append(accuracy)
    print("Chinese emails accuracy: %.5f\n" % accuracy)
    # spam_words_prob_cn = nb_classifier_cn.words_prob_category["spam"]
    # visualizeSpamWordsProb(spam_words_prob_cn, cn=True)
print(accuracy_list_poly)
print(accuracy_list_bernoulli)
plt.plot(train_nums, accuracy_list_poly, 's-', color='r', label="polynomial")
plt.plot(train_nums, accuracy_list_bernoulli, 'o-', color='g', label="bernoulli")
plt.xlabel("number of train samples")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()
