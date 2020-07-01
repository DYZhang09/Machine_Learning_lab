import tensorflow as tf
import tensorflow.keras as keras

from tools_logistic_and_svm import *


class LinearSVM(keras.Model):
    def __init__(self, num_class):
        super().__init__()
        self.dense = keras.layers.Dense(units=num_class, use_bias=True)

    def call(self, input):
        y = self.dense(input)
        return y


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    num_epoch = 100
    lr = 1e-3
    dataset_path = r"H:\机器学习\结课实验\income.csv"
    (train_data, train_label), (test_data, test_label) = getData(dataset_path, False)
    train_label = keras.utils.to_categorical(train_label)
    test_label = keras.utils.to_categorical(test_label)

    model = LinearSVM(num_class=2)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.hinge,
                  metrics=[keras.metrics.categorical_accuracy])
    model.fit(train_data, train_label, epochs=num_epoch)
    print(model.evaluate(test_data, test_label)[-1])
