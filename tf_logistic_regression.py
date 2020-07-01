import tensorflow as tf
import tensorflow.keras as keras

from tools_logistic_and_svm import *


class LogisticRegression(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(units=2, use_bias=True,
                                        activation=keras.activations.sigmoid)

    def call(self, inputs):
        y = self.dense(inputs)
        return y


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    num_epoch = 100
    lr = 1e-3
    dataset_path = r"H:\机器学习\结课实验\income.csv"
    (train_data, train_label), (test_data, test_label) = getData(dataset_path, False)

    model = LogisticRegression()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])
    model.fit(train_data, train_label, epochs=num_epoch)
    print(model.evaluate(test_data, test_label)[-1])
