# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/14 10:57 PM'

import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist
tf.logging.set_verbosity(tf.logging.INFO)

(x_train, y_train), (x_test, y_test) = mnist.load_data()


classifier = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.numeric_column('x', shape=[28*28])],
    n_classes=10,
    hidden_units=[128, 32],
    optimizer=tf.train.AdamOptimizer(0.08)
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.reshape(x_train, [-1, 28*28])},
    y=y_train.astype(np.int32),
    num_epochs=100,
    batch_size=100,
    shuffle=True
)


def input_fn(x_data, y_data):

    def build_fn():

        return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.reshape(x_data, [-1, 28*28])},
        y=y_data.astype(np.int32),
        num_epochs=10,
        batch_size=100,
        shuffle=True
        )

    return build_fn


classifier.train(input_fn=input_fn(x_train, y_train)())


accuracy_score = classifier.evaluate(input_fn=input_fn(x_test, y_test)())["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))