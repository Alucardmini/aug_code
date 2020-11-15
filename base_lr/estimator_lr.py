# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/15 11:32 AM'

import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


def liner_model(x):
    w = tf.get_variable("w", shape=[1], initializer=tf.random_normal_initializer(0.0, 0.1))
    b = tf.get_variable("b", shape=[1], initializer=tf.random_normal_initializer(0.0, 0.1))
    return w*x + b


def model_fn(features, labels, mode):
    y = liner_model(features)
    loss = tf.reduce_mean(tf.square(labels - y))
    train_op = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)


def build_fn(x_data, y_data):
    return tf.estimator.inputs.numpy_input_fn(
        x=x_data.astype(np.float32),
        y=y_data.astype(np.float32),
        num_epochs=1000,
        batch_size=100,
        shuffle=True
    )

if __name__ == "__main__":
    xs = np.random.randint(0, 100, 1000)
    ys = xs * 3 + 0.25
    estimator = tf.estimator.Estimator(model_fn=model_fn,  model_dir="model/lr")

    estimator.train(build_fn(xs, ys))

    results = estimator.evaluate(build_fn(xs, ys))

    print(results)

    print(estimator.predict(build_fn(xs, ys)))
