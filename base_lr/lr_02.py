# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/15 11:11 AM'

import tensorflow as tf
import numpy as np


def lrModel():

    x = tf.placeholder(dtype=tf.float32, name="x", shape=[None, 1])
    y = tf.placeholder(dtype=tf.float32, name="y", shape=[None, 1])

    w = tf.get_variable(shape=[1, 1], initializer=tf.random_normal_initializer(0.0, 0.1), name="w")
    b = tf.get_variable(shape=[1], initializer=tf.random_normal_initializer(0.0, 0.1), name="b")

    y_pred = w * x + b

    loss = tf.reduce_mean(tf.square(y - y_pred))

    train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

    return x, y, w, b, loss, train_op


if __name__ == "__main__":
    x, y, w, b, loss, train_op = lrModel()

    init = tf.global_variables_initializer()

    xs = np.random.randint(0, 100, 1000)
    ys = xs * 3 + 0.25

    xs = np.reshape(xs, [-1, 1])
    ys = np.reshape(ys, [-1, 1])

    with tf.Session() as sess:
        sess.run(init)

        for i in range(10000):
            sess.run(train_op, feed_dict={x:xs, y:ys})

            if i % 100 == 0:

                loss_val, w_val, b_val = sess.run([loss, w, b], feed_dict={x:xs, y:ys})
                print(loss_val, w_val, b_val)

