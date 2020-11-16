# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/15 10:36 AM'


import tensorflow as tf
import numpy as np

def lr_model():

    x = tf.placeholder("x",  dtype=tf.float32)
    y = tf.placeholder("y",  dtype=tf.float32)

    init = tf.random_normal_initializer(0.0, 1)
    w = tf.get_variable("w", shape=[1], initializer=init)
    b = tf.get_variable("b", shape=[1], initializer=init)

    y_pred = tf.nn.xw_plus_b(x, w, b)
    loss = tf.reduce_mean(tf.sqrt(y_pred*y_pred - y*y))
    opt = tf.train.GradientDescentOptimizer(0.01)
    train_op = opt.minimize(loss)

    return x, y, loss, train_op


if __name__ == "__main__":

    x_data = np.random.randint(0, 100, 100).astype(np.float32)
    y_data = 2*x_data + 0.05
    init = tf.global_variables_initializer()
    x, y, loss_op, train_op = lr_model()
    with tf.Session() as sess:
        sess.run(init)
        loss, _ = sess.run([loss_op, train_op], feed_dict={x: x_data, y: y_data})


