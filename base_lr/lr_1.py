# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/15 10:59 AM'

import tensorflow as tf
import numpy as np

def lr_model():
    x = tf.placeholder(tf.float32)
    W = tf.Variable(tf.zeros([1]))
    b = tf.Variable(tf.zeros([1]))
    y_ = tf.placeholder(tf.float32)

    y = W * x + b

    lost = tf.reduce_mean(tf.square(y_ - y))
    optimizer = tf.train.GradientDescentOptimizer(0.0000001)
    train_step = optimizer.minimize(lost)

    return x, y_, lost, train_step, W, b


if __name__ == '__main__':

    x, y, loss, train_op, w, b = lr_model()

    init = tf.global_variables_initializer()

    xs = np.random.randint(0, 100, 1000)
    ys = xs * 3 + 0.25

    with tf.Session() as sess:
        sess.run(init)
        steps = 100000

        for i in range(steps):

            # xs = [i]
            # ys = [3*i ]
            feed_dict={x:xs, y:ys}
            sess.run([train_op, loss], feed_dict)

            if i % 100 == 0:
                print("After %d iteration:" % i)
                print("W: %f" % sess.run(w))
                print("b: %f" % sess.run(b))
                print("lost: %f" % sess.run(loss, feed_dict))

