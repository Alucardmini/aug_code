# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/2 6:16 PM'


import tensorflow as tf

x = tf.get_variable('x', initializer=42.)
y = tf.square(x)

optimizer = tf.train.GradientDescentOptimizer(0.1)

train_min = optimizer.minimize(y)
train_max = optimizer.minimize(-y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        cost, _ = sess.run([y, train_min])
        print("train_min ==>%s" % cost)
        # cost, _ = sess.run([y, train_max])
        # print("train_max ==>%s" % cost)
