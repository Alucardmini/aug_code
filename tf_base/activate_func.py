# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/9 8:43 AM'


import tensorflow as tf

if __name__ == "__main__":

    a = tf.constant([[1., 2., 3.], [3., 2., 1.]])

    with tf.Session() as sess:
        print(sess.run(tf.sigmoid(a)))
        print(sess.run(tf.nn.relu(a)))

        print(sess.run(tf.nn.dropout(a, rate=0.1, noise_shape=[2, 3])))
        print(sess.run(tf.nn.dropout(a, rate=0.1, noise_shape=[2, 1])))