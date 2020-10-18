# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/18 10:22 AM'


import numpy as np
import tensorflow as tf

data = np.random.randn(10, 28, 28, 1)

x = tf.constant(data)
same_conv = tf.layers.conv2d(x, 32, 5, padding="SAME")
valid_conv = tf.layers.conv2d(x, 32, 5, padding="VALID")

random_initializer = tf.random_normal_initializer(0.0, 0.001)
filter = tf.get_variable("filter", shape=[6, 6, 1, 32], initializer=random_initializer)
his_conv2d = tf.nn.conv2d(tf.cast(x, tf.float32), filter, strides=[1, 1, 1, 1], padding="VALID")

filter2 = tf.get_variable("filter2", shape=[6, 6, 32, 64], initializer=random_initializer)
his_conv2d_2 = tf.nn.conv2d(tf.cast(his_conv2d, tf.float32), filter2, strides=[1, 1, 1, 1], padding="VALID")

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print(same_conv.shape)
    print(valid_conv.shape)

    print(his_conv2d.shape)
    print(his_conv2d_2.shape)
