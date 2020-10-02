# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/9/15 9:32 PM'

import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(3, 3), dtype=tf.float32)

output = tf.nn.weighted_cross_entropy_with_logits(logits=input_data,
                                                  targets=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                                                  pos_weight=1.0)

output_2 = tf.nn.weighted_cross_entropy_with_logits(logits=input_data,
                                                  targets=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                                                  pos_weight=2.0)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(output))
    print(sess.run(output))
    print(sess.run(output_2))

    print(sess.run(tf.reduce_sum(output)))
    print(sess.run(tf.reduce_mean(output)))

