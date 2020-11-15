# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/8 8:24 PM'

import tensorflow as tf
from estimator_lr import convNet

load_graph = tf.Graph()

input_shape = [28, 28]

with load_graph.as_default():

    input_data = tf.placeholder(tf.float32, shape=input_shape)  # input_shape的batch_size维度为1
    output = convNet(input_data, False)
    net_saver = tf.compat.v1.train.Saver()

    with tf.Session() as sess:
        net_saver.restore(sess, "model/cnn/checkpoint")

        tf.train.write_graph(sess.graph, "model/pb", "pbfile")






