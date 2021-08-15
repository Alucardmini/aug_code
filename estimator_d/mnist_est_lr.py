# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/10 10:12 PM'


import tensorflow as tf
import input_data
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

feature_columns = [tf.feature_column.numeric_column("image", shape=[784])]

estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[500],
    n_classes=10,
    optimizer=tf.train.AdamOptimizer(),
    model_dir="DNN")


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"image": mnist.train.images},
    y = mnist.train.labels.astype(np.int32),
    num_epochs = None,
    batch_size = 128,
    shuffle = True)

estimator.train(input_fn=train_input_fn, steps=10000)


def serving_input_fn():
    inputs = {'image': tf.placeholder(tf.float32, [None, 28, 28])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


estimator.export_savedmodel(export_dir_base="DNN/saved_model", serving_input_receiver_fn=serving_input_fn)

