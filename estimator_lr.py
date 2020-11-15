# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/18 10:12 AM'

import tensorflow as tf
import numpy as np
import input_data

tf.logging.set_verbosity(tf.logging.INFO)


def convNet(x, is_training):

    x = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

    fc1 = tf.contrib.layers.flatten(conv2)

    fc1 = tf.layers.dense(fc1, 1024)
    fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_training)
    return tf.layers.dense(fc1, 10)


def model_fn(features, labels, mode, params):

    predict = convNet(features["image"], mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"result": tf.arg_max(predict, 1)})

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(tf.arg_max(predict, 1), labels)}
    return tf.estimator.EstimatorSpec(train_op=train_op, mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
model_params = {"learning_rate": 0.001}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
input_fn = tf.estimator.inputs.numpy_input_fn(x={"image": mnist.train.images},
                                                    y=mnist.train.labels.astype(np.int32),
                                                    num_epochs=2,
                                                    batch_size=128,
                                                    shuffle=True)

estimator.train(input_fn=input_fn)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"image": mnist.test.images},
                                              y=mnist.test.labels.astype(np.int32),
                                              num_epochs=200,
                                              batch_size=128,
                                              shuffle=True)

eval_results = estimator.evaluate(test_input_fn)
print(eval_results)

eval_results = estimator.evaluate(input_fn)
print(eval_results)




























