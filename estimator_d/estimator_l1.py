# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/5 8:22 PM'


import tensorflow as tf
import input_data
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


def cnnNet(x, is_training):

    x = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(x, 32, 5)  # (28 - 5 + 1) / 1 = 24
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)  # (12, 12, 32)

    conv2 = tf.layers.conv2d(pool1, 64, 5)  # (12 - 5 + 1) / 1 = 8
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # 4, 4, 64

    fc1 = tf.contrib.layers.flatten(pool2)

    fc1 = tf.layers.dense(fc1, 1024)
    fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_training)

    return tf.layers.dense(fc1, 10)


def model_fn(features, labels, mode, params):

    predict = cnnNet(features["image"], mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"result": tf.arg_max(predict, 1)})

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=labels))

    optimizor = tf.train.GradientDescentOptimizer(learning_rate=params["lr_rate"])
    train_op = optimizor.minimize(loss)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(tf.arg_max(predict, 1), labels)}

    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, eval_metric_ops=eval_metric_ops, loss=loss)


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    model_params = {"lr_rate": 0.01}
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="model/cnn")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"image": mnist.train.images},
                                                        y=mnist.train.labels.astype(np.int32),
                                                        num_epochs=2,
                                                        batch_size=128,
                                                        shuffle=True)

    estimator.train(input_fn=train_input_fn, steps=30000)


    def serving_input_fn():
        inputs = {'image': tf.placeholder(tf.float32, [None, 28, 28])}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)


    estimator.export_savedmodel(export_dir_base="DNN/saved_model", serving_input_receiver_fn=serving_input_fn)






