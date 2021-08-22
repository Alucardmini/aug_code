# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/21 10:23 PM'

import os
import tensorflow as tf
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tf.logging.set_verbosity(tf.logging.INFO)


def my_init_fn():

    x = np.arange(100, dtype=np.float32) / 100.0
    np.random.seed(0)
    y = x * 0.8 - 0.2 + np.random.normal(0, 0.1, size=[100])
    d = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)


    return d.make_one_shot_iterator().get_next()


def serving_input_fn():
    # features_placeholder = tf.placeholder(tf.float32, [None])
    # labels_placeholder = tf.placeholder(tf.float32, [None])

    inputs = {'x': tf.placeholder(tf.float32, [None])
              }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def model_fn(features, labels, mode, params):

    logits = tf.layers.dense(tf.reshape(features, [-1, 1]), 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 获取损失函数
    if labels is not None:
        labels = tf.to_float(tf.reshape(labels, [-1, 1]))
    loss = tf.losses.mean_squared_error(labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        # 定义性能指标
        mean_absolute_error = tf.metrics.mean_absolute_error(labels, logits)
        metrics = {'mean_absolute_error': mean_absolute_error}
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # 构建优化器与梯度更新操作
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)




if __name__ == '__main__':

    model = tf.estimator.Estimator(model_fn=model_fn, model_dir='./logs')

    model.export_saved_model("./logs/pb/", serving_input_fn)

    # NUM_EPOCHS = 20
    # for i in range(NUM_EPOCHS):
    #     model.train(my_init_fn)


    # # 预测
    # # 每行预测结果类似 {'logits': array([0.07357793], dtype=float32)}
    # # 这里使用 my_init_fn2 也有一样的结果
    # predictions = model.predict(input_fn=my_init_fn)
    # for pred in predictions:
    #     print(pred)
    #
    # # 评估
    # # 评估结果类似 {'loss': 0.03677843, 'mean_absolute_error': 0.15645184, 'global_step': 500}
    # print(model.evaluate(my_init_fn))