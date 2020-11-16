# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/15 11:32 AM'

import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


def liner_model(x):
    w = tf.get_variable("w", shape=[1], initializer=tf.random_normal_initializer(0.0, 0.1))
    b = tf.get_variable("b", shape=[1], initializer=tf.random_normal_initializer(0.0, 0.1))
    return w*x + b


def model_fn(features, labels, mode):

    y = liner_model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'feats': features,
            'logit': y,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.square(labels - y))
    train_op = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)


def build_fn(x_data, y_data):
    return tf.estimator.inputs.numpy_input_fn(
        x=x_data.astype(np.float32),
        y=y_data.astype(np.float32),
        num_epochs=1000,
        batch_size=100,
        shuffle=True
    )

if __name__ == "__main__":
    xs = np.random.randint(0, 100, 1000)
    ys = xs * 3 + 0.25
    model_path = "model/lr"
    estimator = tf.estimator.Estimator(model_fn=model_fn,  model_dir=model_path)
    estimator.train(build_fn(xs, ys))


    # estimator = tf.contrib.learn.Estimator(model_fn=model_fn,
    #                                        model_dir=model_path,
    #                                        config=tf.contrib.learn.RunConfig(
    #                                            save_checkpoints_steps=10,
    #                                            save_summary_steps=10,
    #                                            save_checkpoints_secs=None
    #                                        )
    #                                        )




    # results = estimator.evaluate(build_fn(xs, ys))
    #
    # print(results)
    #
    # predict_res = estimator.predict(build_fn(xs, ys))
    #
    # for pred in predict_res:
    #     print(pred)


    # from tensorflow.contrib import predictor
    # model = predictor.from_saved_model(model_path)  # 路径为pb模型所在目录
    #
    #
    # print(model(12.0))

    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph(model_path + '/model.ckpt-0.meta')
    #     print(model_path + '/checkpoint')
    #     # saver.restore(sess, tf.train.latest_checkpoint(model_path + '/checkpoint'))
    #     saver.restore(sess, tf.train.latest_checkpoint('/Users/wuxikun/Documents/LetsAug/base_lr/model/lr/'))



