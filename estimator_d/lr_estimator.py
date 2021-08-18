# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/15 10:36 AM'


import tensorflow as tf
import numpy as np

x_train = np.random.random((100, 1))
y_train = x_train* 3 + 0.7
features = {
    "x": x_train
}

feature_cols = tf.feature_column.numeric_column("x", default_value=0.0)

def model_fn(features, labels, mode, params):

    x = features['x']

    w = tf.get_variable(name="w", shape=[1], initializer=tf.zeros_initializer)
    b = tf.get_variable(name="b", shape=[1], initializer=tf.zeros_initializer)
    y_pred = w*x + b


    loss = tf.reduce_mean(tf.square(y_pred - labels))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=params['lr_rate']).minimize(loss)


    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)

if __name__ == '__main__':
    x_train = np.random.random((1000, 1)).astype(np.float32)
    y_train = x_train * 3 + 0.7

    print(x_train.dtype)


    model_params = {"lr_rate": 0.01}
    model = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="model/lr")



    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},
                                                    y=y_train,
                                                    num_epochs=10,
                                                    batch_size=8,
                                                    shuffle=True)

    model.train(input_fn=train_input_fn, steps=1000)


    for name in  model.get_variable_names():
        print(name, model.get_variable_value(name=name))








# x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
# y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
# w = tf.get_variable(name="w", shape=[1], initializer=tf.zeros_initializer)
# b = tf.get_variable(name="b", shape=[1], initializer=tf.zeros_initializer)
# y_pred = w*x + b
# loss = tf.reduce_mean(tf.square(y_pred - y))
# train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
# x_train = np.random.random((100, 1))
# y_train = x_train* 3 + 0.7
# print(x_train.shape)
# print(y_train.shape)
#
# with tf.Session() as sess:
#
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(1000):
#
#        _, loss_val = sess.run([train_op, loss], feed_dict={x: x_train, y: y_train})
#
#        if i % 100 == 0:
#            print(loss_val)





