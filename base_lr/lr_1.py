# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/15 10:59 AM'

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util


def lr_model():
    x = tf.placeholder(tf.float32, name="x")
    W = tf.Variable(tf.zeros([1]), name="w")
    b = tf.Variable(tf.zeros([1]), name="b")
    y_ = tf.placeholder(tf.float32, name="y")

    y = tf.add(tf.multiply(W,x), b, name="logit")

    lost = tf.reduce_mean(tf.square(y_ - y))
    optimizer = tf.train.GradientDescentOptimizer(0.0000001)
    train_step = optimizer.minimize(lost)

    return x, y_, lost, train_step, W, b


def train(sess, xs, ys, steps, x, y, train_op, loss):
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(steps):
        feed_dict = {x: xs, y: ys}
        sess.run([train_op, loss], feed_dict)
        if i % 100 == 0:
            print("After %d iteration:" % i)
            print("W: %f" % sess.run(w))
            print("b: %f" % sess.run(b))
            print("loss: %f" % sess.run(loss, feed_dict))


if __name__ == '__main__':
    pb_file_path = "./model/model_builder/"
    ckpt_path = "./model/model_ckpt/model"
    # x, y, loss, train_op, w, b = lr_model()
    # init = tf.global_variables_initializer()
    # xs = np.random.randint(0, 100, 1000)
    # ys = xs * 3 + 0.25
    # with tf.Session() as sess:
    #    train(sess, xs, ys, 10000, x, y, train_op, loss)
    #    # with tf.gfile.FastGFile(pb_file_path + 'model.pb', mode='wb') as f:
    #    #     constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['w', 'b', 'logit'])
    #    #     f.write(constant_graph.SerializeToString())
    #
    #    save_path = saver.save(sess, ckpt_path, global_step=10000)



    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    #
    # with tf.gfile.FastGFile(pb_file_path + 'model.pb', 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')  # 导入计算图
    #
    # input_x = sess.graph.get_tensor_by_name('x:0')
    # logit = sess.graph.get_tensor_by_name('logit:0')
    # ret = sess.run(logit, feed_dict={input_x: 5, })
    # print(ret)
    #
    # print(sess.run('w:0'))
    # print(sess.run('b:0'))


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        graph = tf.train.import_meta_graph(ckpt_path + "-10000.meta")
        graph.restore(sess, tf.train.latest_checkpoint( "./model/model_ckpt/" ))
        print(sess.run('w:0'))
        print(sess.run('b:0'))

