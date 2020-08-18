# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/8/18 2:43 PM'

import tensorflow as tf
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np

import input_data


class LRMNIST(object):
    def __init__(self, learning_rate=0.1):
        super(LRMNIST, self).__init__()
        self.learning_rate = learning_rate
        self.build_graph()

    def build_graph(self):

        self.x = tf.placeholder(tf.float32, shape=[None, 28*28], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
        w = tf.Variable(tf.random_uniform([28*28, 10], minval=0.0, maxval=0.99), name="w")
        b = tf.Variable(tf.zeros([10]), name="b")
        y_pred = tf.nn.softmax(tf.add(tf.matmul(self.x, w), b))

        self.loss = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)), reduction_indices=1))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def train(self, sess:tf.Session, train_x, train_y, test_x, test_y, epoch=100, mod=10):

        for ep in range(epoch):
            sess.run(self.accuracy, feed_dict={self.x: train_x, self.y: train_y})
            if ep % mod == 0:
                loss, acc, _ = sess.run([self.loss, self.accuracy, self.train_op], feed_dict={self.x: test_x, self.y: test_y})
                print(self.__class__.__name__, "epoch", ep, "loss:", loss, "acc:", acc)


class CNNMNIST(object):

    def __init__(self, learning_rate=0.1):
        super(CNNMNIST, self).__init__()
        self.learning_rate = learning_rate
        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 28*28], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
        image = tf.reshape(self.x, [-1, 28, 28, 1])
        filter_one = tf.Variable(tf.truncated_normal([5, 5, 1, 32], 0.0, 0.1))
        biase_1 = tf.Variable(tf.constant(0.1, shape=[32]))
        conv1 = tf.nn.conv2d(image, filter_one, strides=[1, 1, 1, 1], padding="SAME") # batch_sz, 28,28,32
        h_conv1 = tf.nn.relu(conv1 + biase_1)
        pool_layer = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")   # (batch_size, 14, 14, 32)
        fc = tf.reshape(pool_layer, [-1, 14*14*32])
        w = tf.Variable(tf.truncated_normal([14*14*32, 10], 0.0, 0.1), name="w")
        y_pred = tf.nn.softmax(tf.matmul(fc, w))
        self.loss = -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def train(self, sess: tf.Session, train_x, train_y, test_x, test_y, epoch=100, mod=10):
        for ep in range(epoch):
            sess.run(self.accuracy, feed_dict={self.x: train_x, self.y: train_y})
            if ep % mod == 0:
                loss, acc, _ = sess.run([self.loss, self.accuracy, self.train_op],
                                        feed_dict={self.x: test_x, self.y: test_y})
                print(self.__class__.__name__, "epoch", ep, "loss:", loss, "acc:", acc)

if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(x_train.shape, y_train.shape)
    # x_train = np.reshape(x_train, [-1, 28*28])
    # y_train = np_utils.to_categorical(y_train, 10)
    # x_test = np.reshape(x_test, [-1, 28*28])
    # y_test = np_utils.to_categorical(y_test, 10)
    #
    # print(x_train.shape, y_train.shape)
    #
    # # model = LRMNIST(0.1)
    # model = CNNMNIST(1e-4)
    # init_global = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_global)
    #     model.train(sess, x_train, y_train, x_test, y_test, epoch=3000, mod=1)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    model = CNNMNIST(0.0015)
    # model = LRMNIST(0.1)
    init_global = tf.global_variables_initializer()
    batch = mnist.train.next_batch(50)

    with tf.Session() as sess:
        sess.run(init_global)
        model.train(sess, batch[0], batch[1], batch[0], batch[1], epoch=2000, mod=100)















