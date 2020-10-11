# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/9 1:27 PM'

import tensorflow as tf
import numpy as np


class Sqrt(object):
    def __init__(self, learn_rate=0.001):
        super(Sqrt, self).__init__
        self.weights = dict()
        self.learn_rate = learn_rate
        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(dtype=tf.float32, name="x", shape=[None, 1])
        self.y = tf.placeholder(dtype=tf.float32, name="y", shape=[None, 1])

        random_initializer = tf.random_normal_initializer(0, 0.005)

        self.weights = {
            'w1': tf.get_variable(name='w1', shape=[1, 30], initializer=random_initializer),
            'b1': tf.get_variable(name='b1', shape=[30], initializer=random_initializer),
            'w2': tf.get_variable(name='w2', shape=[30, 1], initializer=random_initializer),
            'b2': tf.get_variable(name='b2', shape=[1], initializer=random_initializer)
        }

        h1 = tf.nn.relu(tf.nn.xw_plus_b(self.x, self.weights['w1'], self.weights['b1']))
        self.output = tf.nn.xw_plus_b(h1, self.weights['w2'], self.weights['b2'])

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.output - self.y), reduction_indices=[1]))
        self.train_op = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.loss)

    def train(self, sess: tf.Session, train_x, train_y, test_x, test_y, epoch=1000, mod=10):

        for ep in range(epoch):
            sess.run(self.train_op, feed_dict={self.x: train_x, self.y: train_y})
            if ep % mod == 0:
                loss,  _ = sess.run([self.loss, self.train_op],
                                        feed_dict={self.x: test_x,
                                                   self.y: test_y})
                print(self.__class__.__name__, "epoch", ep, "loss:", loss)

    def predict(self, sess, x):
        out_put = sess.run([self.output], feed_dict={self.x: x})
        return out_put


if __name__ == "__main__":
    train_x = np.linspace(0.0, 10, 20000)
    # noise = np.random.normal(0, 0.0005, train_x.shape)
    # train_y = np.sqrt(train_x) + noise
    train_y = np.sqrt(train_x)
    train_x = np.reshape(train_x, [-1, 1])
    train_y = np.reshape(train_y, [-1, 1])

    model = Sqrt(0.0001)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess, train_x, train_y, train_x, train_y, epoch=2000)

        print("predict-> ", model.predict(sess, np.array([[1]])))
        print("predict-> ", model.predict(sess, np.array([[2]])))
        print("predict-> ", model.predict(sess, np.array([[3]])))
        print("predict-> ", model.predict(sess, np.array([[5]])))
        print("predict-> ", model.predict(sess, np.array([[8]])))
        print("predict-> ", model.predict(sess, np.array([[16]])))
        print("predict-> ", model.predict(sess, np.array([[81]])))
        print("predict-> ", model.predict(sess, np.array([[100]])))
        print("predict-> ", model.predict(sess, np.array([[6400]])))

