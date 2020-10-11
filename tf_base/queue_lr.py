# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/9 9:15 AM'

import tensorflow as tf


if __name__ == "__main__":
    q = tf.FIFOQueue(3, "float")

    init = q.enqueue_many(([1.0, 3.0, 8.0],))

    x = q.dequeue()
    y = x + 1
    q_inc = q.enqueue(y)

    with tf.Session() as sess:
        sess.run(init)
        sess.run(q_inc)
        q_size = sess.run(q.size())
        for i in range(q_size):
            print(sess.run(q.dequeue()))
