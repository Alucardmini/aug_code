# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/3 5:02 PM'

import tensorflow as tf
import numpy as np
import sys
from FMFamily.data_utils import load_data


class ModelArgs(object):
    feat_size = 100
    embed_size = 100
    deep_layers = [512, 256, 128]
    field_size = 100
    epoch = 3
    batch_size = 64
    learning_rate = 0.05
    is_train = True
    l2_reg_rate = 0.01
    checkpoint_dir = '../model'


class FFM(object):

    def __init__(self, args: ModelArgs):
        super(FFM, self).__init__()

        self.feat_size = args.feat_size
        self.field_size = args.field_size
        self.feat_embed_size = args.embed_size

        self.learning_rate = args.learning_rate
        self.l2_reg_rate = args.l2_reg_rate

        self.weights = dict()
        self.build_model()

    def build_model(self):

        self.feat_index = tf.placeholder(shape=[None, None], dtype=tf.int32, name="feat_index")
        self.feat_value = tf.placeholder(shape=[None, None], dtype=tf.float32, name="feat_value")
        self.label = tf.placeholder(shape=[None, None], dtype=tf.float32, name="label")

        random_initializer = tf.random_normal_initializer(0.0, 0.01)
        self.weights["first_factor_weight"] = tf.get_variable(name="first_factor_weight", shape=[self.feat_size, 1],
                                                              initializer=random_initializer)
        # 第一个因子
        self.first_factor = tf.multiply(tf.nn.embedding_lookup(self.weights["first_factor_weight"], self.feat_index),
                    tf.reshape(self.feat_value, [-1, self.field_size, 1]))

        self.first_factor = tf.reduce_sum(self.first_factor, axis=-1)

        # 第二个因子
        self.weights["embedding_weight"] = tf.get_variable(name="embedding_weight",
                                                           shape=[self.field_size, self.feat_size, self.feat_embed_size],
                                                           initializer=random_initializer)

        ffm_part = tf.constant(0, dtype=tf.float32)

        for i in range(self.field_size):
            for j in range(i + 1, self.field_size):

                vi_fj = tf.nn.embedding_lookup(self.weights["embedding_weight"][j], self.feat_index)
                vj_fi = tf.nn.embedding_lookup(self.weights["embedding_weight"][i], self.feat_index)

                wij = tf.reduce_sum(tf.multiply(vi_fj, vj_fi), axis=2)
                x_i = self.feat_value[:, i]
                x_j = self.feat_value[:, j]
                xij = tf.multiply(x_i, x_j)
                xij = tf.expand_dims(xij, 1)

                ffm_part += tf.multiply(wij, xij)

        merge = ffm_part + self.first_factor

        self.weights["merge_layer"] = tf.get_variable(name="merge_layer",
                                                           shape=[self.field_size, self.feat_embed_size],
                                                           initializer=random_initializer)

        self.weights["merge_bias"] = tf.get_variable(name="merge_bias",
                                                           shape=[1],
                                                           initializer=random_initializer)

        self.out = tf.add(tf.matmul(merge, self.weights["merge_layer"]), self.weights["merge_bias"])

        self.out = tf.nn.sigmoid(self.out)

        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["merge_layer"])

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()

        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, feat_index, feat_value, label):
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label
        })
        return loss, step

    def predict(self, sess, feat_index, feat_value):
        result = sess.run([self.out], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], np.array(y[start:end])

if __name__ == "__main__":
    args = ModelArgs()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    data = load_data()
    args.feature_sizes = data['feat_dim']
    args.field_size = len(data['xi'][0])
    args.is_training = True

    with tf.Session(config=gpu_config) as sess:
        Model = FFM(args)
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        cnt = int(len(data['y_train']) / args.batch_size)
        print('time all:%s' % cnt)
        sys.stdout.flush()
        if args.is_train:
            for i in range(args.epoch):
                print('epoch %s:' % i)
                for j in range(0, cnt):
                    X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                    loss, step = Model.train(sess, X_index, X_value, y)
                    if j % 100 == 0:
                        print('the times of training is %d, and the loss is %s' % (j, loss))
                        Model.save(sess, args.checkpoint_dir)
        else:
            Model.restore(sess, args.checkpoint_dir)
            for j in range(0, cnt):
                X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                result = Model.predict(sess, X_index, X_value)
                print(result)