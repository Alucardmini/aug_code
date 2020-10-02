# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/9/24 7:07 PM'
import tensorflow as tf
import numpy as np
from FMFamily.data_utils import load_data
import sys


class ModelArgs(object):
    feat_size = 100
    embed_size = 100
    deep_layers = [512, 256, 128]
    field_size = 100
    epoch = 3
    batch_size = 64
    learning_rate = 0.05
    is_train = False
    l2_reg_rate = 0.01
    checkpoint_dir = '../model'


class DeepFM(object):

    def __init__(self, args: ModelArgs):
        super(DeepFM, self).__init__()
        self.feat_embed_dim = args.embed_size
        self.deep_layers = args.deep_layers
        self.l2_reg_rate = args.l2_reg_rate
        self.learning_rate = args.learning_rate
        self.feature_size = args.feat_size
        self.field_size = args.field_size
        self.deep_activation = tf.nn.relu

        self.weights = dict()
        self.build_model()

    def build_model(self):

        self.feat_index = tf.placeholder(tf.int32, [None, None], name="feat_index")
        self.feat_value = tf.placeholder(tf.float32, [None, None], name="feat_value")
        self.label = tf.placeholder(tf.float32, [None, None], name="label")

        #  x*w
        self.weights["first_factor_weight"] = tf.Variable(tf.random_normal(shape=[self.feature_size, 1],
                                                                              mean=0.0, stddev=0.01))

        self.weights["first_factor_weight"] = tf.get_variable(name="first_factor_weight", shape=[self.feature_size, 1], initializer=tf.random_normal_initializer(0.0, 0.01))

        self.first_factor = tf.multiply(tf.nn.embedding_lookup(self.weights["first_factor_weight"], self.feat_index),
                                        tf.reshape(self.feat_value, [-1, self.field_size, 1]))

        self.first_factor = tf.reduce_sum(self.first_factor, axis=2)

        self.weights["embedding_weight"] = tf.Variable(tf.random_normal([self.feature_size, self.feat_embed_dim],
                                                                        0.0, 0.01), name="embedding_weight")

        self.embed_part_weight = tf.nn.embedding_lookup(self.weights["embedding_weight"], self.feat_index)

        tmp = tf.reshape(self.feat_value, [-1, self.field_size, 1])
        self.embed_part = tf.multiply(self.embed_part_weight, tmp)

        self.second_factor_sum_square = tf.square(tf.reduce_sum(self.embed_part, 1))
        self.second_factor_square_sum = tf.reduce_sum(tf.square(self.embed_part), 1)

        self.second_factor = 0.5 * tf.subtract(self.second_factor_sum_square, self.second_factor_square_sum)

        deep_input_size = self.feat_embed_dim * self.field_size

        init_value = np.sqrt(2.0 / (deep_input_size + self.deep_layers[0]))

        self.weights["layer_0"] = tf.Variable(
            tf.random_normal([self.feat_embed_dim * self.field_size, self.deep_layers[0]], 0.0, init_value), name="layer_0"
        )

        self.weights["bias_0"] = tf.Variable(
            tf.random_normal([1, self.deep_layers[0]], 0.0, init_value), name="bias_0"
        )

        cnt_hidden_layer = len(self.deep_layers)

        if cnt_hidden_layer > 0:
            for i in range(1, cnt_hidden_layer):
                init_value = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))

                self.weights["layer_" + str(i)] = tf.Variable(
                    tf.random_normal([self.deep_layers[i-1], self.deep_layers[i]], 0.0, init_value),
                    name="layer_" + str(i))

                self.weights["bias_" + str(i)] = tf.Variable(
                    tf.random_normal([1, self.deep_layers[i]], 0.0, init_value), name="bias_" + str(i))

        self.deep_embedding = tf.reshape(self.embed_part, [-1, deep_input_size])

        for i in range(0, cnt_hidden_layer):
            self.deep_embedding = tf.add(tf.matmul(self.deep_embedding, self.weights["layer_" + str(i)]),
                                         self.weights["bias_" + str(i)])
            self.deep_embedding = self.deep_activation(self.deep_embedding)

        merge_layer_size = self.field_size + self.feat_embed_dim + self.deep_layers[-1]


        init_value = np.sqrt(np.sqrt(2.0 / (merge_layer_size + 1)))

        self.weights["merge_layer"] = tf.Variable(
            tf.random_normal([merge_layer_size, 1], 0, init_value), name="merge_layer"
        )

        self.weights["merge_bias"] = tf.Variable(
            tf.constant(0.01), dtype=tf.float32, name="merge_bias"
        )

        self.fm_part = tf.concat([self.first_factor, self.second_factor], axis=1)

        self.out = tf.add(tf.matmul(tf.concat([self.fm_part, self.deep_embedding], axis=1),
                                    self.weights["merge_layer"]), self.weights["merge_bias"])

        self.out = tf.nn.sigmoid(self.out)

        # self.loss = -tf.reduce_mean(self.label * tf.log(self.out + 1e-24) +
        #                             (1 - self.label) * tf.log(1 - self.out + 1e-24))
        #
        # self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["merge_layer"])
        #
        # for i in range(len(self.deep_layers)):
        #     self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["layer_" + str(i)])
        #
        # self.global_step = tf.Variable(0, trainable=False)
        # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # trainable_params = tf.trainable_variables()
        #
        # gradients = tf.gradients(self.loss, trainable_params)
        # clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        #
        # self.train_op = opt.apply_gradients(
        #     zip(clip_gradients, trainable_params), global_step=self.global_step)

        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["merge_layer"])

        for i in range(len(self.deep_layers)):
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["layer_%d" % i])

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
        tf.train.Saver.save(sess, save_path=path)

    def restore(self, sess, path):
        tf.train.Saver.restore(sess, save_path=path)


def get_batch(xi, xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)

    return xi[start: end], xv[start: end], np.array(y[start: end])


if __name__ == "__main__":

    args = ModelArgs()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    data = load_data()
    args.feat_size = data['feat_dim']
    args.field_size = len(data['xi'][0])
    args.is_train = True

    with tf.Session(config=gpu_config) as sess:
        model = DeepFM(args)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        cnt = int(len(data['y_train']) / args.batch_size)
        print('time all %s' % cnt)
        sys.stdout.flush()

        if args.is_train:
            for i in range(args.epoch):
                print("epoch %s:" % i)
                for j in range(0, cnt):
                    x_index, x_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                    loss, step = model.train(sess, x_index, x_value, y)

                    if j % 100 == 0:
                        print("epoch %s step %s, loss %s" % (i, j, loss))

