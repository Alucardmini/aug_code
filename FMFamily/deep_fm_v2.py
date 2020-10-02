
import tensorflow as tf
from FMFamily.data_utils import load_data
import sys
import numpy as np


class ModelArgs(object):

    def __init__(self):
        self.feature_size = 30
        self.feature_embed_size = 256
        self.field_size = 30

        self.learn_rate = 0.05
        self.epochs = 3
        self.batch_size = 64
        self.l2_reg_rate = 0.01
        self.deep_layers = [512, 256, 128]


        self.is_training = False


class DeepFM(object):

    def __init__(self, args: ModelArgs):
        super(DeepFM, self).__init__()
        self.field_size = args.field_size
        self.feature_size = args.feature_size
        self.feature_embed_size = args.feature_embed_size

        self.deep_activation = tf.nn.relu
        self.l2_reg_rate = args.l2_reg_rate
        self.deep_layers = args.deep_layers
        self.learn_rate = args.learn_rate
        self.weights = dict()
        self.build_model()

    def build_model(self):

        self.feature_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name="feature_index")
        self.feature_value = tf.placeholder(dtype=tf.float32, shape=[None, None], name="feature_value")
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, None], name="label")

        random_normal_initializer = tf.random_normal_initializer(0.0, 0.01)
        # 第一个因子
        self.weights["first_factor_weights"] = tf.get_variable("first_factor_weights", shape=[self.feature_size, 1],
                                                               initializer=random_normal_initializer)
        self.fm_first_factor = tf.multiply(
            tf.nn.embedding_lookup(self.weights["first_factor_weights"], self.feature_index),
            tf.reshape(self.feature_value, [-1, self.field_size, 1])
        )

        self.fm_first_factor = tf.reduce_sum(self.fm_first_factor, axis=-1)

        self.weights["feature_embeding"] = tf.get_variable("feature_embeding",
                                                           shape=[self.feature_size, self.feature_embed_size],
                                                           initializer=random_normal_initializer)

        embed_part = tf.nn.embedding_lookup(self.weights["feature_embeding"], self.feature_index)

        # 第二个因子
        tmp = tf.reshape(self.feature_value, [-1, self.field_size, 1])

        self.fm_second_factor = 0.5 * tf.subtract(tf.square(tf.reduce_sum(tf.multiply(embed_part, tmp), axis=1)),
            tf.reduce_sum(tf.square(tf.multiply(embed_part, tmp)), axis=1))

        deep_input_size = self.field_size * self.feature_embed_size

        self.weights['layer_0'] = tf.get_variable("layer_0", shape=[deep_input_size, self.deep_layers[0]],
                                                  initializer=random_normal_initializer)

        for i in range(1, len(self.deep_layers)):
            self.weights['layer_%s' % i] = tf.get_variable("layer_" + str(i), shape=[self.deep_layers[i-1], self.deep_layers[i]],
                                                      initializer=random_normal_initializer)

        deep_input = tf.reshape(embed_part, [-1, deep_input_size])
        self.deep_part = self.deep_activation(tf.matmul(deep_input, self.weights['layer_0']))
        for i in range(1, len(self.deep_layers)):

            self.deep_part = self.deep_activation(
                tf.matmul(self.deep_part, self.weights['layer_%s' % i]))

        merge_layer_size = self.field_size + self.feature_embed_size + self.deep_layers[-1]

        self.weights["merge_layer"] = tf.get_variable(
            "merge_layer", shape=[merge_layer_size, 1], initializer=random_normal_initializer
        )

        # self.weights["merge_bias"] = tf.get_variable(
        #      name="merge_bias", shape=[1], initializer=random_normal_initializer
        # )
        self.weights["merge_bias"] = tf.Variable(
            tf.constant(0.01), dtype=tf.float32, name="merge_bias"
        )
        self.out_put = tf.concat([self.fm_first_factor, self.fm_second_factor, self.deep_part], axis=1)

        self.out = tf.add(tf.matmul(self.out_put, self.weights["merge_layer"]), self.weights["merge_bias"])

        self.out = tf.nn.sigmoid(self.out)

        self.loss = -tf.reduce_mean(self.label * tf.log(self.out + 1e-24) + (1-self.label) * tf.log(1-self.out + 1e-24))

        # opt = tf.train.GradientDescentOptimizer(self.learn_rate)
        # self.train_op = opt.minimize(self.loss)

        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["merge_layer"])

        for i in range(len(self.deep_layers)):
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["layer_%d" % i])

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learn_rate)
        trainable_params = tf.trainable_variables()

        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, feat_index, feat_value, label):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.feature_index: feat_index,
            self.feature_value: feat_value,
            self.label: label
        })
        return loss


def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], np.array(y[start:end])

if __name__ == '__main__':
    args = ModelArgs()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    data = load_data()
    args.feature_size = data['feat_dim']
    args.field_size = len(data['xi'][0])
    args.is_training = True

    with tf.Session(config=gpu_config) as sess:
        Model = DeepFM(args)
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        cnt = int(len(data['y_train']) / args.batch_size)
        print('time all:%s' % cnt)
        sys.stdout.flush()
        if args.is_training:
            for i in range(args.epochs):
                print('epoch %s:' % i)
                for j in range(0, cnt):
                    X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                    loss = Model.train(sess, X_index, X_value, y)
                    if j % 100 == 0:
                        print('the times of training is %d, and the loss is %s' % (j, loss))
                        # Model.save(sess, args.checkpoint_dir)















