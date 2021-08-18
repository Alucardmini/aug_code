# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/14 10:48 PM'

import tensorflow as tf
import numpy as np
from PIL import Image



def write():
    path = "./tf_record"
    tf.python_io.TFRecordWriter(path)

    tfrecords_filename = 'train.tfrecords'
    write = tf.python_io.TFRecordWriter(tfrecords_filename)

    for i in range(100):

      img_raw = np.random.random_integers(0,255,size=(30,30))  # Create image size 30*30

      img_raw = img_raw.tostring()  # float data to bytes

      example = tf.train.Example(features=tf.train.Features(

          feature = {

              'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),

              'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))

          }

      ))

      write.write(example.SerializeToString())

    write.close()

def read():
    input_files = ['/Users/wuxikun/Documents/LetsAug/tf_record_lr/train.tfrecords']

    dataset = tf.data.TFRecordDataset(input_files)

    def parser(record):
        features = tf.parse_single_example(
            record,
            features={  # 和原来生成tfrecord时候要对应相同的
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)

            }
        )
        return features["label"], features['img_raw']

    dataset = dataset.map(parser)  # 接受的参数是一个函数

    iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


if __name__ == '__main__':
    iterator, next_elem = read()

    with tf.Session() as sess:

        sess.run(iterator.initializer)

        for i in range(10):

            image = tf.decode_raw(next_elem[1], tf.uint8)
            print(sess.run(next_elem[0]))
            print(sess.run(image))
