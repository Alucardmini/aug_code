# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/14 10:48 PM'

import tensorflow as tf
import numpy as np
path = "./tf_record"
writer = tf.python_io.TFRecordWriter(path)

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