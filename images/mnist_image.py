# -*- coding: utf-8 -*-
"""
@created on: 30/08/2017,
@author: Umesh Kumar,
Description:
"""

import tensorflow as tf

from rztdl.utils.file import read_csv
import numpy as np

datapath = "../data/mnist_dataset.csv"
train_data, train_label, test_data, test_label = read_csv(datapath, split_ratio=[80, 0, 20],
                                                          delimiter=";", label_vector=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

logs_path = "../graphs/mnist_image"

init = tf.global_variables_initializer()

k = tf.reshape(x, shape=[-1, 28, 28, 1])
tf.summary.image("input", k)


summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(10):
        _, summary = sess.run([k, summary_op], feed_dict={x: train_data})
        writer.add_summary(summary, i)
