# -*- coding: utf-8 -*-
"""
@created on: 29/08/2017,
@author: Umesh Kumar,
Description:
"""

import tensorflow as tf

from rztdl.utils.file import read_csv

datapath = "../data/mnist_dataset.csv"
train_data, train_label, test_data, test_label = read_csv(datapath, split_ratio=[80, 0, 20],
                                                          delimiter=";", label_vector=True)

# Parameters
learning_rate = 0.01
epoch = 100
display_step = 10
logs_path = "../graphs/simple_mnist_train"

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([448, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([10]))
}

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


def model(x, dropout):
    layer1 = tf.reshape(x, shape=[-1, 28, 28, 1])
    layer2 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(layer1, weights['wc1'], strides=[1, 4, 1, 1], padding='SAME'), biases['bc1']))
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer4 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(layer3, weights['wc2'], strides=[1, 4, 1, 1], padding='SAME'), biases['bc2']))
    layer5 = tf.nn.max_pool(layer4, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    layer6 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(layer1, weights['wc1'], strides=[1, 1, 4, 1], padding='SAME'), biases['bc1']))
    layer7 = tf.nn.max_pool(layer6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer8 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(layer7, weights['wc2'], strides=[1, 1, 4, 1], padding='SAME'), biases['bc2']))
    layer9 = tf.nn.max_pool(layer8, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
    layer10 = tf.reshape(layer9, shape=[-1, 1, 7, 64])

    layer11 = tf.add(layer5, layer10)
    layer12 = tf.reshape(layer11, [-1, 448])
    layer13 = tf.nn.relu(tf.add(tf.matmul(layer12, weights['wd1']), biases['bd1']))
    layer13 = tf.nn.dropout(layer13, dropout)
    layer14 = tf.add(tf.matmul(layer13, weights['out']), biases['out'])
    return layer14


pred = model(x, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(epoch):
        sess.run(optimizer, feed_dict={x: train_data, y: train_label, keep_prob: 0.75})
        if i % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: train_data, y: train_label, keep_prob: 0.75})
            print("Epochs ", i, "Loss= ", loss, " Training Accuracy= ", acc)
    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: train_data, y: train_label, keep_prob: 0.0}))
