# -*- coding: utf-8 -*-
"""
@created on: 28/08/2017,
@author: Umesh Kumar,
Description:
"""

import tensorflow as tf
from rztdl.utils.file import read_csv

learning_rate = 0.01
epochs = 20
batch_size = 20
display_step = 1
logs_path = '../graphs/simple_tensorboard_logs'

data_path = "../data/mnist_dataset.csv"
train_data, train_label, test_data, test_label = read_csv(data_path, split_ratio=[80, 0, 20], delimiter=";",
                                                          normalize=False, randomize=True, label_vector=True)
x = tf.placeholder(tf.float32, [None, 784], name='input')
y = tf.placeholder(tf.float32, [None, 10], name='output')

# Set model weights and bias
W = tf.Variable(tf.zeros([784, 10]), name='weights')
b = tf.Variable(tf.zeros([10]), name='bias')

# model building
with tf.name_scope('Model'):
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(epochs):
        sess.run(optimizer, feed_dict={x: train_data, y: train_label})
        if i % display_step == 0:
            loss, accuracy, summary = sess.run([cost, acc, merged_summary_op],
                                               feed_dict={x: train_data, y: train_label})
            summary_writer.add_summary(summary, i)
            print("epoch ", i, " Loss= ", loss, " Accuracy= ", accuracy)
    print("Testing Accuracy:", sess.run(acc, feed_dict={x: test_data, y: test_label}))

# Run Tensorboard
print("Run the command line:\n", "tensorboard --logdir=" + logs_path +
      "\nThen open http://0.0.0.0:6006/ into your web browser")
