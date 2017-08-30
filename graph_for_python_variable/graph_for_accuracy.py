# -*- coding: utf-8 -*-
"""
@created on: 29/08/2017,
@author: Umesh Kumar,
Description:
"""

import tensorflow as tf
from rztdl.utils.file import read_csv

data_path = "../data/energy.csv"
train_data, train_label, test_data, test_label = read_csv(data_path, split_ratio=[80, 0, 20], delimiter=",",
                                                          output_label=True)
# parameters
learning_rate = 0.001
epochs = 5000
display_step = 100
drop_out = 0.75
log_path = "../graphs/graph_for_accuracy"

input_data = tf.placeholder(tf.float32, [None, 8], name='Input')
output_data = tf.placeholder(tf.float32, [None, 1], name='Output')
accuracy_placeholder = tf.placeholder(tf.float32, shape=[], name='accuracy')
accuracy_tensor = tf.Variable(initial_value=0, dtype=tf.float32)
keep_prob = tf.placeholder(tf.float32)

weights = {
    'weight1': tf.Variable(tf.random_normal([8, 32])),
    'weight2': tf.Variable(tf.random_normal([32, 64])),
    'weight3': tf.Variable(tf.random_normal([64, 10])),
    'weight4': tf.Variable(tf.random_normal([10, 8])),
    'weight5': tf.Variable(tf.random_normal([8, 1]))
}

bias = {
    'bias1': tf.Variable(tf.random_normal([32])),
    'bias2': tf.Variable(tf.random_normal([64])),
    'bias3': tf.Variable(tf.random_normal([10])),
    'bias4': tf.Variable(tf.random_normal([8])),
    'bias5': tf.Variable(tf.random_normal([1]))
}


def model(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weight1']), bias['bias1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2']))
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['weight3']), bias['bias3']))
    layer4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer3, weights['weight4']), bias['bias4'])), drop_out)
    layer5 = tf.add(tf.matmul(layer4, weights['weight5']), bias['bias5'])
    return layer5


with tf.name_scope("model"):
    pred = model(input_data)
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.square(output_data - pred))
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("accuracy"):
    def accuracy(pred, actual, threshold=5):
        acc = 0
        for i, j in list(zip(pred, actual)):
            i, j = int(i[0]), int(j[0])
            if i in range(j - threshold, j + threshold):
                acc += 1
        return (acc / len(pred)) * 100

init = tf.global_variables_initializer()
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy_tensor)
accuracy_tensor = tf.assign(ref=accuracy_tensor, value=accuracy_placeholder)

summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logdir=log_path, graph=tf.get_default_graph())
    for each_epoch in range(epochs):
        # Run optimization op (backprop)
        _ = sess.run(optimizer, feed_dict={input_data: train_data, output_data: train_label})
        if each_epoch % display_step == 0:
            loss, p, summary = sess.run([cost, pred, summary_op],
                                        feed_dict={input_data: train_data, output_data: train_label})
            acc = accuracy(p.tolist(), train_label)
            sess.run(accuracy_tensor, feed_dict={accuracy_placeholder: acc})
            writer.add_summary(summary, each_epoch)
            print("Epoch =", each_epoch, ", Loss= ", loss, ", Training Accuracy= ", acc)
    print("Optimization Finished!")
    p = sess.run(pred, feed_dict={input_data: test_data, output_data: test_label})
    acc = accuracy(p.tolist(), test_label)
    print("Testing Accuracy :", acc)
