# -*- coding: utf-8 -*-
"""
@created on: 28/08/2017,
@author: Umesh Kumar,
Description:
"""

import tensorflow as tf

task_index = 0
cluster = tf.train.ClusterSpec({"tf_job": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="tf_job", task_index=task_index)

with tf.device("/job:tf_job/task:0"):
    with tf.name_scope('Loss'):
        A = tf.Variable(0, name="Var_A")
        first_elem = tf.assign_add(A, 1, name="add1_A")

with tf.device("/job:tf_job/task:1"):
    with tf.name_scope('Accuracy'):
        B = tf.Variable(100, name="Var_B")
        second_elem = tf.assign_add(B, 1, name="add1_B")

init = tf.global_variables_initializer()
tf.summary.scalar("varA", first_elem)
tf.summary.scalar("varB", second_elem)
summary_op = tf.summary.merge_all()

logs_path = "../graphs/tensorboard_on_device_" + str(task_index)
print(task_index)

with tf.Session(server.target, config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(10):
        input("Press Enter")
        varA, varB, summary = sess.run([first_elem, second_elem, summary_op])
        writer.add_summary(summary, i)
        print([varA, varB])
