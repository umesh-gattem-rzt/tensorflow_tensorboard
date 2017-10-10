import os

import tensorflow as tf

# initialise placeholders and variable
multiply_placeholder = tf.placeholder(dtype=tf.int32, shape=[], name="multiply_placeholder")
multiply_variable = tf.Variable(initial_value=2, name="multiply_variable")

with tf.name_scope("multiple"):
    multiply_result = tf.multiply(multiply_variable, multiply_placeholder)

init = tf.global_variables_initializer()

# Save the Tensors for future use
tf.add_to_collection(name="multiply_result", value=multiply_result)
tf.add_to_collection(name="multiply_placeholder", value=multiply_placeholder)

# Summary writers
tf.summary.scalar(name="multiply", tensor=multiply_result)
writer = tf.summary.FileWriter(logdir="/tmp/umesh_logs/tensorflow_session", graph=tf.get_default_graph())
summary = tf.summary.merge_all()

# train saver
saver = tf.train.Saver(name="saver")

# Running Session
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        res, summ = sess.run([multiply_result, summary], feed_dict={multiply_placeholder: i})
        print(res)
        writer.add_summary(summ)
        saver.save(sess=sess, save_path="/tmp/umesh_logs/tensorflow_session/session_saving" + '.model')

# Run the tensorboard
os.system('tensorboard --logdir=' + "/tmp/umesh_logs/tensorflow_session")
