import os

import tensorflow as tf

# initialise placeholders and variable
add_placeholder = tf.placeholder(dtype=tf.int32, shape=[], name="add_placeholder")
add_variable = tf.Variable(initial_value=2, name="add_variable")

with tf.name_scope("add"):
    add_result = tf.add(add_variable, add_placeholder)

tf.add_to_collection(name="add_result", value=add_result)
tf.add_to_collection(name="add_placeholder", value=add_placeholder)
session = tf.Session()

# restore train saver
saver = tf.train.import_meta_graph("/tmp/umesh_logs/tensorflow_session/session_saving" + '.model.meta',
                                   clear_devices=True)
saver.restore(sess=session, save_path="/tmp/umesh_logs/tensorflow_session/session_saving" + '.model')
init = tf.variables_initializer(var_list=[add_variable])

# Summary writer
tf.summary.scalar(name="add", tensor=add_result)
writer = tf.summary.FileWriter(logdir="/tmp/umesh_logs/session_restore", graph=tf.get_default_graph())
summary = tf.summary.merge_all()

mul_result = tf.get_collection(key="multiply_result")[0]
multiply_placeholder = tf.get_collection(key="multiply_placeholder")[0]

session.run(init)
for i in range(10):
    res, mul, summ = session.run([add_result, mul_result, summary],
                                 feed_dict={add_placeholder: i, multiply_placeholder: i})
    print(res, mul)
    writer.add_summary(summ)
    saver.save(sess=session, save_path="/tmp/umesh_logs/session_restore/session_restore" + '.model')

# run tensorboard
os.system('tensorboard --logdir=' + "/tmp/umesh_logs/session_restore" + " --port 6008")
