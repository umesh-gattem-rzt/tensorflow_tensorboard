import os

import tensorflow as tf

# initialise placeholders and variable
sub_placeholder = tf.placeholder(dtype=tf.int32, shape=[], name="sub_placeholder")
sub_variable = tf.Variable(initial_value=1, name="sub_variable")

with tf.name_scope("sub"):
    result = tf.subtract(sub_variable, sub_placeholder)

session = tf.Session()

# restore train saver
saver = tf.train.import_meta_graph("/tmp/umesh_logs/session_restore/session_restore" + '.model.meta',
                                   clear_devices=True)
saver.restore(sess=session, save_path="/tmp/umesh_logs/session_restore/session_restore" + '.model')
init = tf.variables_initializer([sub_variable])

# Summary writer
tf.summary.scalar(name="sub", tensor=result)
writer = tf.summary.FileWriter(logdir="/tmp/umesh_logs/persist_training", graph=tf.get_default_graph())
summary = tf.summary.merge_all()

mul_result = tf.get_collection(key="multiply_result")[0]
multiply_placeholder = tf.get_collection(key="multiply_placeholder")[0]

add_result = tf.get_collection(key="add_result")[0]
add_placeholder = tf.get_collection(key="add_placeholder")[0]

tf.train.write_graph(session.graph_def, logdir="/tmp/umesh_logs/persist_training", name="persist_train.pb")

session.run(init)
for i in range(10):
    res, mul, summ, add = session.run([result, mul_result, summary, add_result],
                                      feed_dict={add_placeholder: i, multiply_placeholder: i, sub_placeholder: i})
    print(res, mul)
    writer.add_summary(summ)
    saver.save(sess=session, save_path="/tmp/umesh_logs/persist_training/persist_session" + '.model')

# run tensorboard
os.system('tensorboard --logdir=' + "/tmp/umesh_logs/persist_training" + " --port 6010")
