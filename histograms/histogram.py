import tensorflow as tf

weights = tf.ones(shape=[1000])

tf.summary.histogram("weights", weights)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("../graphs/histogram")
    for i in range(10):
        sess.run(tf.global_variables_initializer())
        _, summary = sess.run([weights, summary_op])
        writer.add_summary(summary, i)
