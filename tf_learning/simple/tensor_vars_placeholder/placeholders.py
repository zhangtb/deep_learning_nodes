import tensorflow as tf
import tf_learning.tf_core.tensorboard_wrapper as tf_wrapper

ph = tf.placeholder(tf.float32)
value = ph * 2

directory = {ph: [[[1, 5, 6], [2, 6, 9], [29, 4, 6]], [[2, 6, 3], [3, 19, 6], [23, 26, 28]],
                  [[100, 23, 21], [123, 232, 22], [12, 34, 22]]]}

with tf.Session() as session:
    result = session.run(value, feed_dict={ph: 3.5})
    tf_wrapper.dump_graph_data(tf.get_default_graph())
    print result

    result = session.run(value, feed_dict=directory)
    tf_wrapper.dump_graph_data()
    print result
