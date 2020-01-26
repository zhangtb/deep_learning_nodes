import tensorflow as tf

import tf_learning.tf_core.tensorboard_wrapper as tf_wrapper

state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))

    tf_wrapper.dump_graph_data()

    for _ in range(3):
        session.run(update)
        print(session.run(state))
        tf_wrapper.dump_graph_data()
