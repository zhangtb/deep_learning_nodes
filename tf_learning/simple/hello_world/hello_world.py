# -*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a, b)

# session = tf.Session()
# result = session.run(c)
# print(result)
# session.close()

with tf.Session() as session:
    result = session.run(c)
    print result


