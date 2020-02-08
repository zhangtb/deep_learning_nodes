# -*- coding: utf-8 -*-

import tensorflow as tf

tensor_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]

with tf.Session() as sess:
    y1 = tf.train.batch_join([tensor_list], batch_size=2, enqueue_many=True)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("y1 batch:" + "-" * 10)

    print(sess.run(y1))


    print("-" * 10)

    coord.request_stop()

    coord.join(threads)
