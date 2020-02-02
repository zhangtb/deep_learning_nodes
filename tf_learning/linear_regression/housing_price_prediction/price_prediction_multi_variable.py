# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tf_learning.linear_regression.housing_price_prediction.train_data_constant as td_const
import tf_learning.tf_core.feature_wrapper as fw
import tf_learning.tf_core.tensorboard_wrapper as wrapper
from tf_learning.tf_core.plotting.plot_wrapper import GradualPlot

FEATURE_COUNT = 3

df_train = td_const.read_house_train_data()
x_data = df_train['GrLivArea'].values
y_data = df_train['SalePrice'].values

x2_data = np.square(x_data)
x_data = fw.feature_normal(x_data)

x2_data = fw.feature_normal(x2_data)
y_data = fw.feature_normal(y_data)
X_DATA = np.c_[np.ones(len(x_data)), x_data, x2_data]
X_DATA = X_DATA.transpose()

X = tf.placeholder(tf.float32, [FEATURE_COUNT, None])
Y = tf.placeholder(tf.float32, [None])

W = tf.Variable(np.random.randn(1, FEATURE_COUNT), name="weight", dtype=tf.float32)
value = tf.matmul(W, X)


def value2_pre(index, x, f):
    return np.matmul(f, X_DATA[:, index])[0]


loss = tf.reduce_mean(tf.square(value - Y))
wrapper.dump_op_graph(loss, "price_prediction_loss")

optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

g_plot = GradualPlot()
init = tf.initialize_all_variables()
loss_x = []
loss_y = []
with tf.Session() as sess:
    sess.run(init)

    for step in range(4000):
        sess.run(train, feed_dict={X: X_DATA, Y: y_data})

        wrapper.dump_graph_data()
        if step % 10 == 0 or step == 1999:
            c = sess.run(loss, feed_dict={X: X_DATA, Y: y_data})
            w1 = sess.run(W)
            print step, c, w1
            g_plot.add_train_data(w1)
            loss_x.append(step)
            loss_y.append(c)

    plt.figure()
    plt.xlabel(" ")
    plt.ylabel("loss")
    plt.plot(loss_x, loss_y)

    g_plot.drawing(x_data, y_data, predict=value2_pre)
