# -*- coding: UTF-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_core.python.ops import gradients

import tf_learning.linear_regression.housing_price_prediction.train_data_constant as td_const
import tf_learning.tf_core.tensorboard_wrapper as wrapper
from tf_learning.tf_core.plotting.plot_wrapper import GradualPlot

# plt.rcParams['figure.figsize'] = (10, 6)
#
# X = np.arange(0.0, 5.0, 0.1)
#
# a = 1
# b = 0
#
# Y = a * X + b

# plt.plot(X, Y)
# plt.xlabel("Indepdendent Variable")
# plt.ylabel("Dependent Variable")
# plt.show()

# line regression : Y = 3.0 * X + 2.0
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data * 3.0 + 2.0
#
# # add random gaussian noise
# y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

# zip(x_data, y_data) [0:5]

df_train = td_const.read_house_train_data()
x_data = df_train['GrLivArea'].values.reshape(-1, 1)
x1_data = df_train['TotalBsmtSF'].values.reshape(-1, 1)
x2_data = df_train['LotArea'].values.reshape(-1, 1)
y_data = df_train['SalePrice'].values.reshape(-1, 1)

print np.mean()

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name="weight", dtype=tf.float32)
W2 = tf.Variable(np.random.randn(), name="weight2", dtype=tf.float32)
b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)

# value = tf.add(W * X, b, W2 * X ^2 )
# value = tf.add(tf.multiply(W, X), b)
value2 = tf.add(tf.add(tf.multiply(W2, tf.square(X)), tf.multiply(W, X)), b)


# value2_pre = lambda x,f: [w1, w2, b] = f,
def value2_pre(x, f):
    [w1, w2, b] = f
    return math.sqrt(x) * w2 + w1 * x + b;


# plt.scatter(x_data, y_data)
# plt.xlabel("Indepdendent Variable")
# plt.ylabel("Dependent Variable")
# plt.show()

loss = tf.reduce_mean(tf.square(value2 - Y))
wrapper.dump_op_graph(loss, "price_prediction_loss")

t_grad = gradients.gradients([loss], [W, W2, b])
for t in t_grad:
    wrapper.dump_op_graph(t, "____aaa___")

# optimizer = tf.train.GradientDescentOptimizer(0.5)
# AdamOptimizer 是SDG的实现
optimizer = tf.train.AdamOptimizer(0.5)
# minimize 等效于下面两个代码，主要功能如下
# step1：计算梯度，返回（当前梯度值，当前计算出的变量值）
# step2：用返回的变量值更新对于的变量
# grads_and_vars = optimizer.compute_gradients(loss)
# train_op = optimizer.apply_gradients(grads_and_vars)
train = optimizer.minimize(loss)

g_plot = GradualPlot()
init = tf.initialize_all_variables()
loss_x = []
loss_y = []
with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        for (x, y) in zip(x_data, y_data):
            sess.run(train, feed_dict={X: x, Y: y})

        wrapper.dump_graph_data()
        if step % 10 == 0 or step == 99:
            c = sess.run(loss, feed_dict={X: x_data, Y: y_data})
            w1 = sess.run(W)
            w2 = sess.run(W2)
            b1 = sess.run(b)
            print step, c, w1, w2, b1
            g_plot.add_train_data([w1, w2, b1])
            loss_x.append(step)
            loss_y.append(c)

    plt.figure()
    plt.xlabel(" ")
    plt.ylabel("loss")
    plt.plot(loss_x, loss_y)

    g_plot.drawing(x_data, y_data, predict=value2_pre)
