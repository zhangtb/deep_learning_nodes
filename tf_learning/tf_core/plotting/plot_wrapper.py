# -*- coding: UTF-8 -*-
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def predict_fun(x, p):
    [a, b] = p
    return a * x + b


class GradualPlot(object):

    def __init__(self):
        self._train_data = []

    def add_train_data(self, data):
        self._train_data.append(data)

    def drawing(self, x_data, y_data, predict=predict_fun):
        cr, cg, cb = (1.0, 1.0, 0.0)
        train_len = len(self._train_data)
        plt.figure()
        for f in self._train_data:
            cb += 1.0 / train_len
            cg -= 1.0 / train_len
            if cb > 1.0: cb = 1.0
            if cg < 0.0: cg = 0.0
            f_y = np.vectorize(lambda x: predict(x, x_data[x], f))(range(len(x_data)))
            line = plt.plot(x_data, f_y)
            plt.setp(line, color=(cr, cg, cb))

        plt.plot(x_data, y_data, 'ro')

        green_line = mpatches.Patch(color='red', label='Data Points')
        plt.legend(handles=[green_line])
        plt.show()
