# -*- coding: utf-8 -*-

import numpy as np


# feature 归一化
def feature_normal(feature):
    return (feature - np.mean(feature)) / np.std(feature)
