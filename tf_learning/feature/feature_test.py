# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

import tf_learning.linear_regression.housing_price_prediction.train_data_constant as td_const
import tf_learning.tf_core.tensorboard_wrapper as wrapper


def test_numeric_column():
    price = {'price': [[1], [2], [3], [4]]}  # 4行样本

    builder = _LazyBuilder(price)

    price_mean = np.mean(np.array(price['price']))
    std = np.std(np.array(price['price']))

    def transform_fn(x):
        return (x - price_mean) / std

    price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)

    price_transformed_tensor = price_column._get_dense_tensor(builder)
    wrapper.dump_op_graph(price_transformed_tensor)

    with tf.Session() as session:
        print(session.run([price_transformed_tensor]))

    '''
    [array([[-1.],
            [ 0.],
            [ 1.],
            [ 2.]], dtype=float32)]
    '''


def test_bucketized_column():
    price = {'price': [[5.], [15.], [25.], [35.]]}  # 4行样本

    builder = _LazyBuilder(price)
    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column, [10, 20, 30, 40])
    price_bucket_tensor = bucket_price._get_dense_tensor(builder)
    wrapper.dump_op_graph(price_bucket_tensor)
    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))

    '''
    [array([[[1., 0., 0., 0., 0.]],
           [[0., 1., 0., 0., 0.]],
           [[0., 0., 1., 0., 0.]],
           [[0., 0., 0., 1., 0.]]], dtype=float32)]
    '''


def test_categorical_column():
    color_data = {'color': [['R'], ['G'], ['B'], ['A'], ['G']]}
    builder = _LazyBuilder(color_data)

    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    color_dense_tensor = color_column._get_sparse_tensors(builder)
    wrapper.dump_op_graph(color_dense_tensor.id_tensor)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_dense_tensor.id_tensor]))

    '''
    [SparseTensorValue(indices=array([[0, 0],
       [1, 0],
       [2, 0],
       [3, 0],
       [4, 0]]), values=array([ 0,  1,  2, -1,  1]), dense_shape=array([5, 1]))] 
    '''


def test_categorical_column_with_multi_hot():
    color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A'], ['G', '']]}
    builder = _LazyBuilder(color_data)

    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    # multi-hot
    color_column_identy = feature_column.indicator_column(color_column)
    color_dense_tensor = color_column_identy._get_dense_tensor(builder)

    wrapper.dump_op_graph(color_dense_tensor)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_dense_tensor]))

    '''
    [array([[2., 0., 0.],
       [1., 1., 0.],
       [0., 1., 1.],
       [0., 0., 0.],
       [0., 1., 0.]], dtype=float32)]
    '''


def test_hash_column():
    color_data = {'color': [[2], [5], [-1], [0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_hash_bucket('color', 10, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    wrapper.dump_op_graph(color_column_tensor.id_tensor)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    '''
    [SparseTensorValue(indices=array([[0, 0],
       [1, 0],
       [3, 0]]), values=array([9, 1, 5]), dense_shape=array([4, 1]))]
    '''


def test_crossed_column():
    df_train = td_const.read_house_predict_data()
    hource_price_featrues = [
        "LotShape",
        "LotConfig",
        # "RoofStyle",
        # "Exterior2nd",
        # "Foundation"
    ]

    featues_list = []
    feature_inputs = {}
    for feature_name in hource_price_featrues:
        feature_value = df_train[feature_name].values
        featues_list.append(
            feature_column.categorical_column_with_vocabulary_list(feature_name, np.unique(feature_value)))
        feature_inputs[feature_name] = feature_value.reshape(-1, 1)

    builder = _LazyBuilder(feature_inputs)
    cross_column = feature_column.crossed_column(featues_list, 32)
    cross_identy_dense_tensor = cross_column._get_sparse_tensors(builder)
    wrapper.dump_op_graph(cross_identy_dense_tensor.id_tensor)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([cross_identy_dense_tensor.id_tensor]))

    '''
    [SparseTensorValue(indices=array([[ 0,  0],
       [ 1,  0],
       [ 2,  0],
       [ 3,  0],
       [ 4,  0],
       [ 5,  0],
       [ 6,  0],
       [ 7,  0],
       [ 8,  0],
       [ 9,  0],
       [10,  0],
       [11,  0],
       [12,  0],
       [13,  0],
       [14,  0],
       [15,  0],
       [16,  0],
       [17,  0],
       [18,  0],
       [19,  0]]), values=array([24, 27,  4,  8, 14,  4, 24,  8, 24,  9, 24,  4,  3,  4,  8,  9,  3,
       24, 24, 24]), dense_shape=array([20,  1]))]
    '''


def test_weighted_column():
    color_data = {'color': [['R'], ['G'], ['B'], ['A']],
                  'weight': [[1.0], [2.0], [4.0], [8.0]]}  # 4行样本
    builder = _LazyBuilder(color_data)

    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_weight_categorical_column = feature_column.weighted_categorical_column(color_column, 'weight')
    id_tensor, weight = color_weight_categorical_column._get_sparse_tensors(builder)

    wrapper.dump_op_graph(id_tensor)
    wrapper.dump_op_graph(weight)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('weighted categorical' + '-' * 40)
        print(session.run([id_tensor]))
        print('-' * 40)
        print(session.run([weight]))

    '''
    weighted categorical----------------------------------------
    [SparseTensorValue(indices=array([[0, 0],
       [1, 0],
       [2, 0],
       [3, 0]]), values=array([ 0,  1,  2, -1]), dense_shape=array([4, 1]))]
    ----------------------------------------
    [SparseTensorValue(indices=array([[0, 0],
       [1, 0],
       [2, 0],
       [3, 0]]), values=array([1., 2., 4., 8.], dtype=float32), dense_shape=array([4, 1]))]
    '''


def test_embedding():
    df_train = td_const.read_house_predict_data()
    hource_price_featrues = [
        "LotShape",
        "LotConfig",
        # "RoofStyle",
        # "Exterior2nd",
        # "Foundation"
    ]

    featues_list = []
    feature_inputs = {}
    for feature_name in hource_price_featrues:
        feature_value = df_train[feature_name].values
        featues_list.append(
            feature_column.categorical_column_with_hash_bucket(feature_name, 7))
        feature_inputs[feature_name] = feature_value.reshape(5, 4)

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_embed = feature_column.shared_embedding_columns(featues_list, 4, combiner='sum')
    # color_column_embed = feature_column.embedding_column(featues_list, 4, combiner='sum')

    # price_transformed_tensor = color_column_embed._get_dense_tensor(builder)
    color_dense_tensor = feature_column.input_layer(feature_inputs, color_column_embed)
    wrapper.dump_op_graph(color_dense_tensor)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run(color_dense_tensor))

    '''
    use input_layer________________________________________
    [[-0.9249029   1.3081536  -0.88930225  0.56293714 -1.1476531   2.2818325
        0.26385954 -1.1767051 ]
     [-0.9249029   1.3081536  -0.88930225  0.56293714 -0.7241223   1.6956066
        0.62426734 -1.4107678 ]
     [-1.4598088   2.381219   -0.6731292  -0.07282123 -1.5711839   2.8680582
        -0.09654826 -0.94264233]
     [-0.9249029   1.3081536  -0.88930225  0.56293714 -0.7701344   1.6746099
        0.9991817  -1.8514115 ]
     [-1.4598088   2.381219   -0.6731292  -0.07282123 -1.5711839   2.8680582
        -0.09654827 -0.94264233]]
    '''


# run
# test_numeric_column()
# test_bucketized_column()
# test_categorical_column()
# test_categorical_column_with_multi_hot()
# test_hash_column()
# test_crossed_column()
# test_weighted_column()
test_embedding()
