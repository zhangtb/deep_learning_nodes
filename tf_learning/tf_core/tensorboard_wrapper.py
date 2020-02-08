import os

import tensorflow as tf

TENSOR_BOARD_LOGS_PATH = os.path.join(os.environ['HOME'], "tensorflow/logs")


class TensorBoardWrapper:

    def __init__(self, logs_dirs):
        self.logs = logs_dirs

    def dump_graph(self, graph_data, filename_suffix):
        with tf.summary.FileWriter(self.logs, graph_data, filename_suffix=filename_suffix) as f:
            pass


_tb_wrapper = TensorBoardWrapper(TENSOR_BOARD_LOGS_PATH)


def dump_graph_data(graph_data=tf.get_default_graph(), filename_suffix=None):
    _tb_wrapper.dump_graph(graph_data, filename_suffix)


def dump_op_graph(t, filename_suffix=None):
    if not isinstance(t, (tf.Operation, tf.Tensor, tf.SparseTensor)):
        raise TypeError("t needs to be an Operation or Tensor: %s" % t)
    _tb_wrapper.dump_graph(t.graph, filename_suffix)
