import os

import tensorflow as tf

TENSOR_BOARD_LOGS_PATH = os.path.join(os.environ['HOME'], "tensorflow/logs")


class TensorBoardWrapper:

    def __init__(self, logs_dirs):
        self.logs = logs_dirs

    def dump_graph(self, graph_data):
        with tf.summary.FileWriter(self.logs, graph_data) as f:
            pass


_tb_wrapper = TensorBoardWrapper(TENSOR_BOARD_LOGS_PATH)


def dump_graph_data(graph_data=tf.get_default_graph()):
    _tb_wrapper.dump_graph(graph_data)
