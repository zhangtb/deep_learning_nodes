# -*- coding: utf-8 -*-

import tensorflow as tf

WIDE_MODE = "wide"
DEEP_MODE = "deep"
WIDE_N_DEEP_MODE = "wide_n_deep"


class ModuleConfig(object):

    def __init__(self):
        self.learning_rate = 0.1  # alpha 0.1
        self.learning_rate_power = -0.5
        self.initial_accumulator_value = 1.0  # beta 1.0
        self.l1_regularization_strength = 0.5
        self.l2_regularization_strength = 1.0
        self.training_steps = 1000 * 2000
        self.training_max_steps = None


class ModuleFactory(object):

    def __init__(self, module_dir, module_config, module_type, feature_column_generator):
        self._module_dir = module_dir
        self._module_conf = module_config
        self._module_type = module_type
        self._feature_column_generator = feature_column_generator

    def get_module_type(self):
        return self._module_type

    def get_model_wide_columns(self):
        return self._feature_column_generator.get_model_columns('wide')

    def get_model_deep_columns(self):
        return self._feature_column_generator.get_model_columns('deep')

    def build_estimator(self, run_config=None):
        wide_columns = self.get_model_wide_columns()
        deep_columns = self.get_model_deep_columns()

        if self.get_module_type() == WIDE_MODE:
            m = tf.contrib.learn.LinearRegressor(model_dir=self._module_dir,
                                                 feature_columns=wide_columns,
                                                 config=run_config)
        elif self.get_module_type() == DEEP_MODE:
            m = tf.contrib.learn.DNNClassifier(model_dir=self._module_dir,
                                               feature_columns=deep_columns,
                                               hidden_units=[100, 50],
                                               config=run_config)
        elif self.get_module_type() == WIDE_N_DEEP_MODE:
            m = tf.contrib.learn.DNNLinearCombinedClassifier(
                model_dir=self._module_dir,
                linear_feature_columns=wide_columns,
                dnn_feature_columns=deep_columns,
                dnn_hidden_units=self._module_conf.deep_net,
                linear_optimizer=tf.train.FtrlOptimizer(
                    learning_rate=self._module_conf.learning_rate,
                    learning_rate_power=self._module_conf.learning_rate_power,
                    initial_accumulator_value=self._module_conf.initial_accumulator_value,
                    l1_regularization_strength=self._module_conf.l1_regularization_strength,
                    l2_regularization_strength=self._module_conf.l2_regularization_strength
                ),
                input_layer_min_slice_size=2048 << 20,
                config=run_config)

        else:
            raise Exception('unknown model_type %s' % self.get_module_type())

        return m
