# -*- coding: UTF-8 -*-
import json

import numpy as np
import tensorflow as tf

import tf_learning.linear_regression.housing_price_prediction.train_data_constant as td_const
from tf_learning.tf_core.gen_feature_column import FeatureColumnGenerator
from tf_learning.tf_core.model_factory import ModuleConfig
from tf_learning.tf_core.model_factory import ModuleFactory

PREDICT_DATA_COUNT = 20


def df_split(df, head_size):
    hd = df.head(len(df) - head_size)
    tl = df.tail(head_size)
    return hd, tl


# linear_data_frame = pd.read_csv(HOUSE_PRICES_TRAIN_DATA, sep=",")
# df_train, df_predict = df_split(linear_data_frame, PREDICT_DATA_COUNT) if len(
#     linear_data_frame) > PREDICT_DATA_COUNT else linear_data_frame, None


df_train = td_const.read_house_train_data()
df_predict = td_const.read_house_predict_data()

print('df_train.describe()=%s\n' % df_train.describe())
print('df_predict.describe()=%s\n' % df_predict.describe())

features_context = '''{
"features":[
    {
        "feature_name":"GrLivArea",
        "feature_type":"id_feature",
        "value_type":"int32",
        "wide_feature":true
    }, 
    {
        "feature_name":"TotalBsmtSF",
        "feature_type":"id_feature",
        "value_type":"int32",
        "wide_feature":true
    },
    {
        "feature_name":"LotArea",
        "feature_type":"id_feature",
        "value_type":"int32",
        "wide_feature":true
    }
]
}
'''

module_config = ModuleConfig()
feature_generator = FeatureColumnGenerator(json.loads(features_context))
module_factory = ModuleFactory("/tmp/feature_test", module_config, "wide", feature_generator)
estimator = module_factory.build_estimator()


def process_feature(data_frame, features):
    return {key: np.array(data_frame[key]) for key in features.keys()}


def csv_file_input_in(data_frame, features, output):
    return tf.contrib.learn.io.numpy_input_fn(process_feature(data_frame, features),
                                              np.array(data_frame[output]),
                                              batch_size=4,
                                              num_epochs=1000)


estimator.fit(input_fn=csv_file_input_in(df_train, feature_generator.features, "SalePrice"),
              steps=module_config.training_steps,
              max_steps=module_config.training_max_steps)

predict_scores = process_feature(df_predict, feature_generator.features)
predict_data = estimator.predict_scores(predict_scores)

# Convert generator to list and print it.
print "predict_scores", list(predict_data)
