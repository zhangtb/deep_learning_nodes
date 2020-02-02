# -*- coding: utf-8 -*-

from tensorflow.contrib import layers

FEATURE_KEY = 'features'
FEATURE_NAME_KEY = 'feature_name'
VALUE_TYPE_KEY = 'value_type'
MODEL_COLUMNS_KEY = 'model_columns'
MODEL_NAME_KEY = 'model_name'
COLUMNS_KEY = 'columns'


class FeatureColumnGenerator(object):
    def __init__(self, feature_configs):
        self.features = {}
        self.model_columns = {}
        self._parse(feature_configs)

    def _parse(self, feature_configs):
        self.model_columns = [
            {'model_name': 'wide', 'columns': []},
            {'model_name': 'deep', 'columns': []},
        ]
        features = feature_configs[FEATURE_KEY]
        for feature in features:
            self.features[feature[FEATURE_NAME_KEY]] = self.build_feature(feature)
        for feature in features:
            if feature.get('wide_feature'):
                self.model_columns[0]['columns'].append(feature['feature_name'])
            else:
                self.model_columns[1]['columns'].append(feature['feature_name'])

    def build_feature(self, feature):
        feature_name = feature[FEATURE_NAME_KEY]
        #value_type = feature[VALUE_TYPE_KEY]
        if 'hash_bucket_size' in feature:
            id_feature = layers.sparse_column_with_hash_bucket(
                column_name=feature_name,
                hash_bucket_size=10,
                combiner=feature['combiner'] if 'combiner' in feature else 'sum'
            )
            if not 'embedding_dimension' in feature:
                return id_feature
            return layers.embedding_column(
                id_feature,
                dimension=feature['embedding_dimension'],
                combiner=feature['combiner'] if 'combiner' in feature else 'sum'
            )
        else:
            return layers.real_valued_column(column_name=feature_name, default_value=0.0)

    def get_model_columns(self, model_column):
        columns = []
        for model in self.model_columns:
            if model[MODEL_NAME_KEY] == model_column:
                for column in model[COLUMNS_KEY]:
                    columns.append(self.features.get(column))

        return columns
