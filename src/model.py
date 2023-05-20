from preprocessing.transform_data import transform
from pyspark.ml.clustering import KMeans
from pyspark.sql import DataFrame


class PySparkKMeans:
    __slots__ = ['_train_df', '_model', '_model_params']

    def __init__(self, raw_train_df: DataFrame, model_params: dict):
        self._train_df = transform(raw_train_df)
        self._model = None
        self._model_params = model_params

    @property
    def train_df(self):
        return self._train_df

    @property
    def model(self):
        return self._model

    @property
    def model_params(self):
        return self._model_params

    def train(self):
        assert self._model is None, "Model already has been trained."

        trainer = KMeans() \
            .setK(self._model_params.get("k", 2)) \
            .setSeed(1) \
            .setMaxIter(self._model_params.get("max_iter", 5))
        self._model = trainer.fit(self.train_df)

    def predict(self, raw_test_df: DataFrame) -> DataFrame:
        assert self._model is not None, "Model is not trained."

        return self._model.transform(raw_test_df)

    def save_model(self, path):
        self._model.save(path)


__all__ = ['PySparkKMeans']
