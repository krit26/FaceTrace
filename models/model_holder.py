# Standard Imports
import time
from typing import Union

# Third Part Imports

# Internal Imports
from models.detectors.mtcnn import FastMtcnn
from models.embeddings.facenet import FaceNet512
from models.detectors import AbstractDetectionModel
from models.embeddings import AbstractEmbeddingModel


class ModelHolder(object):
    _model_holder, _last_load_time = {}, {}

    @staticmethod
    def get_or_load_model(model_name, load=True, model_path=None, **kwargs):
        if model_name not in ModelHolder._model_holder:
            if model_name not in globals():
                raise Exception(f"{model_name} model does not exists")
            ModelHolder._model_holder[model_name]: Union[
                AbstractDetectionModel, AbstractEmbeddingModel
            ] = globals()[model_name](**kwargs)

            if load:
                ModelHolder._load_model(model_name, model_path)

        return ModelHolder._model_holder[model_name]

    @staticmethod
    def _load_model(model_name, model_path):
        if model_name not in ModelHolder._last_load_time:
            ModelHolder._last_load_time[model_name] = time.time()
            ModelHolder._model_holder[model_name].load(model_path)
