# Standard Imports
from typing import List

# Third Party Imports
import numpy as np

# Internal Imports


class AbstractEmbeddingModel:

    def load(self, model_path):
        raise NotImplementedError

    def predict(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError
