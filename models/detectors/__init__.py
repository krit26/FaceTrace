# Standard Imports
from typing import List

# Third Party Imports
import numpy as np

# Internal Imports
from structures.image import FaceSegment


class AbstractDetectionModel:

    def load(self, model_path):
        raise NotImplementedError

    def predict(self, inputs: List[np.ndarray]) -> List[List[FaceSegment]]:
        raise NotImplementedError
