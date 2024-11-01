# Standard Imports
import logging
from typing import List, Union

# Third Party Imports
import torch
import numpy as np

# Internal Imports
from structures.image import FaceSegment
from models.detectors.fast_mtcnn.model import MTCNN
from models.detectors import AbstractDetectionModel


class FastMtcnn(AbstractDetectionModel):

    def __init__(self, **kwargs):
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

    def load(self, model_path: Union[str, None]):
        self.model = MTCNN(device=self.device)

    def predict(self, inputs: List[np.ndarray]) -> List[List[FaceSegment]]:
        outputs = []
        for image in inputs:
            detections = self.model.detect(image, landmarks=True)
            detected_faces = []

            if detections is None or detections[0] is None:
                outputs.append(detected_faces)
                logging.error("No faces detected")
                continue

            for regions, confidence, eyes in zip(*detections):
                x, y, w, h = self._xyxy_to_xywh(regions)
                right_eye = eyes[0]
                left_eye = eyes[1]

                left_eye = tuple(int(i) for i in left_eye)
                right_eye = tuple(int(i) for i in right_eye)
                detected_faces.append(
                    FaceSegment(
                        x,
                        y,
                        w,
                        h,
                        left_eye,
                        right_eye,
                        confidence,
                    )
                )
            outputs.append(detected_faces)
        return outputs

    @staticmethod
    def _xyxy_to_xywh(regions):
        """
        Convert (x1, y1, x2, y2) format to (x, y, w, h) format.
        Args:
            regions (list or tuple): facial area coordinates as x, y, x+w, y+h
        Returns:
            regions (tuple): facial area coordinates as x, y, w, h
        """
        x, y, x_plus_w, y_plus_h = regions[0], regions[1], regions[2], regions[3]
        w = x_plus_w - x
        h = y_plus_h - y
        return x, y, w, h
