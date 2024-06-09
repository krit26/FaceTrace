# Standard Imports
import os
from typing import List, Dict
from dataclasses import dataclass, field

# Third Party Imports
import numpy as np

# Internal Imports
from utils.image_utils import image_hash


@dataclass
class FaceSegment:
    x: int
    y: int
    w: int
    h: int
    left_eye: tuple = None
    right_eye: tuple = None
    confidence: float = 0

    def to_json(self):
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "confidence": self.confidence,
        }

    @staticmethod
    def from_json(json_dict):
        return FaceSegment(
            x=json_dict["x"],
            y=json_dict["y"],
            w=json_dict["w"],
            h=json_dict["h"],
            left_eye=json_dict["left_eye"],
            right_eye=json_dict["right_eye"],
            confidence=json_dict["confidence"],
        )


@dataclass
class DetectedFace:
    model_name: str
    image: np.ndarray
    facial_segments: FaceSegment
    alignment: bool = False
    expand_percentage: float = 0
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    def __str__(self):
        return f"{self.model_name}_{self.alignment}_{self.expand_percentage}"

    def get_embedding(self, model_name):
        return self.embeddings.get(model_name, None)

    def add_embedding(self, model_name: str, embedding: np.ndarray):
        self.embeddings[model_name] = embedding

    def to_json(self):
        return {
            "model_name": self.model_name,
            "image": self.image.tolist(),
            "face_segments": self.facial_segments.to_json(),
            "alignment": self.alignment,
            "expand_percentage": self.expand_percentage,
            "embeddings": {
                model_name: embedding.tolist()
                for model_name, embedding in self.embeddings.items()
            },
        }

    @staticmethod
    def from_json(json_dict):
        return DetectedFace(
            model_name=json_dict["model_name"],
            image=np.array(json_dict["image"]),
            facial_segments=FaceSegment.from_json(json_dict["face_segments"]),
            alignment=json_dict["alignment"],
            expand_percentage=json_dict["expand_percentage"],
            embeddings={
                model_name: np.array(embedding)
                for model_name, embedding in json_dict["embeddings"].items()
            },
        )


class ImageMetadata:

    def __init__(self, image_path, user_id, hash_key=None, detected_faces=None):
        self._image_path = image_path
        self._user_id = user_id

        if hash_key is None:
            hash_key = image_hash(image_path)
        self._image_hash = hash_key

        if detected_faces is None:
            detected_faces = []
        self._detected_faces: List[DetectedFace] = detected_faces

    def to_json(self):
        return {
            "image_path": "/".join(
                self._image_path.split("/")[-3:]
            ),  # relative path "images/<user_id>/image.jpeg"
            "user_id": self._user_id,
            "image_hash": self._image_hash,
            "detected_faces": [face.to_json() for face in self._detected_faces],
        }

    @staticmethod
    def from_json(metadata, base_path=None):
        return ImageMetadata(
            image_path=(
                os.path.join(base_path, metadata["image_path"])
                if base_path
                else metadata["image_path"]
            ),
            user_id=metadata["user_id"],
            hash_key=metadata.get("image_hash", None),
            detected_faces=[
                DetectedFace.from_json(face)
                for face in metadata.get("detected_faces", [])
            ],
        )

    @property
    def hash_key(self):
        return self._image_hash

    @property
    def image_path(self):
        return self._image_path

    @property
    def user_id(self):
        return self._user_id

    @property
    def detected_faces(self):
        return self._detected_faces

    @detected_faces.setter
    def detected_faces(self, detected_faces):
        self._detected_faces = detected_faces


@dataclass
class ImageVectorMetadata:
    index: int
    image_hash_key: str
    embedding_model: str
    embedding: np.ndarray

    def __str__(self):
        return f"{self.image_hash_key}_{self.index}_{self.embedding_model}"
