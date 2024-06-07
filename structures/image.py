# Standard Imports
from typing import List
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
class FaceEmbeddings:
    model_name: str
    embedding: np.ndarray

    def __str__(self):
        return f"{self.model_name}"

    def to_json(self):
        return {"model_name": self.model_name, "embedding": self.embedding.tolist()}

    @staticmethod
    def from_json(json_dict):
        return FaceEmbeddings(
            model_name=json_dict["model_name"],
            embedding=np.array(json_dict["embedding"]),
        )


@dataclass
class DetectedFace:
    model_name: str
    image: np.ndarray
    facial_segments: FaceSegment
    alignment: bool = False
    expand_percentage: float = 0
    embeddings: List[FaceEmbeddings] = field(default_factory=list)

    def __post_init__(self):
        self.model_vs_embeddings = {}
        if self.embeddings is not None:
            self.model_vs_embeddings = {
                embedding.model_name: embedding for embedding in self.embeddings
            }

    def __str__(self):
        return f"{self.model_name}_{self.alignment}_{self.expand_percentage}"

    def get_embedding(self, model_name):
        return self.model_vs_embeddings.get(model_name, None)

    def to_json(self):
        return {
            "model_name": self.model_name,
            "image": self.image.tolist(),
            "face_segments": self.facial_segments.to_json(),
            "alignment": self.alignment,
            "expand_percentage": self.expand_percentage,
            "embeddings": [embedding.to_json() for embedding in self.embeddings],
        }

    @staticmethod
    def from_json(json_dict):
        return DetectedFace(
            model_name=json_dict["model_name"],
            image=np.array(json_dict["image"]),
            facial_segments=FaceSegment.from_json(json_dict["face_segments"]),
            alignment=json_dict["alignment"],
            expand_percentage=json_dict["expand_percentage"],
            embeddings=[
                FaceEmbeddings.from_json(embedding)
                for embedding in json_dict["embeddings"]
            ],
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
            "image_path": self._image_path,
            "user_id": self._user_id,
            "image_hash": self._image_hash,
            "detected_faces": [face.to_json() for face in self._detected_faces],
        }

    @staticmethod
    def from_json(metadata):
        return ImageMetadata(
            image_path=metadata["image_path"],
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
