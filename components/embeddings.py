# Standard Imports
import logging
from typing import Union, List

# Third Part Imports
import numpy as np

# Internal Imports
from utils.utils import timeit
from components.detection import detection
from models.model_holder import ModelHolder
from utils.image_utils import load_image_using_pil
from models.embeddings import AbstractEmbeddingModel
from structures.image import FaceSegment, DetectedFace


@timeit
def embeddings(
    images: Union[str, np.ndarray, List[str], List[np.ndarray]], model_name
) -> List[np.ndarray]:

    try:
        embedding_model: AbstractEmbeddingModel = ModelHolder.get_or_load_model(
            model_name
        )
    except Exception as e:
        raise Exception(f"Error in loading {model_name} embedding model: {str(e)}")

    if not isinstance(images, list):
        images = [images]

    images = [load_image_using_pil(image) for image in images]

    try:
        outputs = embedding_model.predict(images)
    except Exception as e:
        raise Exception(f"Error in embedding model: {str(e)}")

    return [np.array(output) for output in outputs]


@timeit
def represent(
    images: Union[str, np.ndarray, List[str], List[np.ndarray]],
    embedding_name,
    detector_name=None,
    align=False,
    expand_percentage=0,
) -> List[List[DetectedFace]]:

    if not isinstance(images, list):
        images = [images]

    if len(images) == 0:
        return []

    images = [load_image_using_pil(image) for image in images]

    detected_outputs = []
    if detector_name:
        detected_outputs = detection(
            images, detector_name, align=align, expand_percentage=expand_percentage
        )
    else:
        for idx, image in enumerate(images):
            detected_outputs.append(
                [
                    DetectedFace(
                        model_name=detector_name,
                        image=image,
                        facial_segments=FaceSegment(
                            x=0, y=0, w=image.shape[0], h=image.shape[1]
                        ),
                        alignment=align,
                        expand_percentage=expand_percentage,
                    )
                ]
            )

    embedding_batch = []
    for detected_output in detected_outputs:
        for face in detected_output:
            embedding_batch.append(face.image)

    embeds = embeddings(images=embedding_batch, model_name=embedding_name)

    pointer = 0
    for detected_output in detected_outputs:
        for face in detected_output:
            face.add_embedding(model_name=embedding_name, embedding=embeds[pointer])
            pointer += 1
    return detected_outputs
