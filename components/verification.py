# Standard Imports
import logging
from typing import Union, List, Tuple

# Third Part Imports
import numpy as np

# Internal Imports
from utils.utils import timeit
from components.embeddings import represent
from stores.store_holder import StoreHolder
from stores.image_store import ImageMetadataStore
from constants.constants import VERIFICATION_THRESHOLDS


@timeit
def verification(
    image_tuples: Union[
        Tuple[str, str],
        Tuple[np.ndarray, np.ndarray],
        List[Tuple[str, str]],
        List[Tuple[np.ndarray, np.ndarray]],
    ],
    embedding_name,
    detector_name=None,
    align=False,
    expand_percentage=0,
):
    if not isinstance(image_tuples, list):
        image_tuples = [image_tuples]

    results = []
    for image1, image2 in image_tuples:
        try:
            faces1 = represent(
                images=image1,
                embedding_name=embedding_name,
                detector_name=detector_name,
                align=align,
                expand_percentage=expand_percentage,
            )[0]
        except Exception as e:
            raise Exception("Error in generating embedding for image1")

        try:
            faces2 = represent(
                images=image2,
                embedding_name=embedding_name,
                detector_name=detector_name,
                align=align,
                expand_percentage=expand_percentage,
            )[0]
        except Exception as e:
            raise Exception("Error in generating embedding for image1")

        matching_faces, distance = None, None
        for face1 in faces1:
            for face2 in faces2:
                embedding1 = face1.get_embedding(embedding_name)
                embedding2 = face2.get_embedding(embedding_name)
                cosine_similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                if distance is None or cosine_similarity > distance:
                    distance = cosine_similarity
                    matching_faces = (face1, face2)

        if matching_faces is None:
            results.append({})
        else:
            results.append(
                {
                    "verified": distance
                    >= VERIFICATION_THRESHOLDS[embedding_name]["cosine_similarity"],
                    "distance": distance,
                    "metric": "cosine_similarity",
                    "threshold": VERIFICATION_THRESHOLDS[embedding_name][
                        "cosine_similarity"
                    ],
                    "embedding_model": embedding_name,
                    "detector_model": detector_name,
                    "faces": {
                        "image1": matching_faces[0].facial_segments.to_json(),
                        "image2": matching_faces[1].facial_segments.to_json(),
                    },
                }
            )
    return results


def recognize(
    images: Union[str, np.ndarray, List[str], List[np.ndarray]],
    embedding_name,
    store_name,
    detector_name=None,
    align=False,
    expand_percentage=0,
):
    try:
        image_store: ImageMetadataStore = StoreHolder.get_store(store_name)
    except Exception as e:
        raise Exception(f"Error in getting {store_name} store: {str(e)}")

    if not isinstance(images, list):
        images = [images]

    representations = represent(
        images=images,
        embedding_name=embedding_name,
        detector_name=detector_name,
        align=align,
        expand_percentage=expand_percentage,
    )

    queries = []
    for detected_faces in representations:
        for face in detected_faces:
            queries.append(face.get_embedding(embedding_name))

    search_results = image_store.search(queries, 1)
    outputs, pointer = [], 0
    for idx in range(len(images)):
        detected_faces = representations[idx]
        search_result = search_results[pointer : pointer + len(detected_faces)]
        closest_match, closest_distance = None, None
        for idy, face in enumerate(detected_faces):
            req_result = search_result[idy]
            for _search in req_result:
                if len(_search) > 0 and (
                    closest_distance is None or _search[1] >= closest_distance
                ):
                    closest_distance = _search[1]
                    closest_match = _search[0]
        if closest_match is not None:
            outputs.append(
                {
                    "verified": (
                        True
                        if closest_distance
                        >= VERIFICATION_THRESHOLDS[embedding_name]["cosine_similarity"]
                        else False
                    ),
                    "distance": float(closest_distance),
                    "metric": "cosine_similarity",
                    "threshold": VERIFICATION_THRESHOLDS[embedding_name][
                        "cosine_similarity"
                    ],
                    "embedding_model": embedding_name,
                    "detector_model": detector_name,
                    "userId": image_store.get(closest_match.image_hash_key).user_id,
                }
            )
        else:
            outputs.append(
                {
                    "verified": False,
                    "distance": 0.0,
                    "metric": "cosine_similarity",
                    "threshold": VERIFICATION_THRESHOLDS[embedding_name][
                        "cosine_similarity"
                    ],
                    "embedding_model": embedding_name,
                    "detector_model": detector_name,
                    "userId": None,
                }
            )
        pointer += len(detected_faces)
    return outputs
