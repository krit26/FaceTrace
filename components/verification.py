# Standard Imports
import logging
from typing import Union, List, Tuple

# Third Part Imports
import numpy as np

# Internal Imports
from utils.utils import timeit
from components.embeddings import represent
from stores.store_holder import StoreHolder
from constants.constants import VERIFICATION_THRESHOLDS, COSINE_SIMILARITY, EUCLIDEAN_L2
from stores.image_store import ImageMetadataStore, FaissSearchResult


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
    metric="cosine_similarity",
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

        matching_faces, optimal_distance = None, None
        for face1 in faces1:
            for face2 in faces2:
                embedding1 = face1.get_embedding(embedding_name)
                embedding2 = face2.get_embedding(embedding_name)
                distance = get_distance(embedding1, embedding2, metric)
                if distance and (
                    optimal_distance is None
                    or verify_distance(
                        distance,
                        VERIFICATION_THRESHOLDS[embedding_name][metric],
                        metric,
                    )
                ):
                    optimal_distance = distance
                    matching_faces = (face1, face2)

        if matching_faces is None:
            results.append({})
        else:
            results.append(
                {
                    "verified": verify_distance(
                        optimal_distance,
                        VERIFICATION_THRESHOLDS[embedding_name][metric],
                        metric,
                    ),
                    "distance": round(optimal_distance, 2),
                    "metric": "cosine_similarity",
                    "threshold": VERIFICATION_THRESHOLDS[embedding_name][metric],
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

    search_results: List[List[FaissSearchResult]] = image_store.search(queries, 1)
    outputs, pointer = [], 0
    for idx in range(len(images)):
        detected_faces = representations[idx]
        search_result = search_results[pointer : pointer + len(detected_faces)]
        closest_result = None
        for idy, face in enumerate(detected_faces):
            req_result = search_result[idy]
            for _search in req_result:
                if _search and (
                    closest_result is None
                    or verify_distance(
                        _search.distance, closest_result.distance, _search.metric_type
                    )
                ):
                    closest_result = _search

        if closest_result is not None:
            output = {
                "verified": verify_distance(
                    closest_result.distance,
                    VERIFICATION_THRESHOLDS[embedding_name][closest_result.metric_type],
                    closest_result.metric_type,
                ),
                "distance": round(float(closest_result.distance), 2),
                "metric": closest_result.metric_type,
                "threshold": VERIFICATION_THRESHOLDS[embedding_name][
                    closest_result.metric_type
                ],
                "embedding_model": embedding_name,
                "detector_model": detector_name,
            }
            if output["verified"]:
                output["userId"] = image_store.get(
                    closest_result.key.image_hash_key
                ).user_id
            else:
                output["userId"] = None
            outputs.append(output)
        else:
            outputs.append(
                {
                    "verified": False,
                    "distance": 0.0,
                    "metric": None,
                    "threshold": None,
                    "embedding_model": embedding_name,
                    "detector_model": detector_name,
                    "userId": None,
                }
            )
        pointer += len(detected_faces)
    return outputs


def verify_distance(distance, threshold, metric_type):
    if distance is None:
        return False
    if metric_type == "cosine_similarity":
        if distance >= threshold:
            return True
        return False
    if distance < threshold:
        return True
    return False


def get_distance(vector1, vector2, metric_type):
    if metric_type == COSINE_SIMILARITY:
        return cosine_similarity(vector1, vector2)
    if metric_type == EUCLIDEAN_L2:
        return euclidean_distance(vector1, vector2)
    return None


def cosine_similarity(vector1, vector2):
    if not isinstance(vector1, np.ndarray):
        vector1 = np.array(vector1)
    if not isinstance(vector1, np.ndarray):
        vector2 = np.array(vector2)
    distance = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    return distance


def euclidean_distance(vector1, vector2):
    if not isinstance(vector1, np.ndarray):
        vector1 = np.array(vector1)
    if not isinstance(vector2, np.ndarray):
        vector2 = np.array(vector2)
    difference = vector1 - vector2
    distance = np.sqrt(np.sum(np.multiply(difference, difference)))
    return distance
