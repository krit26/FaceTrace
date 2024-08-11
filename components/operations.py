# Standard Imports
import os
import logging
from datetime import datetime
from typing import Union, List

# Third Part Imports
import numpy as np

# Internal Imports
from utils.utils import timeit
from structures.image import ImageMetadata
from stores.store_holder import StoreHolder
from components.embeddings import represent
from stores.image_store import ImageMetadataStore
from utils.image_utils import export_image_using_pil
from constants.constants import DEFAULT_DATABASE_PATH


@timeit
def add_images_to_image_store(
    images: Union[str, np.ndarray, List[str], List[np.ndarray]],
    user_ids: List[str],
    store_name,
    embedding_name,
    detector_name,
    store_path=None,
    align=False,
    expand_percentage=0,
    **kwargs,
):
    if store_path is None:
        store_path = DEFAULT_DATABASE_PATH

    image_store: ImageMetadataStore = StoreHolder.get_store(store_name)

    if not isinstance(images, list):
        images = [images]

    representations = represent(
        images=images,
        embedding_name=embedding_name,
        detector_name=detector_name,
        align=align,
        expand_percentage=expand_percentage,
        **kwargs,
    )
    try:
        image_metadata = []
        for idx, detected_face in enumerate(representations):
            image_base_path = os.path.join(
                store_path, "database", "images", user_ids[idx]
            )

            if not os.path.exists(image_base_path):
                os.makedirs(image_base_path, exist_ok=True)
            image_path = os.path.join(
                image_base_path,
                f"image_{datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')}.jpeg",
            )
            image_path = export_image_using_pil(images[idx], image_path)
            image_metadata.append(
                ImageMetadata(
                    image_path=image_path,
                    user_id=user_ids[idx],
                    detected_faces=detected_face,
                )
            )

        results = []
        for metadata in image_metadata:
            is_add, errors = image_store.add(metadata)

            # if image already exists delete it from the path
            if is_add is False:
                logging.error(
                    f"Trying to add duplicate image for user {metadata.user_id}"
                )
                if os.path.exists(metadata.image_path):
                    os.remove(metadata.image_path)

            results.append({"success": is_add, "errors": errors})
    except Exception as e:
        import traceback

        logging.info(traceback.format_exc())
    return results
