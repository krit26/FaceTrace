# Standard Imports
import os
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


@timeit
def add_images_to_image_store(
    images: Union[str, np.ndarray, List[str], List[np.ndarray]],
    user_ids: List[str],
    store_name,
    store_path,
    embedding_name,
    detector_name,
    align=False,
    expand_percentage=0,
):
    try:
        image_store: ImageMetadataStore = StoreHolder.get_store(store_name)
    except Exception as e:
        raise Exception(f"Error in getting {store_name} store: {str(e)}")

    if not isinstance(images, list):
        images = [images]

    outputs = represent(
        images=images,
        embedding_name=embedding_name,
        detector_name=detector_name,
        align=align,
        expand_percentage=expand_percentage,
    )

    image_metadata = []
    for idx, outputs in enumerate(outputs):
        image_path = export_image_using_pil(
            images[idx],
            os.path.join(
                store_path,
                "images",
                user_ids[idx],
                datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S"),
            ),
        )
        image_metadata.append(
            ImageMetadata(
                image_path=image_path,
                user_id=user_ids[idx],
                detected_faces=outputs[idx],
            )
        )

    results = []
    for metadata in image_metadata:
        is_add, errors = image_store.add(metadata)
        results.append({"success": is_add, "errors": errors})
    return results
