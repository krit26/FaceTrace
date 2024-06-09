# Standard Imports
import os
import glob
import logging

# Third Part Imports

# Internal Imports
from utils.utils import load_json
from stores import AbstractStoreBuilder
from structures.image import ImageMetadata
from components.embeddings import represent
from configurations.config import app_config
from stores.image_store import ImageMetadataStore


class ImageMetadataStoreBuilder(AbstractStoreBuilder):

    def __init__(self, store_path: str, **kwargs):
        self._image_store = None
        self.store_path = store_path

    def load(self, base_path: str, **kwargs):
        self.store_path = os.path.join(base_path, "metadata.json")
        kwargs.update({"store_path": self.store_path})

        # When base path doesn't exist
        if not os.path.exists(base_path):
            logging.info(f"Image store base path: {base_path} does not exists")
            os.makedirs(base_path, exist_ok=True)
            self._image_store = ImageMetadataStore([], **kwargs)
            return self._image_store

        # When base path exists
        image_metadata = []
        if os.path.exists(self.store_path):
            metadata = load_json(self.store_path)

            if not isinstance(metadata, list):
                raise Exception("image metadata should be list of dictionaries")

            logging.info(f"Length of metadata list: {len(metadata)}")
            for meta in metadata:
                if not isinstance(meta, dict):
                    logging.warning("image metadata should be a dictionary")
                    continue
                metadata = ImageMetadata.from_json(meta, base_path)
                if not os.path.exists(metadata.image_path):
                    logging.warning(
                        "image path {} does not exists. please check if the base path is correct"
                    )
                else:
                    image_metadata.append(ImageMetadata.from_json(meta, base_path))

        # Following logics check if their any image not present in metadata
        expected_images_path = os.path.join(base_path, "images", "*", "*.jpeg")
        image_paths = glob.glob(expected_images_path)
        if len(image_paths) == 0:
            logging.warning("No images found in {}".format(expected_images_path))

        existing_paths = set([metadata.image_path for metadata in image_metadata])
        image_paths = list(set(image_paths) - set(existing_paths))

        if len(image_paths) > 0:
            logging.info(
                "Number of images missing embeddings: {}".format(len(image_paths))
            )
            representations = represent(
                images=image_paths,
                embedding_name=app_config.embedding_model.name,
                detector_name=app_config.detector_model.name,
                align=app_config.detector_model.arguments.get("align", False),
                expand_percentage=app_config.detector_model.arguments.get(
                    "expand_percentage", 0
                ),
            )
            for idx, detected_faces in enumerate(representations):
                image_metadata.append(
                    ImageMetadata(
                        image_path=image_paths[idx],
                        user_id=image_paths[idx].split("/")[-2],
                        detected_faces=detected_faces,
                    )
                )

        self._image_store = ImageMetadataStore(image_metadata, **kwargs)
        logging.info(f"image metadata successfully loaded from {self.store_path}")
        return self._image_store
