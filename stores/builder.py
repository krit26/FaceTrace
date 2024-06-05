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

    def __init__(self, metadata_file_path: str):
        self.image_metadata_store = None
        self.file_path = metadata_file_path

    def load(self, base_path: str):
        metadata_file_path = os.path.join(base_path, "metadata.json")

        if os.path.exists(metadata_file_path):
            metadata = load_json(metadata_file_path)

            if not isinstance(metadata, list):
                raise Exception("image metadata should be list of dictionaries")

            logging.info(f"Length of metadata list: {len(metadata)}")
            image_metadata = []
            for meta in metadata:
                if not isinstance(meta, dict):
                    raise Exception("image metadata should be a dictionary")
                image_metadata.append(ImageMetadata.from_json(meta))

            self.image_metadata_store = ImageMetadataStore(
                image_metadata=image_metadata, metadata_path=metadata_file_path
            )
            logging.info("image metadata store loaded successfully")
            return self.image_metadata_store

        expected_images_path = os.path.join(base_path, "images", "*", "*.jpeg")
        images = glob.glob(expected_images_path)
        if len(images) == 0:
            logging.warning("No images found in {}".format(expected_images_path))
        logging.info("Number of images: {}".format(len(images)))
        image_metadata = [
            ImageMetadata(image_path=image, user_id=image.split("/")[-2])
            for image in images
        ]
        for image in image_metadata:
            detected_faces = represent(
                images=[image.image_path],
                embedding_name=app_config.embedding_model.name,
                detector_name=app_config.detector_model.name,
                align=app_config.detector_model.arguments.get("align", False),
                expand_percentage=app_config.detector_model.arguments.get(
                    "expand_percentage", 0
                ),
            )
            image.detected_faces = detected_faces[0]
        self.image_metadata_store = ImageMetadataStore(
            image_metadata=image_metadata, metadata_path=metadata_file_path
        )
        logging.info("image metadata store loaded successfully")
        return self.image_metadata_store
