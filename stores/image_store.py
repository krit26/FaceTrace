# Standard Imports
import logging
import time
import asyncio
from typing import List
from threading import Thread
from dataclasses import dataclass

# Third Party Imports

# Internal Imports
from utils.utils import dump_json
from structures.image import ImageMetadata


class ImageMetadataStore:

    def __init__(self, image_metadata: List[ImageMetadata], metadata_path):
        self._image_metadata = image_metadata
        self._metadata_path = metadata_path
        self._hash_vs_images = {
            metadata.hash_key: metadata for metadata in image_metadata
        }
        self._user_id_vs_images = {
            metadata.user_id: metadata for metadata in image_metadata
        }

        self._dump_in_progress = False
        self._dump_loop_stop_event = asyncio.Event()
        self._dump_thread = Thread(
            target=self._dump_loop, args=(self._dump_loop_stop_event,), daemon=True
        ).start()

    def add(self, image_metadata: ImageMetadata):
        if image_metadata.hash_key() in self._hash_vs_images:
            raise Exception("This image metadata already exists")

        while True:
            if self._dump_in_progress:
                time.sleep(1)
            else:
                break

        self._image_metadata.append(image_metadata)
        self._hash_vs_images[image_metadata.hash_key()] = image_metadata
        self._user_id_vs_images[image_metadata.user_id()] = image_metadata

    def get(self, user_id):
        if user_id in self._user_id_vs_images:
            return self._user_id_vs_images[user_id]
        raise KeyError(f"User {user_id} not found in ImageMetadataStore")

    def _dump_loop(self, stop_event):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.__dump_loop(stop_event))
        loop.close()

    async def __dump_loop(self, stop_event):
        while not stop_event.is_set():
            logging.info("Starting ImageMetadataStore Dumping")
            self._dump_in_progress = True
            metadata = [
                image_metadata.to_json() for image_metadata in self._image_metadata
            ]
            _ = dump_json(metadata, self._metadata_path)
            self._dump_in_progress = False
            logging.info("dumping ImageMetadataStore Successfully")
            await asyncio.sleep(10)
