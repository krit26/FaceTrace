# Standard Imports
import logging
import time
import asyncio
from typing import List
from threading import Thread

# Third Party Imports
import faiss
import numpy as np

# Internal Imports
from utils.utils import dump_json
from utils.utils import normalize_vectors
from structures.image import ImageMetadata, ImageVectorMetadata


class ImageMetadataStore:

    def __init__(self, image_metadata: List[ImageMetadata], **kwargs):
        self._image_metadata = image_metadata
        self._hash_vs_images = {
            metadata.hash_key: metadata for metadata in image_metadata
        }
        self._user_id_vs_images = {}
        for metadata in self._image_metadata:
            if metadata.user_id not in self._user_id_vs_images:
                self._user_id_vs_images[metadata.user_id] = []
            self._user_id_vs_images[metadata.user_id].append(metadata)

        logging.info("kwargs: {}".format(kwargs))
        self.vector_indexing = kwargs.get("vector_indexing", False)
        if self.vector_indexing:
            _vector_indexing_kwargs = kwargs.get("vector_indexing_kwargs", {})
            self._index_type = kwargs.get("index_type", "Flat")
            self._metric = kwargs.get("metric", "inner_product")
            self._embedding_model = kwargs.get("embedding_model", "FaceNet512")
            self._build_index()

        self._metadata_path = kwargs.get("metadata_path", None)
        if self._metadata_path:
            self._dump_in_progress = False
            self._dump_loop_stop_event = asyncio.Event()
            self._dump_thread = Thread(
                target=self._dump_loop, args=(self._dump_loop_stop_event,), daemon=True
            ).start()

    def _build_index(self):
        logging.info("building index")
        vectors_metadata = []
        for metadata in self._image_metadata:
            for idx, face in enumerate(metadata.detected_faces):
                for vector in face.embeddings:
                    if self._embedding_model == vector.model_name:
                        vectors_metadata.append(
                            ImageVectorMetadata(
                                index=idx,
                                embedding_model=self._embedding_model,
                                image_hash_key=metadata.hash_key,
                                embedding=vector.embedding,
                            )
                        )
        vectors = [meta.embedding for meta in vectors_metadata]
        self._build_faiss_index(vectors)
        self._vector_index_metadata = vectors_metadata
        logging.info("vector index created successfully")

    def _build_faiss_index(self, vectors):
        logging.info("building faiss index")
        vectors = np.array(vectors)
        dimension = vectors.shape[1]
        if self._metric == "inner_product":
            self._faiss = faiss.index_factory(
                dimension, self._index_type, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self._faiss = faiss.index_factory(dimension, self._index_type)
        if self._metric == "inner_product":
            vectors = normalize_vectors(vectors)
        self._faiss.add(vectors)
        logging.info(
            "Faiss index created with index size: {}".format(self._faiss.ntotal)
        )

    def add(self, image_metadata: ImageMetadata):
        if image_metadata.hash_key() in self._hash_vs_images:
            raise Exception("This image metadata already exists")

        while True:
            if self._dump_in_progress:
                time.sleep(1)
            else:
                break

        if image_metadata.hash_key in self._hash_vs_images:
            metadata = self._hash_vs_images[image_metadata.hash_key]
            return False, f"Duplicate image. image {metadata.image_path} already exists"

        self._image_metadata.append(image_metadata)
        self._hash_vs_images[image_metadata.hash_key()] = image_metadata
        self._user_id_vs_images[image_metadata.user_id()].append(image_metadata)

        if self.vector_indexing:
            self._add_to_index(image_metadata)
        return True, None

    def _add_to_index(self, image_metadata: ImageMetadata):
        vectors = []
        for idx, face in enumerate(image_metadata.detected_faces):
            for vector in face:
                if self._embedding_model == vector.model_name:
                    vectors.append(
                        ImageVectorMetadata(
                            index=idx,
                            embedding_model=self._embedding_model,
                            image_hash_key=image_metadata.hash_key,
                            embedding=vector.embedding,
                        )
                    )
        if len(vectors) == 0:
            return
        if self._metric == "inner_product":
            vectors = normalize_vectors(vectors)
        self._faiss.add(vectors)
        logging.info("Faiss index size: {}".format(self._faiss.ntotal))

    def search(self, queries, nearest_neighbours=3):
        if self._faiss is None:
            raise Exception("Make sure vectors indexing is enable")
        if self._faiss.ntotal == 0:
            return [], []

        if not isinstance(queries, np.ndarray):
            queries = np.array(queries, dtype=np.float32)

        if self._metric == "inner_product":
            queries = normalize_vectors(queries)

        indices, distances = self._faiss.search(queries, nearest_neighbours)
        search_result = []
        for idx in range(len(distances)):
            results = []
            for index, dist in zip(indices[idx], distances[idx]):
                results.append((self._vector_index_metadata[index], dist))
            search_result.append(results)
        return search_result

    def get(self, hash_key: str) -> ImageMetadata:
        if hash_key in self._hash_vs_images:
            return self._hash_vs_images[hash_key]
        raise KeyError(f"hash key {hash_key} not found in ImageMetadataStore")

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
