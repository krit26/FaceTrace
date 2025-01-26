# Standard Imports
import logging
import time
import asyncio
from threading import Thread
from typing import List, Union

# Third Party Imports
import faiss
import numpy as np

# Internal Imports
from utils.utils import dump_pickle
from utils.utils import normalize_vectors
from constants.constants import (
    EMBEDDING_MODEL_DIMENSION,
    COSINE_SIMILARITY,
    EUCLIDEAN_L2,
)
from structures.image import ImageMetadata, ImageVectorMetadata, FaissSearchResult


class ImageMetadataStore:

    def __init__(self, image_metadata: List[ImageMetadata], **kwargs):
        self._image_metadata = image_metadata
        self._hash_vs_images = {
            metadata.hash_key: metadata for metadata in image_metadata
        }

        # Vector indexing arguments
        self.vector_indexing = kwargs.get("vector_indexing", False)
        _indexing_kwargs = kwargs.get("indexing_kwargs", {})
        self._index_type = _indexing_kwargs.get("index_type", "Flat")
        self._metric = _indexing_kwargs.get("metric", COSINE_SIMILARITY)
        self._embedding_model = _indexing_kwargs.get("embedding_model", "FaceNet512")
        self._vector_index_metadata = []
        self._faiss = None
        if self.vector_indexing:
            self._build_index()

        # Store dumping thread initializer
        self._store_path = kwargs.get("store_path", None)
        if self._store_path:
            _dumping_kwargs = kwargs.get("dumping_kwargs", {})
            self._dump_in_progress = False
            self._dump_loop_stop_event = asyncio.Event()
            self._dump_thread = Thread(
                target=self._dump_loop, args=(self._dump_loop_stop_event,), daemon=True
            ).start()
            self._dumping_interval = _dumping_kwargs.get("interval", 300)

    def add(self, image_metadata: ImageMetadata):
        while True:
            if self._dump_in_progress:
                time.sleep(1)
            else:
                break

        if image_metadata.hash_key in self._hash_vs_images:
            metadata = self._hash_vs_images[image_metadata.hash_key]
            return False, f"Duplicate image. image {metadata.image_path} already exists"

        self._image_metadata.append(image_metadata)
        self._hash_vs_images[image_metadata.hash_key] = image_metadata

        if self.vector_indexing:
            self._add_to_index(image_metadata)
        return True, None

    def _build_index(self):
        logging.info("building index")
        self._initialize_faiss_index()

        self._add_to_index(self._image_metadata)
        logging.info("vector index created successfully")

    def _initialize_faiss_index(self):
        dimension = EMBEDDING_MODEL_DIMENSION[self._embedding_model]
        if self._metric == COSINE_SIMILARITY:
            self._faiss = faiss.index_factory(
                dimension, self._index_type, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self._faiss = faiss.index_factory(dimension, self._index_type)

    def _create_vector_metadata_from_image_metadata(
        self, image_metadata: ImageMetadata
    ):
        vectors_metadata = []
        for idx, face in enumerate(image_metadata.detected_faces):
            vectors_metadata.append(
                ImageVectorMetadata(
                    index=idx,
                    embedding_model=self._embedding_model,
                    image_hash_key=image_metadata.hash_key,
                    embedding=face.get_embedding(self._embedding_model),
                )
            )
        return vectors_metadata

    def _add_to_index(self, image_metadata: Union[ImageMetadata, List[ImageMetadata]]):
        if not isinstance(image_metadata, list):
            image_metadata = [image_metadata]
        vectors_metadata = []
        for metadata in image_metadata:
            vectors_metadata.extend(
                self._create_vector_metadata_from_image_metadata(metadata)
            )
        self._vector_index_metadata.extend(vectors_metadata)
        self._add_vectors_to_index([meta.embedding for meta in vectors_metadata])
        assert len(self._vector_index_metadata) == self._faiss.ntotal

    def _add_vectors_to_index(self, vectors: List[np.ndarray]):
        if len(vectors) == 0:
            return
        vectors = np.array(vectors, dtype=np.float32)
        if self._metric in [COSINE_SIMILARITY]:
            vectors = normalize_vectors(vectors)
        self._faiss.add(vectors)
        logging.info("Faiss index size: {}".format(self._faiss.ntotal))

    def search(self, queries, nearest_neighbours=3):
        if self._faiss is None:
            raise Exception("Make sure vectors indexing is enabled")
        if self._faiss.ntotal == 0:
            return [[None] * nearest_neighbours] * len(queries)

        if not isinstance(queries, np.ndarray):
            queries = np.array(queries, dtype=np.float32)

        if self._metric in [COSINE_SIMILARITY]:
            queries = normalize_vectors(queries)

        logging.info("queries shape: {}".format(queries.shape))
        search_result = []
        if queries.shape[0] == 0:
            return [[None] * nearest_neighbours] * len(queries)

        distances, indices = self._faiss.search(queries, nearest_neighbours)
        for idx in range(len(queries)):
            results = []
            for index, dist in zip(indices[idx], distances[idx]):
                results.append(
                    FaissSearchResult(
                        key=self._vector_index_metadata[index],
                        distance=dist,
                        metric_type=self._metric,
                    )
                )
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
            start_time = time.time() * 1000
            self._dump_in_progress = True
            metadata = [
                image_metadata.to_json() for image_metadata in self._image_metadata
            ]
            _ = dump_pickle(metadata, self._store_path)
            self._dump_in_progress = False
            logging.info(
                f"dumping ImageMetadataStore Successfully, time taken: {round(time.time() * 1000 - start_time)} ms"
            )
            await asyncio.sleep(self._dumping_interval)
