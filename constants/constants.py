COSINE_SIMILARITY = "cosine_similarity"
EUCLIDEAN_L2 = "euclidean_l2"


EMBEDDING_MODEL_DIMENSION = {"FaceNet512": 512, "FaceNet128": 128}
VERIFICATION_THRESHOLDS = {
    "FaceNet512": {COSINE_SIMILARITY: 0.70, EUCLIDEAN_L2: 23.56},
    "FaceNet128": {COSINE_SIMILARITY: 0.70, EUCLIDEAN_L2: 23.56},
}

DEFAULT_DATABASE_PATH = "/tmp"

DEFAULT_RECOGNITION_RESPONSE = {
    "verified": False,
    "distance": 0.83,
    "metric": "cosine_similarity",
    "threshold": 0.6,
    "embedding_model": "FaceNet512",
    "detector_mode": "FastMtcnn",
    "userId": None,
}
