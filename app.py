# Standard Imports
import logging
import tornado

# Third Party Imports

# Internal Imports
from components.detection import detection
from components.embeddings import represent
from models.model_holder import ModelHolder
from stores.store_holder import StoreHolder
from configurations.config import app_config

PORT = 8000
TMP_DIR = "/tmp"


class AddHandler(tornado.web.RequestHandler):

    async def post(self):
        try:
            payload = tornado.escape.json_decode(self.request.body)
        except Exception as e:
            raise tornado.web.HTTPError(status_code=400, log_message=str(e))

        logging.info("Payload: {}".format(payload))
        return "Looks fine to me"


class RecognitionHandler(tornado.web.RequestHandler):

    async def post(self):
        try:
            payload = tornado.escape.json_decode(self.request.body)
        except Exception as e:
            raise tornado.web.HTTPError(status_code=400, log_message=str(e))

        logging.info("Payload: {}".format(payload))
        return "Looks fine to me"


class FaceDetectionHandler(tornado.web.RequestHandler):

    async def post(self):

        try:
            payload = tornado.escape.json_decode(self.request.body)
        except Exception as e:
            raise tornado.web.HTTPError(status_code=400, log_message=str(e))

        outputs = detection(
            images=[payload["image"] for payload in payload["payloads"]],
            model_name=app_config.detector_model.name,
            align=app_config.detector_model.arguments.get("align", False),
            expand_percentage=app_config.detector_model.arguments.get(
                "expand_percentage", 0
            ),
        )

        responses = []
        for output in outputs:
            faces = []
            for face in output:
                faces.append(
                    {
                        "x": face.facial_segments.x,
                        "y": face.facial_segments.y,
                        "w": face.facial_segments.w,
                        "h": face.facial_segments.h,
                        "conf": face.facial_segments.confidence,
                    }
                )
            responses.append({"faces": faces})
        return self.write({"results": responses})


class FaceRepresentationHandler(tornado.web.RequestHandler):

    async def post(self):

        try:
            payload = tornado.escape.json_decode(self.request.body)
        except Exception as e:
            raise tornado.web.HTTPError(status_code=400, log_message=str(e))

        outputs = represent(
            images=[payload["image"] for payload in payload["payloads"]],
            embedding_name=app_config.embedding_model.name,
            detector_name=app_config.detector_model.name,
            align=app_config.detector_model.arguments.get("align", False),
            expand_percentage=app_config.detector_model.arguments.get(
                "expand_percentage", 0
            ),
        )

        responses = []
        for output in outputs:
            faces = []
            for face in output:
                _dict = {
                    "x": face.facial_segments.x,
                    "y": face.facial_segments.y,
                    "w": face.facial_segments.w,
                    "h": face.facial_segments.h,
                    "conf": face.facial_segments.confidence,
                }
                for embeddings in face.embeddings:
                    if app_config.embedding_model.name == embeddings.model_name:
                        _dict["embedding"] = embeddings.embedding
                faces.append(_dict)
            responses.append({"faces": faces})
        return self.write({"results": responses})


def app_initializer():
    if app_config.detector_model:
        try:
            logging.info(
                f"Loading {app_config.detector_model.name} face detector model"
            )
            _ = ModelHolder.get_or_load_model(
                model_name=app_config.detector_model.name,
                load=True,
                model_path=app_config.detector_model.model_path,
                **app_config.detector_model.arguments,
            )
            logging.info("Successfully loaded face detector model")
        except Exception as e:
            raise tornado.web.HTTPError(status_code=500, log_message=str(e))

    if app_config.detector_model:
        try:
            logging.info(f"Loading {app_config.embedding_model.name} embedding model")
            _ = ModelHolder.get_or_load_model(
                model_name=app_config.embedding_model.name,
                load=True,
                model_path=app_config.embedding_model.model_path,
                **app_config.embedding_model.arguments,
            )
            logging.info("Successfully loaded embedding model")
        except Exception as e:
            raise tornado.web.HTTPError(status_code=500, log_message=str(e))

    if app_config.image_store:
        try:
            logging.info(f"Loading image metadata from {app_config.image_store.path}")
            _ = StoreHolder.get_or_load_store(
                store_name=app_config.image_store.store_name,
                builder_name=app_config.image_store.builder_name,
                store_path=app_config.image_store.path,
                **app_config.image_store.arguments,
            )
            logging.info("Successfully loaded image metadata")
        except Exception as e:
            raise tornado.web.HTTPError(status_code=500, log_message=str(e))

    if app_config.index_model:
        if not StoreHolder.get_store(app_config.index_model.store_name):
            logging.warning(
                f"{app_config.index_model.store_name} store does not exist\n"
                f"Please load the store first for verification service to work"
            )

        store = StoreHolder.get_store(app_config.index_model.store_name)


def main():
    logging.basicConfig(level=logging.INFO)
    app_initializer()
    application = tornado.web.Application(
        handlers=[
            (r"/add", AddHandler),
            (r"/recognize", RecognitionHandler),
            (r"/face-detect", FaceDetectionHandler),
            (r"/represent", FaceRepresentationHandler),
        ]
    )
    application.listen(8000)
    logging.info(f"Starting Application at: {PORT}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
