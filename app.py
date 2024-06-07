# Standard Imports
import logging
import tornado

# Third Party Imports

# Internal Imports
from components.detection import detection
from components.embeddings import represent
from components.verification import verification, recognize
from models.model_holder import ModelHolder
from stores.store_holder import StoreHolder
from configurations.config import app_config
from components.operations import add_images_to_image_store

PORT = 8000
TMP_DIR = "/tmp"


class BaseHandler(tornado.web.RequestHandler):

    async def post(self):
        try:
            payloads = tornado.escape.json_decode(self.request.body)
        except Exception as e:
            raise tornado.web.HTTPError(status_code=400, log_message=str(e))

        results = await self._process_payload(payloads)
        return self.write({"results": results})

    async def _process_payload(self, payloads):
        raise NotImplementedError()


class AddHandler(BaseHandler):

    async def _process_payload(self, payloads):
        images = [payload["image"] for payload in payloads]
        user_ids = [payload["image"] for payload in payloads]

        outputs = add_images_to_image_store(
            images=images,
            user_ids=user_ids,
            store_name=app_config.image_store.store_name,
            store_path=app_config.image_store.path,
            embedding_name=app_config.embedding_model.name,
            detector_name=app_config.detector_model.name,
            **app_config.detector_model.arguments,
        )
        return outputs


class VerifyHandler(BaseHandler):

    async def _process_payload(self, payloads):
        images = [(payload["image1"], payload["image2"]) for payload in payloads]
        outputs = verification(
            image_tuples=images,
            embedding_name=app_config.embedding_model.name,
            detector_name=app_config.detector_model.name,
            **app_config.detector_model.arguments,
        )
        return outputs


class RecognitionHandler(BaseHandler):

    async def _process_payload(self, payloads):
        images = [payload["image"] for payload in payloads]
        outputs = recognize(
            images=images,
            embedding_name=app_config.embedding_model.name,
            store_name=app_config.image_store.store_name,
            detector_name=app_config.detector_model.name,
            **app_config.detector_model.arguments,
        )
        return outputs


class FaceDetectionHandler(BaseHandler):

    async def _process_payload(self, payloads):
        outputs = detection(
            images=[payload["image"] for payload in payloads["payloads"]],
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
        return responses


class FaceRepresentationHandler(BaseHandler):

    async def _process_payload(self, payloads):

        outputs = represent(
            images=[payload["image"] for payload in payloads["payloads"]],
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
        return responses


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
