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
from components.verification import verification, recognize
from components.operations import add_images_to_image_store

PORT = 8000
TMP_DIR = "/tmp"


class BaseHandler(tornado.web.RequestHandler):

    async def post(self):
        try:
            payloads = tornado.escape.json_decode(self.request.body)
        except Exception as e:
            raise tornado.web.HTTPError(
                status_code=400, log_message=f"Error in decoding json inputs: {str(e)}"
            )

        results = await self._process_payload(payloads)
        return self.write({"results": results})

    async def _process_payload(self, payloads):
        raise NotImplementedError()


class AddHandler(BaseHandler):

    async def _process_payload(self, payloads):
        try:
            outputs = add_images_to_image_store(
                images=[payload["image"] for payload in payloads["payloads"]],
                user_ids=[payload["userId"] for payload in payloads["payloads"]],
                store_name=app_config.image_store.store_name,
                store_path=app_config.image_store.path,
                embedding_name=app_config.embedding_model.name,
                detector_name=app_config.detector_model.name,
                **app_config.detector_model.arguments,
            )
        except Exception as e:
            raise tornado.web.HTTPError(status_code=500, log_message=str(e))
        return outputs


class VerifyHandler(BaseHandler):

    async def _process_payload(self, payloads):
        try:
            outputs = verification(
                image_tuples=[
                    (payload["image1"], payload["image2"]) for payload in payloads
                ],
                embedding_name=app_config.embedding_model.name,
                detector_name=app_config.detector_model.name,
                **app_config.detector_model.arguments,
            )
        except Exception as e:
            raise tornado.web.HTTPError(status_code=500, log_message=str(e))
        return outputs


class RecognitionHandler(BaseHandler):

    async def _process_payload(self, payloads):
        try:
            outputs = recognize(
                images=[payload["image"] for payload in payloads["payloads"]],
                embedding_name=app_config.embedding_model.name,
                store_name=app_config.image_store.store_name,
                detector_name=app_config.detector_model.name,
                **app_config.detector_model.arguments,
            )
        except Exception as e:
            raise tornado.web.HTTPError(status_code=500, log_message=str(e))
        return outputs


class FaceDetectionHandler(BaseHandler):

    async def _process_payload(self, payloads):
        try:
            outputs = detection(
                images=[payload["image"] for payload in payloads["payloads"]],
                model_name=app_config.detector_model.name,
                align=app_config.detector_model.arguments.get("align", False),
                expand_percentage=app_config.detector_model.arguments.get(
                    "expand_percentage", 0
                ),
            )
        except Exception as e:
            raise tornado.web.HTTPError(status_code=500, log_message=str(e))

        responses = []
        for output in outputs:
            responses.append(
                {"faces": [face.facial_segments.to_json() for face in output]}
            )
        return responses


class FaceRepresentationHandler(BaseHandler):

    async def _process_payload(self, payloads):
        try:
            outputs = represent(
                images=[payload["image"] for payload in payloads["payloads"]],
                embedding_name=app_config.embedding_model.name,
                detector_name=app_config.detector_model.name,
                align=app_config.detector_model.arguments.get("align", False),
                expand_percentage=app_config.detector_model.arguments.get(
                    "expand_percentage", 0
                ),
            )
        except Exception as e:
            raise tornado.web.HTTPError(status_code=500, log_message=str(e))

        responses = []
        for output in outputs:
            faces = []
            for face in output:
                _dict = face.facial_segments.to_json()
                _dict["embedding"] = face.get_embedding(
                    app_config.embedding_model.name
                ).tolist()
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
            logging.info(f"Successfully loaded {app_config.detector_model.name}")
        except Exception as e:
            raise Exception(f"Error in loading detector model: {str(e)}")

    if app_config.embedding_model:
        try:
            logging.info(f"Loading {app_config.embedding_model.name} embedding model")
            _ = ModelHolder.get_or_load_model(
                model_name=app_config.embedding_model.name,
                load=True,
                model_path=app_config.embedding_model.model_path,
                **app_config.embedding_model.arguments,
            )
            logging.info(f"Successfully loaded {app_config.detector_model.name}")
        except Exception as e:
            raise Exception(f"Error in loading embedding model: {str(e)}")

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
            raise Exception(f"Error in loading image store: {str(e)}")


def main():
    logging.basicConfig(level=logging.INFO)
    app_initializer()
    application = tornado.web.Application(
        handlers=[
            (r"/add", AddHandler),
            (r"/verify", VerifyHandler),
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
