# Standard Imports
import os
import json
import base64
from typing import List

# Third Part Imports
import requests
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

# Internal Imports
from structures.image import FaceSegment
from utils.image_utils import export_image_using_pil


TMP = "/tmp"


def http_call(url, payload):
    response = requests.post(url, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()
    return response


def create_payload(image):
    return {
        "payloads": [
            {
                "image": base64.b64encode(open(image, "rb").read()).decode("utf-8"),
            }
        ]
    }


def create_face_verification_payload(image_1, image_2):
    return {
        "payloads": [
            {
                "image1": base64.b64encode(open(image_1, "rb").read()).decode("utf-8"),
                "image2": base64.b64encode(open(image_2, "rb").read()).decode("utf-8"),
            }
        ]
    }


def create_add_face_payload(image, user_id):
    return {
        "payloads": [
            {
                "image": base64.b64encode(open(image, "rb").read()).decode("utf-8"),
                "userId": user_id,
            }
        ]
    }


def draw_bounding_boxes(image, detected_faces: List[FaceSegment]):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for face in detected_faces:
        x, y, w, h = face.x, face.y, face.w, face.h
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    image = np.array(image)
    return image


def detect_face(image):
    image_path = os.path.join(TMP, "image.jpeg")
    export_image_using_pil(np.array(image), image_path)
    response = http_call(
        "http://localhost:8000/face-detect", payload=create_payload(image_path)
    )
    if "results" in response:
        results = response.get("results", [])
        if len(results) > 0:
            faces = [
                FaceSegment.from_json(face) for face in results[0].get("faces", [])
            ]
            image = draw_bounding_boxes(image, faces)
    if os.path.exists(image_path):
        os.remove(image_path)
    return response, image


def verify_face(image1, image2):
    image1_path = os.path.join(TMP, "image1.jpeg")
    export_image_using_pil(np.array(image1), image1_path)
    image2_path = os.path.join(TMP, "image2.jpeg")
    export_image_using_pil(np.array(image2), image2_path)
    response = http_call(
        "http://localhost:8000/verify",
        payload=create_face_verification_payload(image1_path, image2_path),
    )

    if "results" in response:
        results = response.get("results", [])
        if len(results) > 0:
            results = results[0].get("faces", {})
            if "image1" in results:
                image1 = draw_bounding_boxes(
                    image1, [FaceSegment.from_json(results["image1"])]
                )
            if "image2" in results:
                image2 = draw_bounding_boxes(
                    image2, [FaceSegment.from_json(results["image2"])]
                )

    if os.path.exists(image1_path):
        os.remove(image1_path)
    if os.path.exists(image2_path):
        os.remove(image2_path)
    return response, image1, image2


def recognize_face(image):
    image_path = os.path.join(TMP, "image.jpeg")
    export_image_using_pil(np.array(image), image_path)
    response = http_call(
        "http://localhost:8000/recognize", payload=create_payload(image_path)
    )
    if os.path.exists(image_path):
        os.remove(image_path)
    return response


def add_image(image, user_id):
    image_path = os.path.join(TMP, "image.jpeg")
    export_image_using_pil(np.array(image), image_path)
    response = http_call(
        "http://localhost:8000/add",
        payload=create_add_face_payload(image_path, user_id),
    )
    if os.path.exists(image_path):
        os.remove(image_path)
    return response


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            action_dropdown = gr.Dropdown(
                ["Face Detection", "Face Recognition", "Face Verification", "Add Face"],
                value="Face Detection",
            )

            @gr.render(inputs=action_dropdown)
            def render_action_input(text):
                if text == "Face Detection":
                    with gr.Row():
                        input_1 = gr.Image(height=350, label="Input Image")
                    with gr.Row():
                        output1 = gr.Image(interactive=True, height=350)
                    with gr.Row():
                        detection_button = gr.Button(value="submit")
                        clear_button = gr.ClearButton(
                            value="clear", components=[input_1, json_response, output1]
                        )
                    detection_button.click(
                        detect_face, [input_1], [json_response, output1]
                    )
                elif text == "Face Recognition":
                    with gr.Row():
                        with gr.Column():
                            input_1 = gr.Image(height=350, label="Input Image")
                    with gr.Row():
                        recognition_button = gr.Button(value="submit")
                        clear_button = gr.ClearButton(
                            value="clear", components=[input_1, json_response]
                        )
                    recognition_button.click(recognize_face, [input_1], [json_response])
                elif text == "Face Verification":
                    with gr.Group():
                        with gr.Row():
                            with gr.Column():
                                input_1 = gr.Image(height=350, label="Input Image 1")
                            with gr.Column():
                                input_2 = gr.Image(height=350, label="Input Image 2")
                    with gr.Group():
                        with gr.Row():
                            with gr.Column():
                                output_1 = gr.Image(
                                    interactive=True, height=350, label="Output Image 1"
                                )
                            with gr.Column():
                                output_2 = gr.Image(
                                    interactive=True, height=350, label="Output Image 2"
                                )
                    with gr.Row():
                        verification_button = gr.Button(value="submit")
                        clear_button = gr.ClearButton(
                            value="clear",
                            components=[
                                json_response,
                                input_1,
                                input_2,
                                output_1,
                                output_2,
                            ],
                        )
                    verification_button.click(
                        verify_face,
                        [input_1, input_2],
                        [json_response, output_1, output_2],
                    )
                elif text == "Add Face":
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                input_1 = gr.Image(height=350, label="Input Image 1")
                                user_id = gr.Text(label="User ID")
                    with gr.Row():
                        add_button = gr.Button(value="submit")
                        clear_button = gr.ClearButton(
                            value="clear", components=[json_response, input_1, user_id]
                        )
                    add_button.click(add_image, [input_1, user_id], [json_response])

        with gr.Column():
            json_response = gr.JSON(label="API Response")


demo.launch(debug=True, server_port=8001)
