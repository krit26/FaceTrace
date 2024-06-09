# Standard Imports
from typing import Union, List

# Third Part Imports
import numpy as np

# Internal Imports
from utils.utils import timeit
from models.model_holder import ModelHolder
from utils.image_utils import load_image_using_pil
from models.detectors import AbstractDetectionModel
from structures.image import FaceSegment, DetectedFace
from utils.image_utils import (
    expand_image_with_percentage,
    align_face,
    rotate_facial_area,
)


@timeit
def detection(
    images: Union[str, np.ndarray, List[str], List[np.ndarray]],
    model_name,
    align=False,
    expand_percentage=0,
) -> List[List[DetectedFace]]:

    try:
        detection_model: AbstractDetectionModel = ModelHolder.get_or_load_model(
            model_name
        )
    except Exception as e:
        raise Exception(f"Error in loading {model_name} face detection model: {str(e)}")

    if not isinstance(images, list):
        images = [images]

    images = [load_image_using_pil(image) for image in images]

    try:
        model_output: List[List[FaceSegment]] = detection_model.predict(images)
    except Exception as e:
        raise Exception(f"Error in face detection model: {str(e)}")

    outputs = []
    for idx, detected_faces in enumerate(model_output):
        if len(detected_faces) == 0:
            outputs.append([])
            continue
        faces = []
        image = images[idx]
        for face in detected_faces:
            x, y, w, h = expand_image_with_percentage(
                face.x, face.y, face.w, face.h, image, expand_percentage
            )

            detected_face = image[int(y) : int(y + h), int(x) : int(x + w)]
            if align:
                aligned_img, angle = align_face(
                    img=image, left_eye=face.left_eye, right_eye=face.right_eye
                )
                rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
                    facial_area=(x, y, x + w, y + h),
                    angle=angle,
                    size=(image.shape[0], image.shape[1]),
                )
                detected_face = aligned_img[
                    int(rotated_y1) : int(rotated_y2),
                    int(rotated_x1) : int(rotated_x2),
                ]
            detected_face = detected_face / 255
            faces.append(
                DetectedFace(
                    model_name=model_name,
                    image=detected_face,
                    facial_segments=FaceSegment(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        left_eye=face.left_eye,
                        right_eye=face.right_eye,
                        confidence=face.confidence,
                    ),
                    alignment=align,
                    expand_percentage=expand_percentage,
                )
            )
        outputs.append(faces)
    return outputs
