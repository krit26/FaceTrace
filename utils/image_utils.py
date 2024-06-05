# Standard Imports
import os
import io
import base64
import hashlib
import logging
from typing import List, Union

# Third Party Imports
import cv2
import numpy as np
from PIL import Image as pilImage
from tensorflow.keras.preprocessing import image as tf_image

# Internal Imports


def load_image_using_pil(image: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image, str):
        if os.path.isfile(image):
            # reading an image
            image = pilImage.open(image)
        else:
            # base64 encoded image
            image_string = base64.b64decode(image)
            image = pilImage.open(io.BytesIO(image_string))
        # convert image to numpy array for further processing
        image = np.array(image)
        return image
    elif isinstance(image, np.ndarray):
        return image
    return image


def image_hash(image_path):
    with pilImage.open(image_path) as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format)
        img_bytes = img_byte_arr.getvalue()

        hash_object = hashlib.sha256()
        hash_object.update(img_bytes)
        return hash_object.hexdigest()


def expand_image_with_percentage(x, y, w, h, image, percentage):
    if percentage > 0:
        logging.info("Expand Percentage: {}".format(percentage))
        # Expand the facial region height and width by the provided percentage
        # ensuring that the expanded region stays within img.shape limits
        expanded_w = w + int(w * percentage / 100)
        expanded_h = h + int(h * percentage / 100)

        x = max(0, x - int((expanded_w - w) / 2))
        y = max(0, y - int((expanded_h - h) / 2))
        w = min(image.shape[1] - x, expanded_w)
        h = min(image.shape[0] - y, expanded_h)
        return x, y, w, h
    return x, y, w, h


def align_face(img, left_eye, right_eye):
    """
    Align a given image horizantally with respect to their left and right eye locations
    Args:
        img (np.ndarray): pre-loaded image with detected face
        left_eye (list or tuple): coordinates of left eye with respect to the person itself
        right_eye(list or tuple): coordinates of right eye with respect to the person itself
    Returns:
        img (np.ndarray): aligned facial image
    """
    # if eye could not be detected for the given image, return image itself
    if left_eye is None or right_eye is None:
        return img, 0

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0

    angle = float(
        np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
    )
    img = np.array(pilImage.fromarray(img).rotate(angle))
    return img, angle


def rotate_facial_area(facial_area, angle, size):
    """
    Rotate the facial area around its center.
    Inspried from the work of @UmutDeniz26 - github.com/serengil/retinaface/pull/80

    Args:
        facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
            x2 is equal to x1 + w1, and y2 is equal to y1 + h1
        angle (float): Angle of rotation in degrees. Its sign determines the direction of rotation.
                       Note that angles > 360 degrees are normalized to the range [0, 360).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        rotated_coordinates (tuple of int): Representing the new coordinates
            (x1, y1, x2, y2) or (x1, y1, x1+w1, y1+h1) of the rotated facial area.
    """

    # Normalize the witdh of the angle so we don't have to
    # worry about rotations greater than 360 degrees.
    # We workaround the quirky behavior of the modulo operator
    # for negative angle values.
    direction = 1 if angle >= 0 else -1
    angle = abs(angle) % 360
    if angle == 0:
        return facial_area

    # Angle in radians
    angle = angle * np.pi / 180

    height, weight = size

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + weight / 2
    y_new = y_new + height / 2

    # Calculate projected coordinates after alignment
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    # validate projected coordinates are in image's boundaries
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)

    return x1, y1, x2, y2


def resize_image(img, target_size):
    """
    Resize an image to expected size of a ml model with adding black pixels.
    Args:
        img (np.ndarray): pre-loaded image as numpy array
        target_size (tuple): input shape of ml model
    Returns:
        img (np.ndarray): resized input image
    """
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    # Put the base image in the middle of the padded image
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # make it 4-dimensional how ML models expect
    img = tf_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img
