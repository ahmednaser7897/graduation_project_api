import numpy as np
from PIL import Image as im
import cv2
from io import BytesIO
import base64
import tensorflow.keras as keras
import tensorflow as tf

import torch

def concat_imgs(A, B):
    img = tf.concat([A, B], axis=2)
    # Images.append(img)
    return img


def map_filename_to_image_and_mask(image_A, image_B, height=128, width=128):
    # Resize image and segmentation mask
    image_A = tf.image.resize(image_A, (height, width,))
    image_B = tf.image.resize(image_B, (height, width,))
    print("1 : images.shape")
    print(image_A.shape)
    print(image_B.shape)
    image_A = tf.reshape(image_A, (height, width, 3,))
    image_B = tf.reshape(image_B, (height, width, 3,))

    # print(annotation)
    # Normalize pixels in the input image in range (-1,1)
    image_A = image_A / 255
    image_A = (image_A - tf.reduce_min(image_A)) / (tf.reduce_max(image_A) - tf.reduce_min(image_A))
    image_A = 2 * image_A - 1
    # image -= 1

    # Normalize pixels in the input image in range (-1,1)
    image_B = image_B / 255
    image_B = (image_B - tf.reduce_min(image_B)) / (tf.reduce_max(image_B) - tf.reduce_min(image_B))
    image_B = 2 * image_B - 1
    # image -= 1

    # concat
    image = concat_imgs(image_A, image_B)

    image = np.array(image)
    image.shape = (1, 128, 128, 6)
    print("2 : predictions.shape")
    print(image.shape)
    return image


def weighted_binary_crossentropy(zero_weight, one_weight):
    def weighted_binary_crossentropy(y_true, y_pred):
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return tf.keras.backend.mean(weighted_b_ce)

    return weighted_binary_crossentropy


def load_old_model():
    # Load the model from the file
    loaded_model = keras.models.load_model('old_model.h5', compile=False)
    loaded_model.compile()
    # Load the model from the file
    return loaded_model;


def load_new_model():
    # Load the model from the file
    loaded_model = torch.load('UNET_LossWBC.h5', )
    return loaded_model;


def get_frame_img(image):
    return cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)


def get_encoded_img(image):
    return base64.b64encode(image.read()).decode('utf-8')


def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str


def get_img_from_array(array):
    return im.fromarray(np.uint8(array * 255))


def change_threshold(predictions):
    print(predictions[:, :, :, :].max())
    print(predictions[:, :, :, :].min())
    threshold = 0.75
    predictions = np.where(predictions < threshold, 0.0, 1.0)
    print(np.unique(predictions))
    print(predictions[:, :, :, :].max())
    print(predictions[:, :, :, :].min())
    return predictions