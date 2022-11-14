import tensorflow as tf
import numpy as np

from keras.models import load_model
from metrics import jacc_loss, dice_coef, jacc_coef, acc, mean_IoU, sensitivity, specificity

def load_saved_model(name):
    return load_model(name, custom_objects={jacc_loss: jacc_loss, dice_coef: dice_coef, 
        jacc_coef: jacc_coef, acc: acc, mean_IoU: mean_IoU, sensitivity: sensitivity, specificity: specificity}, compile=False)

@tf.function
def read_image(img_path, image_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img / 255.
    return img

@tf.function
def read_mask(img_path, image_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, image_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img / 255.
    img = tf.round(img)
    return img

def MakeDataset(image_paths, label_paths, image_size, mask_size, length=None, batch_size=1):
    if length is None: length = len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths)).shuffle(length)
    return dataset.map(map_func=lambda x,y: (read_image(x, image_size), read_mask(y, mask_size))).batch(batch_size, drop_remainder=True).repeat()


import cv2
def apply_clahe(img, image_size, clahe):
    path = img.numpy().decode("utf-8")
    image = cv2.imread(path)
    image = cv2.resize(image, image_size)
    blue, green, red = cv2.split(image)

    red = np.asarray(red)
    green = np.asarray(green)
    blue = np.asarray(blue)

    red_clahe = clahe.apply(red)
    green_clahe = clahe.apply(green)
    blue_clahe = clahe.apply(blue)

    clahe_image = cv2.merge((red_clahe, green_clahe, blue_clahe))
    return np.asarray(clahe_image) / 255


def Clahe_Dataset(image_paths, label_paths, image_size, mask_size, length=None, batch_size=1):
    if length is None: length = len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths)).shuffle(length)
    cl = cv2.createCLAHE(3, (3, 3))
    def wrapper(image): 
      return tf.py_function(lambda x: apply_clahe(x, image_size, cl), [image], tf.float32)
    return dataset.map(map_func=lambda x,y: (wrapper(x), read_mask(y, mask_size))).batch(batch_size, drop_remainder=True).repeat()


def laplacian(img, image_size):
    path = img.numpy().decode("utf-8")
    image = cv2.imread(path)
    image = cv2.resize(image, image_size)
    blue, green, red = cv2.split(image)

    abbsLaplace_red = cv2.Laplacian(red, -2)
    abbsLaplace_green = cv2.Laplacian(green, -2)
    abbsLaplace_blue = cv2.Laplacian(blue, -2)

    r = cv2.subtract(red, abbsLaplace_red)
    g = cv2.subtract(green, abbsLaplace_green)
    b = cv2.subtract(blue, abbsLaplace_blue)
    image = cv2.merge((r, g, b))
    return np.asarray(image) / 255

def Laplacian_Dataset(image_paths, label_paths, image_size, mask_size, length=None, batch_size=1):
    if length is None: length = len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths)).shuffle(length)
    def wrapper(image): 
      return tf.py_function(lambda x: laplacian(x, image_size), [image], tf.float32)
    return dataset.map(map_func=lambda x,y: (wrapper(x), read_mask(y, mask_size))).batch(batch_size, drop_remainder=True).repeat()
