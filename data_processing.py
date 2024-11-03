import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import keras_cv
import tensorflow as tf
from constants import CLASSES, N_CLASSES
from sklearn.model_selection import GroupShuffleSplit
import time
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_cv.layers import Equalization


equalize = Equalization((0, 1), bins=256)

def load_image(img_path, img_label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, dtype=tf.float32)
    img = img / 255.0
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    img = (img - img_mean) / img_std
    return img, img_label


def augment_images(img, img_label):
    # img = tf.image.stateless_random_flip_left_right(img, seed=(2,3))
    img = equalize(img)
    return img, img_label


def get_input_data():
    images_paths = {os.path.basename(x): x for x in glob(os.path.join(os.getcwd(), "..", "dataset", "images_all", "*.png"))}
    
    train_path = os.path.join(os.getcwd(), "data", "data_labels", "train_list.txt")
    valid_path = os.path.join(os.getcwd(), "data", "data_labels", "val_list.txt")
    
    train_df = pd.read_csv(train_path, sep=" ", header=None)
    train_df = train_df[~train_df[0].isin(['00000004_000.png','00002846_012.png'])]
    valid_df = pd.read_csv(valid_path, sep=" ", header=None)
    

    train_df[0] = train_df[0].map(images_paths.get)
    train_img_paths = train_df[0]
    train_labels = train_df[list(train_df.columns[1:])]
    train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels))
    train_ds = train_ds.cache().shuffle(len(train_df), seed=42)
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(32)
    train_ds = train_ds.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE) 

    valid_df[0] = valid_df[0].map(images_paths.get)
    valid_img_paths = valid_df[0]
    valid_labels = valid_df[list(valid_df.columns[1:])]
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_img_paths, valid_labels))
    valid_ds = valid_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(32)
    valid_ds = valid_ds.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds

def get_test_data():
    def load_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, dtype=tf.float32) / 255.0
        return img
    
    images_paths = {os.path.basename(x): x for x in glob(os.path.join(os.getcwd(), "..", "dataset", "images_all", "*.png"))}
    
    test_path = os.path.join(os.getcwd(), "data", "data_labels", "test_list.txt")

    test_df = pd.read_csv(test_path, sep=" ", header=None)

    test_df[0] = test_df[0].map(images_paths.get)
    test_img_paths = test_df[0]
    test_ds = tf.data.Dataset.from_tensor_slices(test_img_paths)
    test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

    return test_df, test_ds


def normalize(img):
    # img = img / 255.0
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    img = (img - img_mean) / img_std
    return img

def histogram_equalization(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    # img = img / 255.0
    
    # if img_gray.dtype != np.uint8:
    #     img_gray = img_gray.astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img / 255.0
    return img

def image_complement(img):
    return (255 - img) / 255.0

def clahe(img, clip_limit=20, tile_grid_size=(8, 8)):
    img = cv2.resize(img, (224, 224))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)
    
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_gray)
    img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    img_clahe_rgb = img_clahe_rgb / 255.0
    return img_clahe_rgb

def bcet(image):
    # image = np.float32(image)
    # Calculate min, max, and mean intensity values
    # Calculate min, max, mean, and mean square sum of the intensity values
    Lmin = np.min(image)
    Lmax = np.max(image)
    Lmean = np.mean(image)
    LMssum = np.mean(image ** 2)  # Mean square sum of the input image

    # Desired output range and mean intensity
    Gmin, Gmax, Gmean = 0, 255, 110

    # Compute the 'b' coefficient for the BCET transformation
    bnum = (Lmax ** 2 * (Gmean - Gmin)) - (LMssum * (Gmax - Gmin)) + (Lmin ** 2 * (Gmax - Gmean))
    bden = 2 * (Lmax * (Gmean - Gmin) - Lmean * (Gmax - Gmin) + Lmin * (Gmax - Gmean))
    b = bnum / bden

    # Compute the 'a' and 'c' coefficients
    a = (Gmax - Gmin) / ((Lmax - Lmin) * (Lmax + Lmin - 2 * b))
    c = Gmin - a * (Lmin - b) ** 2

    # Apply the BCET parabolic transformation to each pixel in the image
    bcet_image = a * (image - b) ** 2 + c

    # Clip values to ensure they're within the valid range [0, 255]
    bcet_image = np.clip(bcet_image, Gmin, Gmax).astype(np.uint8)
    return bcet_image / 255.0

def gamma_correction(image, gamma=2.5):
    img_array = np.array(image).astype(np.float32) / 255.0 # Normalize to 0-1 range
    img_corrected = np.power(img_array, gamma)
    img_corrected = (img_corrected).astype(np.uint8)
    return img_corrected