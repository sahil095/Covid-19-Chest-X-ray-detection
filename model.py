import numpy as np
import pandas as pd
import cv2
import PIL
import gc
import psutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from math import ceil
import math
import sys
import gc
import os
import random
from shutil import copyfile
import datetime
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_cv.layers import Equalization
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from constants import N_CLASSES, MODEL_DENSENET, MODEL_EFFICIENTNET, MODEL_INCEPTION, MODEL_VGG19
from tensorflow.keras.applications import DenseNet121, ResNet152, EfficientNetB0
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from data_processing import clahe, histogram_equalization, image_complement, bcet, gamma_correction


def get_model(base_model_type):
    if base_model_type == MODEL_DENSENET:
        base_model = DenseNet121(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
            # pooling="avg",
        )
    elif base_model_type == MODEL_INCEPTION:
        base_model = InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(224,224,3),
            # pooling="avg",
        )
    elif base_model_type == MODEL_VGG19:
        base_model = VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=(224,224,3),
            # pooling="max",
        )
    else:
        raise Exception("Unsupported model type")
    
    # print(base_model.summary())
    for layer in base_model.layers:
        layer.trainable=False
    
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    pred_layer = layers.Dense(N_CLASSES, activation="softmax", name="predictions")(x)
    _model = keras.models.Model(inputs=base_model.input, outputs=pred_layer)
    _model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=['acc'])
    
    return _model

def get_callbacks(model_name, batch_size, enhancement):

    callbacks_list = []
    # model_logs = model_name + '_model_logs_' + str(datetime.datetime.now())
    model_logs = model_name + batch_size + enhancement
    os.mkdir(model_logs)
    csv_logger_path = os.path.join(os.getcwd(), model_logs, "TF_training_logs.csv")
    callbacks_list.append(keras.callbacks.CSVLogger(csv_logger_path))

    # dafuq is max used with val_loss?
    model_ckpt_path = os.path.join(os.getcwd(), model_logs, "TF-Model-Checkpoint")
    callbacks_list.append(keras.callbacks.ModelCheckpoint(filepath=model_ckpt_path, monitor='val_loss', mode='min', save_best_only=True))

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=3, min_lr=1e-6, verbose=0)
    callbacks_list.append(reduce_lr)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, mode='auto', restore_best_weights=True)
    callbacks_list.append(early_stop)

    return callbacks_list



if __name__ == "__main__":
    
    # Use only to split the dataset once
    BASE_PATH = "/home/ssehg1@cfreg.local/covid_detection/COVID-19_Radiography_Dataset"
    INPUT_SHAPE = 224
    BATCH_SiZE = [16, 32]
    TRAINING_BATCH_SIZE = BATCH_SiZE
    VALIDATION_BATCH_SIZE = BATCH_SiZE
    TRAINING_DIR = BASE_PATH + "/research_data/train"
    # [clahe, histogram_equalization, image_complement
    image_enhancements = [clahe, histogram_equalization, image_complement, bcet]
    for batch in BATCH_SiZE:
        for enhancement in image_enhancements:
            train_datagen = ImageDataGenerator(
                            # rescale = 1.0/255.,
                            # rotation_range=15,
                            # width_shift_range=0.1,
                            # height_shift_range=0.1,
                            # shear_range=0.2,
                            # zoom_range=0.2,
                            # horizontal_flip=True,
                            # fill_mode='reflect',
                            preprocessing_function=enhancement
                        )
            train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size = batch, 
                                                                target_size = (INPUT_SHAPE, INPUT_SHAPE), class_mode = "categorical", shuffle=False)

            VAL_DIR = BASE_PATH + "/research_data/val"
            val_datagen = ImageDataGenerator(
                            # rescale = 1.0/255.,
                            # rotation_range=40,
                            # width_shift_range=0.25,
                            # height_shift_range=0.25,
                            # shear_range=0.3,
                            # zoom_range=0.3,
                            # horizontal_flip=True,
                            # fill_mode='nearest',
                            preprocessing_function=enhancement
                        )


            val_generator = val_datagen.flow_from_directory(VAL_DIR, batch_size = batch,
                                                            target_size = (INPUT_SHAPE, INPUT_SHAPE), class_mode = "categorical", shuffle=False)
        
        
            gc.collect()

            keras.backend.clear_session()
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(physical_devices, 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            strategy = tf.distribute.MirroredStrategy()
            
            with strategy.scope():
                
                start_time = time.time()
                # for model_name in [MODEL_DENSENET, MODEL_INCEPTION, MODEL_VGG19]
                
                model = get_model(MODEL_DENSENET)
                callbacks = get_callbacks(MODEL_DENSENET, str(batch), enhancement.__name__)
                
                steps_per_epoch = train_generator.n // train_generator.batch_size
                validation_steps = val_generator.n // val_generator.batch_size
                
                history = model.fit(train_generator, epochs=10, steps_per_epoch = steps_per_epoch, validation_data = val_generator,
                                            validation_steps = validation_steps, verbose=1)
                
                for layer in model.layers:
                    layer.trainable = True
                
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6), loss="categorical_crossentropy", metrics=['acc'])

                history = model.fit(train_generator, epochs=20, steps_per_epoch = steps_per_epoch, validation_data = val_generator,
                                            validation_steps = validation_steps, callbacks=callbacks, verbose=1)
                print("--- %s seconds ---" % (time.time() - start_time))

        # results = model.evaluate(test_generator)
        # print(results)
        # hist_df = pd.DataFrame(history.history)
        # hist_df.to_csv('history.csv', index=False)