import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
os.environ["KERAS_BACKEND"] = "tensorflow"
# TODO: Move me to a dvc pipeline
import datetime
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
from keras_cv.layers import Equalization
from data_processing import clahe, histogram_equalization, image_complement, bcet, gamma_correction


def compute_aucs(predictions, labels):
    num_classes = predictions.shape[1]
    aucs = np.zeros(num_classes)
    for i in range(num_classes):
        aucs[i] = roc_auc_score(labels[:, i], predictions[:, i])
    return aucs


equalize = Equalization((0, 255))
INPUT_SHAPE = 224
VALIDATION_BATCH_SIZE = 16
BASE_PATH = "/home/ssehg1@cfreg.local/covid_detection/COVID-19_Radiography_Dataset"
TEST_DIR = BASE_PATH + "/research_data/test"

# test_datagen = ImageDataGenerator(rescale = 1.0/255., preprocessing_function=equalize)
test_datagen = ImageDataGenerator(
                                # rescale = 1.0/255.,
                                preprocessing_function=bcet

                            )
test_generator = test_datagen.flow_from_directory(TEST_DIR, batch_size = VALIDATION_BATCH_SIZE,
                                                target_size = (INPUT_SHAPE, INPUT_SHAPE), class_mode = "categorical", shuffle=False)
test_labels = test_generator.labels.reshape(400, 1)


def load_model():
    model_name = 'densenet_models/masked_image_results/densenet_model_logs_2024-10-18 19:47:23.164578'
    # new_model = tf.keras.models.load_model('./densenet_models/masked_image_results/' + model_name + '/TF-Model-Checkpoint/')
    new_model = tf.keras.models.load_model('./densenet16bcet/TF-Model-Checkpoint/')
    return new_model


if __name__ == "__main__":
    keras.backend.clear_session()
    model = load_model()
    # print(model.summary())
    pred = model.predict(test_generator)
    y_pred = np.argmax(pred, axis=1).reshape(400, 1)

    # Accuracy Score
    accuracy = accuracy_score(test_labels, y_pred)
    print('Accuracy: ', accuracy)

    # Confusion Matrix
    print('Confusion Matrix: ', confusion_matrix(test_labels, y_pred))

    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    print('True Neg', tn)
    print('False Pos', fp)
    print('False Neg', fn)
    print('True Pos', tp)

    # Classification Report
    class_report = classification_report(test_labels, y_pred)
    print('Classification Report: ', class_report)

    # AUC Scores
    print('AUC Scores:', compute_aucs(y_pred, test_labels))

    # np.savez('pred_' + str(datetime.datetime.now()) + '.npz', pred)