import numpy as np
import tensorflow as tf
from keras import backend as K


def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(tf.cast(y_true_f,np.float32) * y_pred_f)
    return intersection / (K.sum(tf.cast(y_true_f,np.float32)) + K.sum(y_pred_f) - intersection)


def iou_loss(y_true, y_pred):
	return 1-iou(y_true, y_pred)