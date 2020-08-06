import streamlit as st
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dropout
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from NN_loss import iou_loss, iou


@st.cache
def getOurModel(data):
	m = keras.models.load_model('concatenate_cnn.h5', custom_objects={'iou_loss':iou_loss, 'iou':iou, 'adam': Adam})
	# m.load_weights('two.h5')
	m.compile(optimizer=Adam(lr = 0.0002), loss=iou_loss, metrics = [iou])
	return m
