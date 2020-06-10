import streamlit as st
import pydeck as pdk
import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

TEST_DATASET_PATH = 'test.hdf5'

@st.cache
def getTestData():
	def getDataDict(DatasetPath):
		with h5py.File(DatasetPath, 'r') as f:
			data = {}
			for k in list(f):
				data[k] = f[k][:]
			return data

	test_data = getDataDict(TEST_DATASET_PATH)


	def transformDateTime(datetime):
		ret = np.asarray(([[x]*30 for x in datetime]))
		return ret


	def transformLandCover(landCover):
		nanConvert = {
			0: 0,
			1: 0,
			2: 9,
			3: 0,
			4: 235,
			5: 0,
			6: 0,
			16: 0
		}
		ret = []

		for datapoint in landCover:
			for i in range(17):
				if i in nanConvert.keys():
					datapoint[i][np.isnan(datapoint[i])] = nanConvert[i]
				if (i == 1):
					datapoint[i] /= 14.0
				if (i == 2):
					datapoint[i] = 1 - (datapoint[i] - 9) / 30.0
				if (i == 3):
					datapoint[i] = 1 - (datapoint[i] - 32) / 70.0
				if (i == 4):
					datapoint[i] = 1 - (datapoint[i] - 235) / 340.0
				if (i == 5):
					datapoint[i] = norm(1250, 564).pdf(datapoint[i])
			ret.append(datapoint)
		return np.asarray(ret)


	def transformLatAndLong(val):
		ret = np.asarray([x * np.ones((1, 30, 30)) for x in val])
		return ret


	# TODO : define temperature according to datetime average
	def transformMet(met, date):
		nanConvert = {
			1: 26,
			2: 0,
			3: 0,
			4: 0,
		}
		met0 = []
		met1 = []
		index = 0
		for datapoint in met:
			for i in range(5):
				if (i == 0):
					if pd.to_datetime(date[index]).hour > 12:
						datapoint[1][i][np.isnan(datapoint[1][i])] = 290
						datapoint[0][i][np.isnan(datapoint[0][i])] = 301.91
						datapoint[0][i] = np.tanh(datapoint[0][i] - 301.91)
						datapoint[1][i] = np.tanh(datapoint[1][i] - 290)
					else:
						datapoint[1][i][np.isnan(datapoint[1][i])] = 302.54
						datapoint[0][i][np.isnan(datapoint[0][i])] = 287.56
						datapoint[0][i] = np.tanh(datapoint[0][i] - 287.56)
						datapoint[1][i] = np.tanh(datapoint[1][i] - 302.54)
				else:
					datapoint[0][i][np.isnan(datapoint[0][i])] = nanConvert[i]
					datapoint[1][i][np.isnan(datapoint[1][i])] = nanConvert[i]
				if (i == 1):
					datapoint[0][i] = 1 - sigmoid(datapoint[0][i] - 26)
					datapoint[1][i] = 1 - sigmoid(datapoint[1][i] - 26)
				if (i == 2):
					datapoint[0][i] = np.tanh(datapoint[0][i] - 0.4232)
					datapoint[1][i] = np.tanh(datapoint[1][i] - 1.4365)
				if (i == 3):
					datapoint[0][i] = np.tanh(datapoint[0][i] + 0.0854)
					datapoint[1][i] = np.tanh(datapoint[1][i] - 0.495)
			met0.append(datapoint[0])
			met1.append(datapoint[1])
			index += 1
		return np.asarray(met0), np.asarray(met1)


	def transformFire(fire):
		return np.asarray(fire)


	# transform all of them into dict of 3d np arrays.
	# Augmentation step must take place after this.
	# Can store this in h5py file after this.
	def transformAndClean(data):
		X = {}
		Y = {}
		X['datetime'] = transformDateTime(data['datetime'])
		X['landCover'] = transformLandCover(data['land_cover'])
		X['latitude'] = transformLatAndLong(data['latitude'])
		X['longitude'] = transformLatAndLong(data['longitude'])
		X['met0'], X['met1'] = transformMet(data['meteorology'], data['datetime'])
		X['observed'] = transformFire(data['observed'])
		Y['target'] = transformFire(data['target'])
		return X, Y

	return transformAndClean(test_data)
testX,testY = getTestData()

st.title("Junaid is gaandu")