import streamlit as st
import pydeck as pdk
import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime
import matplotlib.pyplot as plt


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
		ret = np.asarray(([[x] * 30 for x in datetime]))
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

	testX, testY = transformAndClean(test_data)
	startDictionary = {
		'datetime': 0,
		'landCover': 1,
		'latitude': 18,
		'longitude': 19,
		'met0': 20,
		'met1': 25,
		'observed': 30,
		'target': 0
	}

	lengthDictionary = {
		'datetime': 1,
		'landCover': 17,
		'latitude': 1,
		'longitude': 1,
		'met0': 5,
		'met1': 5,
		'observed': 5,
		'target': 2
	}

	def flattenData(data):
		length = 0
		for key, value in data.items():
			length += value.shape[1]
			n = value.shape[0]
		ret = np.zeros((n, length, 30, 30))
		for key, arr in data.items():
			for index, datapoint in enumerate(arr):
				ret[index][startDictionary[key]: startDictionary[key] + lengthDictionary[key]][:][:] = datapoint
		return ret;

	# flatX = flattenData(X)
	# flatY = flattenData(Y)
	flatTestX = flattenData(testX)
	flatTestY = flattenData(testY)

	datetime = np.asarray([pd.to_datetime(x[0][0][0]) for x in flatTestX])
	return flatTestX, flatTestY, datetime


flatTestX, flatTestY, dateArr = getTestData()

masterdict = {
	0: 'datetime',
	1: 'Aspect',
	2: 'Canopy Built Density',
	3: 'Canopy Base Height',
	4: 'Canopy Cover',
	5: 'Canopy Height',
	6: 'Elevation',
	7: 'Slope',
	8: 'No Data',
	9: 'Sparse',
	10: 'Tree',
	11: 'Shrub',
	12: 'Herb',
	13: 'Water',
	14: 'Barren',
	15: 'Developed',
	16: 'Snow-Ice',
	17: 'Agriculture',
	18: 'latitude',
	19: 'longitude',
	20: 'temp 0 ',
	21: 'humidity 0 ',
	22: 'uwind 0 ',
	23: 'vwind 0 ',
	24: 'prec 0 ',
	25: 'temp +12 ',
	26: 'humidity +12 ',
	27: 'uwind +12 ',
	28: 'vwind +12 ',
	29: 'prec +12 ',
	30: 'observed 0',
	31: 'observed -12',
	32: 'observed -24',
	33: 'observed -36',
	34: 'observed -48'
}

def _max_width_():
	max_width_str = f"max-width: 1900px;"
	st.markdown(
		f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
		unsafe_allow_html=True,
	)


_max_width_()

startDate = st.sidebar.date_input('Pick A Date', datetime.date(2013, 1, 1))
index = np.searchsorted(dateArr, pd.to_datetime(startDate))
vals = []
for i in range(index, index + 10):
	vals.append(dateArr[i])

date = st.sidebar.selectbox("Select the instance", vals)
index = np.searchsorted(dateArr, pd.to_datetime(date))

# lat = np.asarray( [x[18][0][0] for x in flatTestX ] )
# long = np.asarray( [x[19][0][0] for x in flatTestX ] )



# index = 1170


for i in range(5):
	plt.rcParams['figure.figsize'] = (18, 18)
	plt.subplot(2, 5, i + 1)
	plt.title(f'{-12 * (4 - i)} hours')
	plt.imshow(flatTestX[index][34 - i])
	plt.axis('off')



lat = flatTestX[index][18][0][0]
lon = flatTestX[index][19][0][0]
mapData= {
	"lat": [lat] ,
	"lon": [lon]
}
initial_view_state = pdk.ViewState(
		latitude = lat,
		longitude = lon,
		zoom = 12,
		pitch = 50,
	    height=350,
		width=500,
)
r = pdk.Deck(
	map_style = 'mapbox://styles/mapbox/light-v9',
	initial_view_state = initial_view_state,
	layers = [
		pdk.Layer(
			'ScatterplotLayer',
			data = pd.DataFrame(mapData),
			get_position = '[lon, lat]',
			get_color = '[200, 30, 0, 160]',
			get_radius = 180,
		),
	],
)
initial_view_state = pdk.ViewState(
		latitude = lat,
		longitude = lon,
		zoom = 12,
		pitch = 50,
	    height=350,
		width=500,
	)
st.pydeck_chart(r)
r.initial_view_state= initial_view_state
r.update()



st.pyplot()
# st.title(index)
