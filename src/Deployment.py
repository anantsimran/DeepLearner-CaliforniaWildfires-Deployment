import streamlit as st
import pydeck as pdk
import datetime
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
import pandas as pd

from NN_loss import iou
from OurModel import getOurModel
from loadData import getTestData



flatTestX, flatTestY, dateArr = getTestData()

ourModel = getOurModel(flatTestX)

def persistenceModel(x):
	return scipy.ndimage.gaussian_filter(x, 1.7, output=np.float32)

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

st.sidebar.title("CS 274P Project: Predicting California Wildfires")


# st.sidebar.markdown("<h1 style='text-align: center; color: black;'>CS 274P Project: Predicting California Wildfires</h1>",
#                     unsafe_allow_html=True)

startDate = st.sidebar.date_input('Pick A Date', datetime.date(2013, 1, 1))
index = np.searchsorted(dateArr, pd.to_datetime(startDate))
vals = []
for i in range(index, index + 10):
	vals.append(dateArr[i])

date = st.sidebar.selectbox("Select the instance", vals)
index = np.searchsorted(dateArr, pd.to_datetime(date))

lat = flatTestX[index][18][0][0]
lon = flatTestX[index][19][0][0]
mapData = {
	"lat": [lat],
	"lon": [lon]
}


initial_view_state = pdk.ViewState(
	latitude=lat,
	longitude=lon,
	zoom=14,
	pitch=60,
	height =400,
	width = 1200
)

r = pdk.Deck(
	map_style='mapbox://styles/mapbox/satellite-streets-v9',
	initial_view_state=initial_view_state,
	layers=[
		pdk.Layer(
			'ScatterplotLayer',
			data=pd.DataFrame(mapData),
			get_position='[lon, lat]',
			get_color='[200, 30, 0, 160]',
			get_radius=300,
		),
	],
)

initial_view_state = pdk.ViewState(
	latitude=lat,
	longitude=lon,
	zoom=14,
	pitch=60,
	height =400,
	width = 1200
)
st.pydeck_chart(r)
r.initial_view_state = initial_view_state
r.update()


# plt.figure(figsize=(5, 5))
#
# plt.subplot(1,5,5)
# # plt.title('Fire at 0 hours')
# plt.imshow(flatTestX[index][30])
# plt.axis('off')
#
#
# st.pyplot()

ignitionFeatures = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21, 22, 23]
observedFeatures = [30, 31, 32, 33, 34]
yfeatures = [0]


allFeatures = [2,3,20,21,8,9,10,11,12,14,15,17,0,22,23,4,5,7,13,30,31,32,33,34]



modelTestIgnition = np.take(flatTestX, ignitionFeatures, axis=1)
modelTestObserved = np.take(flatTestX, observedFeatures, axis=1)
modelTestY = np.take(flatTestY, yfeatures, axis=1)

cnnTestX= np.take(flatTestX,allFeatures,axis=1)

for datapoint in modelTestIgnition:
	hr = (pd.to_datetime(datapoint[0][0][0]).hour-12)/24
	datapoint[0]=0*np.ones((30,30))





st.markdown("<h1 style='text-align: center; color: black;'>Observed Wildfires</h1>", unsafe_allow_html=True)


ourModel_pred = ourModel.predict(x=  [ np.reshape(modelTestObserved[index], (1, -1,30,30)), np.reshape(modelTestIgnition[index], (1, -1,30,30))])[0]
# cnn_pred = cnnModel.predict(x=  [ np.reshape(cnnTestX[index], (1, -1,30,30))])[0]

# cnn_pred_2 = cnnModel.predict(x=  [cnnTestX])
# st.text(np.sum(cnn_pred_2))


plt.figure(figsize=(12, 3))

for i in range(5):
	plt.subplot(1, 5, i + 1)
	plt.title(f'{-12 * (4 - i)} hours')
	plt.imshow(flatTestX[index][34 - i])
	plt.axis('off')

st.pyplot()

# st.markdown("<h1 style='text-align: center; color: black;'>Predicted Wildfires - Persistence Model</h1>",
#             unsafe_allow_html=True)
# st.markdown("<br>", unsafe_allow_html=True)

# plt.figure(figsize=(17,3))
#
plt.subplot(1,3,1)
plt.imshow(persistenceModel(flatTestX[index][30]))
plt.title('Persistance Model')
plt.axis('off')
#
plt.subplot(1, 3, 2)
plt.imshow(np.where(ourModel_pred[0]>0.9,1,0))
plt.title('Concatenated CNN Model')
plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(np.where(cnn_pred[0]>0.2,1,0))
# plt.title('One layer CNN Model')
# plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(flatTestY[index][0])
plt.title('Y (T = +12 hours)')
plt.axis('off')

st.pyplot()

# st.markdown("<h1 style='text-align: center; color: black;'>Map Representation</h1>", unsafe_allow_html=True)



# st.markdown("<h1 style='text-align: center; color: black;'>CS 274P Project: Predicting California Wildfires</h1>",
#             unsafe_allow_html=True)
st.markdown(
	"<h2 style='text-align: center; color: black;'>Wildfires in California have led to ecological and wildlife destruction in the recent past and this accelerates with climate change. While it is not possible to curb every wildfire, timely detection of wildfires can help in minimising the environmental loss. Hence, we use Deep Learning Techniques to predict the spread of these wildfires.</h2>",
	unsafe_allow_html=True)