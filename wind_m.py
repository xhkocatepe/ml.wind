# -*- coding: utf-8 -*-
"""
 Dil olarak Tensorflow , Theano Backendleri kullanıldı, 
 en son çalışan infrastructure olarak  Theano Backend çalışıyor.
 Keras üzerinden geliştirme yapılıyor. 
 Theano low level bir kütüphane olduğundan Keras gibi wrappler kullanılıyor.
 
 Ölçeklenebilirliği kabettim.
 
 Thu June 1  2018

@author: Bayram Hakan Kocatepe
"""
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error

def safe_mape(actual_y, prediction_y):
    # MAPE yi hesapladık
    diff = numpy.absolute((actual_y - prediction_y) ) / actual_y # numpy.clip(numpy.absolute(actual_y), 1., None))
    return 100. * numpy.mean(diff)

# matrise çevrime işlemi yapıldı. look_back dinamik olarak çalışıyor.
    
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

numpy.random.seed(7)

# Data setimiz yükleniyor.
dataframe = read_csv('AMASRA_13-14.csv', usecols=[1], engine='python')
#dataframe = read_csv('AMASRA_13-14.csv', usecols=[0], engine='python')
#dataframe = read_csv('AMASRA_13-14.csv', usecols=[0], engine='python')

dataset = dataframe.values
dataset = dataset.astype('float32')


# test ve train setler belirleniyor 2013 ve 2014
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# X ve Y yi reshape yapıyoruz girilen değere göre. 3 geriye dönük array.
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Input da olan değişkenler [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# Sonunda LSTM ağını yaratıyoruz.
#Iki katmanlı bir ağ oluşturuyoruz.
#1.ağ Inputumuza ait bir ağ 4 yapay sinir hücresi var.
#2.ağ output tek yapay sinir hücresi var oda sonuç.
#hata tipimizi compile ederken mean_absolute_error seçtik.
#epoch sayısı kaç kere train edeceği 20 kere train ediyor tüm datayı okuyor.
#batch_size kaçar kaçar datayı alıp okumaya çalıştığı bir step de 24 satır alıyor.
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=24, verbose=2)
# tahminleri model.predict ile yapıyoruz.

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# hatayı hesaplıyoruz.
trainY = [trainY]
testY = [testY]
trainScore = mean_absolute_error(trainY[0], trainPredict[:,0])

print('Train Score: %.2f Mape' % (trainScore))
testScore = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.2f Mape' % (testScore))

print('Yuzdelik olarak MAPE : %.2f ' % safe_mape(trainScore,testScore) )

# Train tahminlerini belirleniyor.
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# Test tahminlerini belirleniyor.
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline ve predictionlar çiziliyor.
#plt.plot_date(x=dataframe.date.values , y =raw_)
    
plt.title("Amasra")
plt.plot(dataset[26:50])
plt.plot(testPredict[24:48])
plt.legend(('Real','Test'))

plt.show()