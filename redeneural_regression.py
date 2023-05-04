import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

TEST_SIZE = 0.3


def loadData():
    data = pd.read_csv('treino_sinais_vitais_com_label.txt', sep=',', header=None)

    data.columns = ["index", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classificação"]

    data.drop(columns=['index', 'pSist', 'pDiast'], inplace=True)

    explicadores = data[['qPA', 'pulso', 'respiração']]

    target = data[['classificação']]

    print('Data shape:', data.shape)

    return explicadores, target


explicadores, target = loadData()

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=TEST_SIZE)

# Define the model architecture
model = Sequential()
# cada linha eh uma camada, o primeiro parametro eh a quantidade de nos
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the data
model.fit(data_train, target_train, epochs=50, batch_size=16)

# Evaluate the model on some test data
loss = model.evaluate(data_test, target_test)

print('Test loss:', loss)

##predictions = model.predict(data_test, batch_size=128)

