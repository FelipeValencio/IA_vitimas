import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

TEST_SIZE = 0.3
EPOCHS = 1000
LEARNING_RATE = 0.01
ACTV_FUNC = 'relu'


def loadData():
    data = pd.read_csv('treino_sinais_vitais_sem_label.txt', sep=',', header=None)

    data.columns = ["index", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade"]

    data.drop(columns=['index', 'pSist', 'pDiast'], inplace=True)

    explicadores = data[['qPA', 'pulso', 'respiração']]

    target = data[['gravidade']]

    print('Data shape:', data.shape)

    return explicadores, target


explicadores, target = loadData()

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=TEST_SIZE)

# Define the model architecture
model = Sequential()
# cada linha eh uma camada, o primeiro parametro eh a quantidade de nos
model.add(Dense(1024, activation=ACTV_FUNC))
model.add(Dense(512, activation=ACTV_FUNC))
model.add(Dense(256, activation=ACTV_FUNC))
model.add(Dense(128, activation=ACTV_FUNC))
model.add(Dense(64, activation=ACTV_FUNC))
model.add(Dense(32, activation=ACTV_FUNC))
model.add(Dense(16, activation=ACTV_FUNC))
model.add(Dense(8, activation=ACTV_FUNC))
model.add(Dense(1, activation='linear'))

# Compile the model
opt = SGD(learning_rate=LEARNING_RATE)
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the data
model.fit(data_train, target_train, epochs=EPOCHS)

# Evaluate the model on some test data
loss = model.evaluate(data_test, target_test)

print('Test loss:', loss)

# predictions = model.predict(data_test, batch_size=128)
# Salvar resultado para futura comparacao
file_object = open('results/resultsRNReg.txt', 'a')
file_object.write(f'epochs: {EPOCHS}, layers: {len(model.layers)}, '
                 # f'LEARNING_RATE: {LEARNING_RATE}, '
                  f'test_size: {TEST_SIZE}, ACTV_FUNC: {ACTV_FUNC}, loss %: {loss}\n')
file_object.close()
