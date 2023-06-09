import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

TEST_SIZE = 0.3
EPOCHS = 200
LEARNING_RATE = 0.01
ACTV_FUNC = 'relu'
# OPTIMIZER = SGD(learning_rate=LEARNING_RATE)
OPTIMIZER = 'adam'


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
model.add(Dense(32, input_dim=3, activation=ACTV_FUNC))
model.add(Dense(50, activation=ACTV_FUNC))
model.add(Dense(50, activation=ACTV_FUNC))
model.add(Dense(50, activation=ACTV_FUNC))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=OPTIMIZER)

# Fit the model to the data
model.fit(data_train, target_train, epochs=EPOCHS)

# Predict on test set
y_pred = model.predict(data_test)

MSE = mean_squared_error(target_test, y_pred)
RMSE = mean_squared_error(target_test, y_pred, squared=False)
MAE = mean_absolute_error(target_test, y_pred)

print("MSE (loss):", MSE)
print('RMSE:', RMSE)
print("MAE:", MAE)

# predictions = model.predict(data_test, batch_size=128)
# Salvar resultado para futura comparacao
file_object = open('results/resultsRNReg.txt', 'a')
file_object.write(f'epochs: {EPOCHS}, layers: {len(model.layers)}, '
                  # f'LEARNING_RATE: {LEARNING_RATE}, '
                  f'test_size: {TEST_SIZE}, ACTV_FUNC: {ACTV_FUNC}, MSE %: {MSE}, RMSE %: {RMSE}, MAE %: {MAE},\n')
file_object.close()
