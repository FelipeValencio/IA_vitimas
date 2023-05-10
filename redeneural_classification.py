import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

TEST_SIZE = 0.3
EPOCHS = 200
LEARNING_RATE = 0.03
ACTV_FUNC = 'relu'
# OPTIMIZER = SGD(learning_rate=LEARNING_RATE)
OPTIMIZER = 'adam'


def loadData():
    data = pd.read_csv('treino_sinais_vitais_com_label.txt', sep=',', header=None)

    data.columns = ["index", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classificação"]

    data.drop(columns=['index', 'pSist', 'pDiast'], inplace=True)

    explicadores = data[['qPA', 'pulso', 'respiração']]

    target = data[['classificação']] - 1

    print('Data shape:', data.shape)

    return explicadores, target


explicadores, target = loadData()

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=TEST_SIZE)

scaler = StandardScaler().fit(data_train)
N_data_train = scaler.transform(data_train)
N_data_test = scaler.transform(data_test)

# Define the model architecture
model = Sequential()
# cada linha eh uma camada, o primeiro parametro eh a quantidade de nos
model.add(Dense(32, input_dim=3, activation=ACTV_FUNC))
model.add(Dense(50, activation=ACTV_FUNC))
model.add(Dense(50, activation=ACTV_FUNC))
model.add(Dense(50, activation=ACTV_FUNC))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# Fit the model to the data
model.fit(N_data_train, target_train, epochs=EPOCHS)

# Evaluate the model on some test data
y_pred = model.predict(N_data_test)

y_pred = np.argmax(y_pred, axis=-1)

plotcm = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(target_test, y_pred),
    display_labels=['C1', 'C2', 'C3', 'C4']
)

plotcm.plot()

class_report = classification_report(target_test, y_pred)

print(class_report)

# Salvar resultado para futura comparacao
file_object = open('results/resultsRNClass.txt', 'a')
file_object.write(f'epochs: {EPOCHS}, layers: {len(model.layers)}, '
                  f'LEARNING_RATE: {LEARNING_RATE}, '
                  f'test_size: {TEST_SIZE}, ACTV_FUNC: {ACTV_FUNC}\n')
file_object.write(class_report + '\n')
file_object.close()

plt.show()
