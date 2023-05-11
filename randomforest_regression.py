from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd

MAX_DEPTH = 20
TEST_SIZE = 0.3
NUM_TREES = 100


def loadData():
    data = pd.read_csv('treino_sinais_vitais_sem_label.txt', sep=',', header=None)

    data.columns = ["index", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade"]

    data.drop(columns=['index', 'pSist', 'pDiast'], inplace=True)

    explicadores = data[['qPA', 'pulso', 'respiração']]

    target = data[['gravidade']]

    print('Data shape:', data.shape)

    return explicadores, target

# Load dataset
explicadores, target = loadData()

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=TEST_SIZE)

# Train the Decision Tree using ID3 algorithm
model = RandomForestRegressor(n_estimators=NUM_TREES, max_depth=MAX_DEPTH, random_state=42)
model.fit(data_train, target_train.values.ravel())

# Predict on test set
y_pred = model.predict(data_test)

# accuracy = model.score(data_test, target_test)

# Métricas de avaliação
MSE = mean_squared_error(target_test, y_pred)
RMSE = mean_squared_error(target_test, y_pred, squared=False)
MAE = mean_absolute_error(target_test, y_pred)

print("MSE:", MSE)
print('RMSE:', RMSE)
print("MAE:", MAE)

# Salvar resultado para futura comparacao
file_object = open('results/resultsRFReg.txt', 'a')
file_object.write(f'MSE: {MSE}, RMSE: {RMSE},'
                  f' MAE: {MAE}\n')
file_object.close()
