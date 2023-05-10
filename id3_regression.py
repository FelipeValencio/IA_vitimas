from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

TEST_SIZE = 0.3
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 4


def loadData():
    data = pd.read_csv('treino_sinais_vitais_com_label.txt', sep=',', header=None)

    data.columns = ["index", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classificação"]

    data.drop(columns=['index', 'pSist', 'pDiast'], inplace=True)

    explicadores = data[['qPA', 'pulso', 'respiração']]

    target = data[['gravidade']]

    return explicadores, target


# Load dataset
explicadores, target = loadData()

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=TEST_SIZE)

# Train the Decision Tree using ID3 algorithm
tree = DecisionTreeRegressor(random_state=42, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
                             min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES)
tree.fit(data_train, target_train)

# Predict on test set
y_pred = tree.predict(data_test)

# Métricas de avaliação
MSE = mean_squared_error(target_test, y_pred)
RMSE = mean_squared_error(target_test, y_pred, squared=False)
MAE = mean_absolute_error(target_test, y_pred)

print("MSE:", MSE)
print('RMSE:', RMSE)
print("MAE:", MAE)

# Salvar resultado para futura comparacao
file_object = open('results/resultsID3Reg.txt', 'a')
file_object.write(
    # f'test_size: {TEST_SIZE}, '
    #               f'MAX_DEPTH: {MAX_DEPTH}, '
    #               f'MIN_SAMPLES_SPLIT: {MIN_SAMPLES_SPLIT}, '
    #               f'MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}, '
    #               f'MAX_FEATURES: {MAX_FEATURES}, '
                  f' MSE: {MSE}, RMSE: {RMSE}, MAE: {MAE}\n')
