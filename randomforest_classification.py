import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

MAX_DEPTH = 10
TEST_SIZE = 0.3
RANDOM_STATE = 0


def loadData():
    data = pd.read_csv('treino_sinais_vitais_com_label.txt', sep=',', header=None)

    data.columns = ["index", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classificação"]

    data.drop(columns=['index', 'pSist', 'pDiast'], inplace=True)

    explicadores = data[['qPA', 'pulso', 'respiração']]

    target = data[['classificação']]

    print('Data shape:', data.shape)

    return explicadores, target


explicadores, target = loadData()

print(explicadores)
print(target)

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=TEST_SIZE)

# Train the Decision Tree using ID3 algorithm
model = RandomForestClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
model.fit(data_train, target_train.values.ravel())

# Predict on test set
y_pred = model.predict(data_test)

cm = confusion_matrix(target_test, y_pred)

print(cm)

accuracy = model.score(data_test, target_test)

# Print accuracy score
print("Accuracy:", accuracy)

print(classification_report(target_test, y_pred, ))

# Salvar resultado para futura comparacao
file_object = open('results/resultsRFClass.txt', 'a')
file_object.write(f'max_depth: {MAX_DEPTH}, test_size: {TEST_SIZE}, accuracy: {accuracy}\n')
file_object.close()