import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

MAX_DEPTH = 20
TEST_SIZE = 0.3
NUM_TREES = 100


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

# Train the Decision Tree using ID3 algorithm
model = RandomForestClassifier(n_estimators=NUM_TREES, max_depth=MAX_DEPTH, random_state=42)
model.fit(data_train, target_train.values.ravel())

# Predict on test set
y_pred = model.predict(data_test)

print(y_pred[:10])
print(target_test[:10])

plotcm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(target_test, y_pred),
                                display_labels=['C1', 'C2', 'C3', 'C4'])
plotcm.plot()
plt.show()

print(classification_report(target_test, y_pred, ))

result_data = classification_report(target_test, y_pred, output_dict=True)

# Salvar resultado para futura comparacao
file_object = open('results/resultsRFClass.txt', 'a')
file_object.write(f'{result_data["accuracy"]}\n')
file_object.close()
