from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

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

    target = data[['classificação']]

    print('Data shape:', data.shape)

    return explicadores, target


# Load dataset
explicadores, target = loadData()

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=TEST_SIZE)

# Train the Decision Tree using ID3 algorithm
tree = DecisionTreeClassifier(criterion="entropy", max_depth=MAX_DEPTH,
                              min_samples_split=MIN_SAMPLES_SPLIT,
                              min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES)
tree.fit(data_train, target_train)

# Predict on test set
y_pred = tree.predict(data_test)

plotcm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(target_test, y_pred),
                                display_labels=['C1', 'C2', 'C3', 'C4'])
plotcm.plot()
plt.show()

accuracy = tree.score(data_test, target_test)

print(classification_report(target_test, y_pred, ))

result_data = classification_report(target_test, y_pred, output_dict=True)

# Salvar resultado para futura comparacao
file_object = open('results/resultsID3Class.txt', 'a')
file_object.write(f'{result_data["accuracy"]}\n')
file_object.close()
