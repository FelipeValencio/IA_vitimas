from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

TEST_SIZE = 0.3


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

print(explicadores)
print(target)

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=0.3)

# Train the Decision Tree using ID3 algorithm
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(data_train, target_train)

# Predict on test set
y_pred = tree.predict(data_test)

cm = confusion_matrix(target_test, y_pred)

print(cm)

accuracy = tree.score(data_test, target_test)

# Print accuracy score
print("Accuracy:", accuracy)

print(classification_report(target_test, y_pred,))

# Salvar resultado para futura comparacao
file_object = open('results/resultsID3Class.txt', 'a')
file_object.write(f'test_size: {TEST_SIZE}, accuracy: {accuracy}\n')
file_object.close()