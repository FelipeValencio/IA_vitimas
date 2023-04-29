from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def loadData():
    data = pd.read_csv('treino_sinais_vitais_com_label.txt',sep=',', header=None)
    
    data.columns = ["index","pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classificação"]
    
    data.drop(columns=['index','pSist','pDiast'], inplace=True)
    
    explicadores = data[['qPA','pulso','respiração']]
    
    target = data[['classificação']]
    
    print('Data shape:', data.shape)

    return explicadores, target

# def loadDataSet():
#     # Open the text file and read the dataSet
#     with open('treino_sinais_vitais_com_label.txt', 'r') as file:
#         dataSet = file.readlines()

#     # Remove any leading or trailing white space and split the dataSet by commas
#     dataSet = [line.strip().split(',') for line in dataSet]

#     # Convert the dataSet to a NumPy array
#     dataSet = np.array(dataSet, dtype=np.float64)

#     print(dataSet)
#     # Exclui a coluna index
#     dataSet = np.delete(dataSet, 0, 1)

#     # Exclui a coluna gravidade
#     dataSet = np.delete(dataSet, 5, 1)


#     # Extract the features (X) and target variable (y)
#     targets = dataSet[:, -1]

#     # Print the shape of the dataSet
#     print('Data shape:', dataSet.shape)

#     return dataSet, targets


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

# Print accuracy score
print("Accuracy:", tree.score(data_test, target_test))
