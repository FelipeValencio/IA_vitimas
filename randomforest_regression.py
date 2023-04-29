from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

def loadData():
    data = pd.read_csv('treino_sinais_vitais_com_label.txt',sep=',', header=None)
    
    data.columns = ["index","pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classificação"]
    
    data.drop(columns=['index','pSist','pDiast'], inplace=True)
    
    explicadores = data[['qPA','pulso','respiração']]
    
    target = data[['gravidade']]
    
    print('Data shape:', data.shape)

    return explicadores, target

# Load dataset
explicadores, target = loadData()

# Split dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(explicadores, target, test_size=0.3)

# Train the Decision Tree using ID3 algorithm
model = RandomForestRegressor(max_depth=5, random_state=0)
model.fit(data_train, target_train.values.ravel())

# Predict on test set
y_pred = model.predict(data_test)

# Print accuracy score
print("Accuracy:", model.score(data_test, target_test))