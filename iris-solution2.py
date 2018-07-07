
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('iris-data.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4:5].values

# Part 1 - Data Preprocessing

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
y[:,0] = labelencoder_Y.fit_transform(y[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle =True)

# Part 2 - Now let's make the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
  #classifier.add(Dropout(p = 0.1))
  classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))
  #classifier.add(Dropout(p = 0.1))
  classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
  classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()
