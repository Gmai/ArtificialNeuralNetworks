
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('iris-data.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4:5].values

# Part 1 - Data Preprocessing ###############################################################################
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y[:,0])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle =True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle =True)

# Part 2 - Now let's make the ANN! ###############################################################################
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
import tensorflow as tf

def build_classifier(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
  classifier.add(Dropout(rate = 0.1))
  classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
  classifier.add(Dropout(rate = 0.1))
  classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
  classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return classifier
  
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy', #scoring =my_scorer
                           cv = 10)
grid_search = grid_search.fit(X=X_train,y=y_train) #groups
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Part 3 - Saving ###############################################################################
#save model
from keras.models import model_from_json
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)
best_classifier = build_classifier(best_parameters["optimizer"])
# Fit the model
best_classifier.fit(X_train,y_train_cat, epochs=best_parameters["epochs"], batch_size=best_parameters["batch_size"])
# evaluate the model
scores = best_classifier.evaluate(X_test, y_test_cat)
print("%s: %.2f%%" % (best_classifier.metrics_names[1], scores[1]*100))
# serialize model to JSON
model_json = best_classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
best_classifier.save_weights("model.h5")
print("Saved model to disk")

# Part 4 - Loading ###############################################################################
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=best_parameters["optimizer"], metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test_cat, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))