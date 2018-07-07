import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

iris = datasets.load_iris()
X= iris.data
#Y = to_categorical(iris.target,3)
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=1000)

def create_model(optimizer='rmsprop'):
    model = Sequential()
    model.add(Dense(8,activation='relu',input_shape = (4,)))
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer = optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model,
                        epochs=10, 
                        batch_size=5,
                        verbose=0)

#results = cross_val_score(model, X_train, Y_train, scoring='precision_macro')

param_grid = {'optimizer':('rmsprop','adam')}
grid = GridSearchCV(model,
                    param_grid=param_grid,
                    return_train_score=True,
                   scoring=['precision_macro','recall_macro','f1_macro'],
                    refit='precision_macro')
grid_results = grid.fit(X_train,Y_train)