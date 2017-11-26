import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


seed = 101
np.random.seed(seed)

dataframe = pd.read_csv('dat.data', header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=.25, random_state=seed)

def my_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, init='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

weights_file = "weights.hdf5"
checkpoint = ModelCheckpoint(weights_file, monitor='acc', mode='max', save_best_only=True, verbose=1)

model = my_model()
model.fit(X_train, Y_train, batch_size=4, nb_epoch=200, verbose=1, callbacks=[checkpoint])

model_json = model.to_json()
with open("net.json", "w") as json_file:
    json_file.write(model_json)


jfile = open("net.json", "r")
loaded_json = jfile.read()
jfile.close()
loaded_model = model_from_json(loaded_json)

loaded_model.load_weights("weights.hdf5")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

predictions = loaded_model.predict_proba(X_test)
print('Accuracy: {}'.format(roc_auc_score(y_true=Y_test, y_score=predictions)))
