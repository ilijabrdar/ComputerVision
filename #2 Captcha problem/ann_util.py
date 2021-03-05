from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json
import numpy as np

# vecina koda ove skripte preuzeta sa vezbi


def create():
    ann = Sequential()
    ann.add(Dense(128, input_dim=12100, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train(ann, x_train, y_train):
    x = np.array(x_train, np.float32)
    y = np.array(y_train, np.float32)
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(x, y, epochs=500, batch_size=1, verbose=1, shuffle=False)
    return ann


def serialize(ann):
    model_json = ann.to_json()
    with open("serialized_model/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    ann.save_weights("serialized_model/neuronska.h5")


def load():
    try:
        with open('serialized_model/neuronska.json', 'r') as json_file:
            model_json = json_file.read()
        ann = model_from_json(model_json)
        ann.load_weights("serialized_model/neuronska.h5")
        return ann
    except Exception as e:
        print(e)
        return None