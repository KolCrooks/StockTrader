import tensorflow as tf
import tensorflow.keras as keras
import settings

def createModel(state_space, action_space):
    model = keras.models.Sequential([
        keras.layers.GRU(settings.GRU_UNITS, input_shape=(1, state_space)),
        keras.layers.Dense(settings.DENSE_UNITS, activation="relu"),
        keras.layers.Dropout(settings.DROPOUT),
        keras.layers.Dense(action_space, activation="softmax")
    ])
    return model