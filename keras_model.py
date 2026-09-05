########################################################################
# import python-library
########################################################################
# from import
import keras
from keras import layers
# import keras.models
# from keras import backend as K
# from keras.layers import Input, Dense, BatchNormalization, Activation
# from keras.models import Model


########################################################################
# keras model
########################################################################
def get_model(input_dim, lr):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """

    x = keras.Input(shape=(input_dim,))

    h = layers.Dense(128)(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(128)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(128)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(128)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(8)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(128)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(128)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(128)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(128)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)

    h = layers.Dense(input_dim)(h)

    model = keras.Model(inputs=x, outputs=h)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mean_squared_error",
    )

    return model

#########################################################################

def load_model(file_path):
    return keras.saving.load_model(file_path, compile=False)

def clear_session():
    keras.backend.clear_session()
#
# def load_model(file_path):
#     return keras.models.load_model(file_path, compile=False)
#
# def clear_session():
#     K.clear_session()
    