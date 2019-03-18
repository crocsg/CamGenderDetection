import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


class GenderPredictor:
    modelpath = ""
    model = None

    def __init__(self, modelpath):
        self.modelpath = modelpath


    def build_model (self):
        # build keras model
        self.model = Sequential()
        self.model.add(Conv2D(128, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(64, 64, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #load model weights
        self.model.load_weights(self.modelpath)


    def predict (self, rgb_image):
        prediction = []

        fimg = np.array(rgb_image, dtype=float)           # convert to float image
        fimg /= 255.0
        train_faces = [fimg]
        tmp = np.array(train_faces, dtype=float)
        prediction = self.model.predict(tmp)

        return prediction
