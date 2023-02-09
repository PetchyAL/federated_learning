import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Client():
    def __init__(self, ID, data, model):
        self.ID = ID
        self.data = data
        self.model = model
    
    def train(self, weights):
        self.model.set_weights(weights)
        self.model.fit(self.data, epochs=1)

    def get_batch_size(self):
        return list(self.data)[0][0].shape[0]

    def get_training_data_points(self):
        return tf.data.experimental.cardinality(self.data).numpy()

if __name__ == '__main__':
    print('-'*50 + '\nThis is the client class, do not run this directly!\nRun main.py instead.\n'+ '-'*50)