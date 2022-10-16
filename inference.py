import os
import tensorflow as tf
import numpy as np

class Denoiser:
    def __init__(self, model_path = 'denoising_model') -> None:
        self.model = tf.keras.models.load_model(model_path)
    
    def process(self, input_data):
        assert len(input_data.shape) == 2
        assert input_data.shape[1] == 80
        if input_data.shape[0] > 2048:
            input_arr = input_data[0: 2048, :]
        elif input_data.shape[0] < 2048:
            input_arr = np.zeros(shape=(2048, 80))
            input_arr[0:input_data.shape[0], :] = input_data
        else:
            input_arr = input_data
        input_arr = np.expand_dims(input_arr, axis = 0)
        prediction = self.model(input_arr)[0].numpy()
        if input_data.shape[0] < input_arr.shape[1]:
            prediction = prediction[0: input_data.shape[0], :]
        return prediction

class Classificator:
    def __init__(self, model_path = 'classification') -> None:
        self.model = tf.keras.models.load_model(model_path)

    def process(self, input_data):
        assert len(input_data.shape) == 2
        assert input_data.shape[1] == 80
        if input_data.shape[0] > 2048:
            input_arr = input_data[0: 2048, :]
        elif input_data.shape[0] < 2048:
            input_arr = np.zeros(shape=(2048, 80))
            input_arr[0:input_data.shape[0], :] = input_data
        else:
            input_arr = input_data
        input_arr = np.expand_dims(input_arr, axis = 0)
        prediction_arr = self.model(input_arr)[0]
        prediction = np.argmax(prediction_arr)
        if prediction == 0:
            pred_class = 'clear'
        else:
            pred_class = 'noise'
        confidence = prediction_arr[prediction]
        return pred_class, confidence