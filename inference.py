import os
import tensorflow as tf
import numpy as np
import argparse

SIGNAL_LENGTH = 2048
N_MELS = 80

class Denoiser:
    def __init__(self, model_path = 'denoising_model') -> None:
        self.model = tf.keras.models.load_model(model_path)
    
    def process(self, input_data):
        assert len(input_data.shape) == 2
        assert input_data.shape[1] == N_MELS
        if input_data.shape[0] > SIGNAL_LENGTH:
            input_arr = input_data[0: SIGNAL_LENGTH, :]
        elif input_data.shape[0] < SIGNAL_LENGTH:
            input_arr = np.zeros(shape=(SIGNAL_LENGTH, N_MELS))
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
        assert input_data.shape[1] == N_MELS
        if input_data.shape[0] > SIGNAL_LENGTH:
            input_arr = input_data[0: SIGNAL_LENGTH, :]
        elif input_data.shape[0] < SIGNAL_LENGTH:
            input_arr = np.zeros(shape=(SIGNAL_LENGTH, N_MELS))
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

def get_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mel_path', help='Path to .npy file with mel-spectrogram', required=True)
    parser.add_argument('-t', '--task', help = "Name of task: c - classification, d- denoising", default='d')
    parser.add_argument('-r', '--result_path', help = "Path to save result mel-spectrogram after denoising", default='.')
    args = parser.parse_args()
    return args

def load_data(path):
    try:
        data = np.load(path)
    except Exception as e:
        print(f"Can't load file {path}, error: {e}")
    return data

def save_data(data, data_path, path):
    fname = data_path.split(os.sep)[-1]
    fname_new = 'result_' + fname
    new_path = os.path.join(path, fname_new)
    try:
        with open(new_path, 'wb') as f:
            np.save(f, data)
        print(f"Data saved to {new_path}")
    except Exception as e:
        print(f"Can't save data to {path}, error: {e}")


if __name__ == "__main__":
    args = get_args()
    data = load_data(args.mel_path)
    if args.task == 'd':
        denoiser = Denoiser(model_path='denoising_model')
        denoized_data = denoiser.process(data)
        save_data(denoized_data, args.mel_path, args.result_path)
    if args.task == 'c':
        classificator = Classificator(model_path='classification')
        classification_results = classificator.process(data)
        print(f"Class: {classification_results[0]}, Confidence: {classification_results[1].numpy()[0]}")