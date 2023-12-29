import datetime
import os
from matplotlib import pyplot as plt
from NoiseGenerator import NoiseGenerator
import numpy as np
from DatasetGenerator import DatasetGenerator
from ModelTrainer import ModelTrainer
import tensorflow as tf
import json

from Models import LinearModel
from helpers import evaluate_and_plot

# Define the equation for the clean signals
# def equation(x):
#     return 2 * x + 3

# model architecture function
def model_architecture():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    return model

# get the data based on the type 
def get_data(data_types, x_clean, y_clean, x_noisy, y_noisy):
    x_data, y_data = [], []
    for data_type in data_types:
        if data_type == 'clean':
            x_data.append(x_clean)
            y_data.append(y_clean)
        elif data_type == 'gx':
            x_data.append(x_noisy)
            y_data.append(y_noisy)
    return np.concatenate(x_data), np.concatenate(y_data)

def main():

    x_len = 1000
    num_noises = 100
    distribution = 'normal'
    percentage = 0.1
    res_folder = f"results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    models_folder = f"{res_folder}/models"
    plots_folder = f"{res_folder}/plots"
    
    with open('./config.json') as f:
        configs = json.load(f)
    
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    equation_str = configs["equation"]
    dataset_generator = DatasetGenerator(equation_str, noise_model, num_samples=x_len)
    
    for config in configs["models"]:
        xy_train, xy_valid, xy_test= dataset_generator.split(config)
        
        trainer = ModelTrainer().get_model(config["type"], 'mean_squared_error')

        model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])
        model_path = f"{models_folder}/{config['model_path']}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        plot_path = f"{plots_folder}/{config['type']}/{config['model_path']}"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        trainer.save_model(f"./{model_path}/model.pkl") 
             
        evaluate_and_plot(model, history, xy_test, f"./{plot_path}")
        
if __name__ == '__main__':
    main()
