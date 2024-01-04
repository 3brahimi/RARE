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
from helpers import evaluate_and_plot, evaluate
from RobustnessMetric import RobustnessMetric
from weights_estimation import estimate_weights
# model architecture function
def model_architecture():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    return model

def main():

    x_len = 1000
    num_noises = 100
    distribution = 'normal'
    percentage = 0.1
    res_folder = f"results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    models_folder = f"{res_folder}/models"
    plots_folder = f"{res_folder}/plots"
    
    with open('./multi_var_config.json') as f:
        configs = json.load(f)
    
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    
    for config in configs["models"]:
        xy_train, xy_valid, xy_test= dataset_generator.split(config)

        trainer = ModelTrainer().get_model(config["type"], input_shape=xy_train[0].shape[1], loss_function='mean_squared_error')
        if config["load"] == True:
            model_path = config["model_path"]
            model = trainer.load_model(f"{model_path}/model.pkl")
            history = None
            plot_path = f"{model_path}/plots"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
        else:
            model_path = f"{models_folder}/{config['model_path']}"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])
            trainer.save_model(model_path)
            
            plot_path = f"{plots_folder}/{config['type']}/{config['model_path']}"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        evaluate_and_plot(model, history, xy_test, plot_path)
        
        ####### evaluate model robustness
        ########### create new data 
        x_clean, y_clean = dataset_generator.generate_dataset()
        
        x_noisy, y_noisy = dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=[0,1])
 
        ########### estimate the weights of the input features
        weights = estimate_weights(f"{model_path}/model.pkl", input_features)

        ########### create y_noisy_new by predicting x_noisy, with the same shape as y_noisy
        y_noisy_new = np.zeros((y_noisy.shape[0], y_noisy.shape[1]))
        for x_noise_vector in x_noisy:
            y_noise_vector = model.predict(x_noise_vector)
            np.append(y_noisy_new, y_noise_vector)
        

        rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy, y_hat=y_noisy_new, outer_dist=["Euclidean", "L1"], weights=weights)

        ########### save rm to txt file
        if not os.path.exists(f"{model_path}/rm_results"):
            os.makedirs(f"{model_path}/rm_results")
        with open(f"{model_path}/rm_results/rm.txt", "w") as outfile:
            json.dump(rm, outfile, indent=4)

if __name__ == '__main__':
    main()
