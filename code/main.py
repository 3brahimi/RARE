import datetime
import os
from DataGens import NoiseGenerator, DatasetGenerator
import numpy as np
from Training.ModelTrainer import ModelTrainer
import tensorflow as tf
import json

from Metric.RobustnessMetric import RobustnessMetric
from utils.helpers import evaluate_and_plot


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
    
    with open('./configs/config.json') as f:
        configs = json.load(f)
    
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
 
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    for config in configs["models"]:
        xy_train, xy_valid, xy_test= dataset_generator.split(config)
        
        trainer = ModelTrainer().get_model(config["type"], shape_input=1, loss_function='mean_squared_error')
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
            plot_path = f"{plots_folder}/{config['type']}/{config['model_path']}"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            trainer.save_model(f"./{model_path}") 
        
        evaluate_and_plot(model, history, xy_test, plot_path)
        
        x_clean, y_clean = dataset_generator.generate_dataset()

        x_noisy, y_noisy = dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=[0])

        rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy, y_hat=y_noisy, outer_dist=["Euclidean", "L1"])

if __name__ == '__main__':
    main()
