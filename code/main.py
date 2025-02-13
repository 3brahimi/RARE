import datetime
import os
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
import numpy as np
from Training.ModelTrainer import ModelTrainer
import tensorflow as tf
import json

from Metric.RobustnessMetric import RobustnessMetric
from utils.helpers import evaluate_and_plot


# # get the data based on the type 
# def get_data(data_types, x_clean, y_clean, x_noisy, y_noisy):
#     x_data, y_data = [], []
#     for data_type in data_types:
#         if data_type == 'clean':
#             x_data.append(x_clean)
#             y_data.append(y_clean)
#         elif data_type == 'gx':
#             x_data.append(x_noisy)
#             y_data.append(y_noisy)
#     return np.concatenate(x_data), np.concatenate(y_data)

def main(res_folder, json_file, loss_fuction, noise_type):

    with open(json_file) as f:
        configs = json.load(f)

    x_len = configs["num_samples"]
    num_noises = 20
    variance = 0.5
    
    
    noise_model = NoiseGenerator(x_len, num_noises, noise_type=noise_type, variance=variance)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    no_noisy_tests = 1
    random_seeds_all = [np.linspace(0, 1000, num_inputs, dtype=int) for _ in range(no_noisy_tests)]

    for idx, config in enumerate(configs["models"]):
        # xy_train_i, xy_valid_i, xy_test_i, xy_noisy_i, xy_clean_i, gx_gy_i, indices_i
        xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(config, metric_instance=metric)        
        trainer = ModelTrainer().get_model(config["type"], shape_input=num_inputs, loss_function=loss_fuction)
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        model_path = f'{models_folder}/model_{idx}'

        if config["load"] == True:
            model = trainer.load_model(f"{model_path}/model.pkl")
            history = None
            plot_path = f"{model_path}/plots"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
        else:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])
            
            plot_path = f"{model_path}/plots"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            trainer.save_model(model, f"./{model_path}") 
        
        evaluate_and_plot(model, history, xy_test, plot_path)
        
        x_clean, y_clean = dataset_generator.generate_dataset()

        target_features = config["noisy_input_feats"]

        x_noisy, y_noisy = dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=target_features, random_seeds=random_seeds_all[0])

        robustness_res_path = f"{model_path}/robusness_metric/"
        if not os.path.exists(robustness_res_path):
            os.makedirs(robustness_res_path)
        rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy, y_hat=y_noisy, outer_dist=["Euclidean", "L1"], path=f"{robustness_res_path}/")
        
        with open(f"{robustness_res_path}/rm_values.txt", "w") as f:
            json.dump(rm, f)

if __name__ == '__main__':
    eqs_json_files ={
        "I_6_2.json"
    }
    
    for json_file in eqs_json_files:
        res_folder = f"./results_mse_dp_{os.path.splitext(json_file)[0]}"
        noise_type = "normal"
        loss_fuction = "mse"

        res_folder = f"./results_{os.path.splitext(json_file)[0]}/loss_{loss_fuction}/{noise_type}"
        json_path = f"./configs/equations/{json_file}"
        
        main(res_folder, json_path, loss_fuction="mse", noise_type="normal")
