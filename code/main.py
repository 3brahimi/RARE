import datetime
import os
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
import numpy as np
from Training.ModelTrainer import ModelTrainer
import tensorflow as tf
import json
import argparse

from Metric.RobustnessMetric import RobustnessMetric
from utils.helpers import evaluate_and_plot, calculate_baseline_metric
from utils.dists import L2_distance
from Metric.weights_estimation import estimate_weights


def evaluate_robustness(model_path, dataset_generator, config, metric, random_seeds_all):
    x_clean, y_clean = dataset_generator.generate_dataset()
    target_features = config["noisy_input_feats"]
    x_noisy, y_noisy = dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=target_features, random_seeds=random_seeds_all[0])
    robustness_res_path = f"{model_path}/robustness_metric/"
    if not os.path.exists(robustness_res_path):
        os.makedirs(robustness_res_path)
    # Calculate the robustness metric using Euclidean and L1 distance metrics for outer distribution
    rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy, y_hat=y_noisy, outer_dist=["Euclidean", "L1"], path=f"{robustness_res_path}/")
    # These metrics are chosen to evaluate the robustness of the model to different types of perturbations
    try:
        with open(f"{robustness_res_path}/rm_values.txt", "w") as f:
            json.dump(rm, f, indent=4)
    except Exception as e:
        print(f"Error writing to file {robustness_res_path}/rm_values.txt: {e}")





def main(res_folder, json_file, loss_function, noise_type):

    with open(json_file) as f:
        configs = json.load(f)

    x_len = configs["num_samples"]
    num_noises = 20
    variance = 0.5
    
    test_num_noises = 40
    test_variance = 0.3
    
    noise_model = NoiseGenerator(x_len, num_noises, noise_type=noise_type, variance=variance)
    test_noise_model = NoiseGenerator(x_len, test_num_noises, noise_type=noise_type, variance=test_variance)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    input_shape = num_inputs
    
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    no_noisy_tests = 1
    random_seeds_all = [np.linspace(0, 1000, num_inputs, dtype=int) for _ in range(no_noisy_tests)]

    for idx, config in enumerate(configs["models"]):
        xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(config, metric_instance=metric)       
        # Extract noisy and clean data
        x_noisy, y_noisy = xy_noisy
        x_clean, y_clean = xy_clean
        
        # Extract gradient information
        gx, gx_y = gx_gy
        
        # Extract training and validation indices
        indices_train, indices_valid = indices
        
        # Split noisy data into training and validation sets
        x_noisy_train = x_noisy[:, indices_train, :]
        x_noisy_valid = x_noisy[:, indices_valid, :]
        
        y_noisy_train = y_noisy[:, indices_train, ]
        y_noisy_valid = y_noisy[:, indices_valid, ]

        x_clean_train = x_clean[indices_train, :]
        x_clean_valid = x_clean[indices_valid, :]
        y_clean_train = y_clean[indices_train,]
        y_clean_valid = y_clean[indices_valid,]
        
        
        # Calculate the baseline metric and weights
        bl_denominator, bl_weights = calculate_baseline_metric(dataset_generator, metric, x_clean_train, y_clean_train, x_noisy_train, y_noisy_train, input_features, res_folder)
        print(f"Baseline denominator: {bl_denominator}", bl_weights)
        
        ############################# calculate the nominator of the metric in case of custom loss
        gxs_dists = []
        for i in range(num_inputs):
            gx = metric.extract_g(x_clean_valid[:, i], x_hat=x_noisy_valid[:, :, i])
            gx, x_clean_valid_i_scaled = metric.rescale_vector(true=x_clean_valid[:, i], noisy=gx)
            gxs_dists.append(L2_distance(gx, x_clean_valid_i_scaled, type="overall"))

        # multiply each gx by the corresponding weight
        gxs_dists = gxs_dists * bl_weights
        gxs_dists = np.sum(gxs_dists)
        
        trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
        
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        model_path = f'{models_folder}/model_{idx}'

        if config["load"] == True:
            model = trainer.load_model(f"{model_path}/model.pkl")
            history = None
            plot_path = f"{model_path}/plots"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
        else:
            
            if loss_function == "msep":
                config["fit_args"]["metric"] = metric
                # config["fit_args"]["x_noisy"] = tf.convert_to_tensor(xy_noisy[0], dtype=tf.float64)
                config["fit_args"]["x_noisy_valid"] = tf.convert_to_tensor(x_noisy, dtype=tf.float64)
                config["fit_args"]["x_noisy_train"] = tf.convert_to_tensor(x_noisy_train, dtype=tf.float64)
                config["fit_args"]["len_input_features"] = input_shape
                config["fit_args"]["bl_ratio"] = tf.convert_to_tensor(bl_denominator, dtype=tf.float64)
                config["fit_args"]["nominator"] = tf.convert_to_tensor(gxs_dists, dtype=tf.float64)
                config["fit_args"]["y_clean_valid"] = tf.convert_to_tensor(y_clean_valid, dtype=tf.float64)
                config["fit_args"]["y_clean_train"] = tf.convert_to_tensor(y_clean_train, dtype=tf.float64)
                
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])     
            trainer.save_model(model, f"./{model_path}")
        test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)     
        evaluate_robustness(model_path, test_dataset_generator, config, metric, random_seeds_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run robustness testing and training.")
    parser.add_argument('--noise_type', type=str, default="normal", help='Type of noise to apply.')
    parser.add_argument('--loss_function', type=str, default="mse", help='Loss function to use for training.')

    args = parser.parse_args()
    
    eqs_json_files = {
        "I_6_2.json"
    }
    
    for json_file in eqs_json_files:
        res_folder = f"./results_{os.path.splitext(json_file)[0]}/loss_{args.loss_function}/{args.noise_type}"
        json_path = f"./configs/equations/{json_file}"
        print(f"Running robustness testing and training for {json_file}.. with noise type {args.noise_type} and loss function {args.loss_function}")
        main(res_folder, json_path, loss_function=args.loss_function, noise_type=args.noise_type)
