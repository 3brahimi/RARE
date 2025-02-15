import datetime
import os

import torch
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


def evaluate_robustness(models, dataset_generator, config, metric, random_seeds_all, robustness_res_path, weights):
    print("Evaluating robustness of models..")
    x_clean, y_clean = dataset_generator.generate_dataset()
    target_features = config["noisy_input_feats"]
    
    # TODO: here we can generate multiple noisy datasets
    x_noisy, y_noisy = dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=target_features, random_seeds=random_seeds_all[0])

    if not os.path.exists(robustness_res_path):
        os.makedirs(robustness_res_path)
    training_type = config["training_type"]
    num_inputs = dataset_generator.num_inputs
    
    if training_type == "noise-aware":
        x_noisy, input_shape = prepare_noisy_data(x_noisy=x_noisy, x_clean=x_clean, gx=None, num_inputs=num_inputs, data_gen=dataset_generator, metric=metric, stage="test")
        
    r_all_models = {}
    for model_idx, model in enumerate(models):
        model_i_robustness_folder = f"{robustness_res_path}/model_{model_idx}"
        if not os.path.exists(model_i_robustness_folder):
            os.makedirs(model_i_robustness_folder)
            
        y_noisy_pred = y_noisy
        for idx_shape, x_noise_vector in enumerate(x_noisy):
            y_noise_vector = model.predict(x_noise_vector)             
            y_noisy_pred[idx_shape, :] = y_noise_vector.flatten()     

        if np.min(x_clean) != np.max(x_clean) and np.min(y_clean) != np.max(y_clean):                              
            rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy, y_hat=y_noisy_pred, outer_dist=["Euclidean", "L1"], weights=weights[model_idx], path=f"{robustness_res_path}/model_{model_idx}/")
        else:
            print("Skipping metric calculation due to identical min and max values in input or output.")
            rm = 1
            rm = {"Output distance": 1}
            
        r_all_models[f"model_{model_idx}"] = rm  
        try:
            with open(f"{model_i_robustness_folder}/robustness_values.txt", "w") as f:
                json.dump(rm, f, indent=4)
        except Exception as e:
            print(f"Error writing to file {model_i_robustness_folder}/robustness_values.txt: {e}")

def prepare_noisy_data(x_noisy, x_clean, gx, num_inputs, data_gen, metric, stage="train"):
    if stage == "train":
        x_noisy, input_shape = _prepare_noisy_data_train(x_noisy, x_clean, gx, num_inputs)
    else:
        x_noisy, input_shape = _prepare_noisy_data_test(x_noisy, x_clean, metric, data_gen)
    return x_noisy, input_shape
    
def _prepare_noisy_data_train(x_noisy, x_clean, gx, num_inputs):
    x_noisy = np.tile(x_noisy, (1, 1, 2))
    # x_noisy[:, :, num_inputs:] = 0
    x_noisy[:, :, num_inputs:] = x_clean
    x_noisy[:, :, num_inputs:2*num_inputs] = gx
    input_shape = num_inputs * 2

    return x_noisy, input_shape

def _prepare_noisy_data_test(x_noisy, x_clean, metric, data_gen):
    # extract the significant patterns
    gx = np.zeros((data_gen.num_samples, data_gen.num_inputs))
    for idx_shape in range(x_noisy.shape[2]):
        gx_temp = metric.extract_g(x_clean[:, idx_shape], x_hat=x_noisy[:, :, idx_shape])
        gx[:, idx_shape] = gx_temp
    # add the significant patterns to the input
    x_noisy_new = np.zeros((x_noisy.shape[0], x_noisy.shape[1], x_noisy.shape[2] * 2))
    for idx_shape in range(x_noisy.shape[2]):
        x_noisy_new[:, :, idx_shape] = x_noisy[:, :, idx_shape]
        x_noisy_new[:, :, idx_shape + x_noisy.shape[2]] = gx[:, idx_shape]
    x_noisy = x_noisy_new
    input_shape = x_noisy.shape[2]
    return x_noisy, input_shape
    
def main(res_folder, json_file, loss_function, noise_type):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
    models_num = 2
    
    for config in configs["models"]:
        # reset config values for each model
        training_type = config["training_type"]
        input_features = configs["features"]
        num_inputs = len(input_features)
        input_shape = num_inputs
        dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)

        xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(config, metric_instance=metric)       
        
        if training_type == "noise-aware":
            x_noisy, input_shape = prepare_noisy_data(xy_noisy[0], xy_clean[0], gx_gy[0], num_inputs, dataset_generator, metric, stage="train")
        else:
            x_noisy = xy_noisy[0]
            input_shape = num_inputs
           
        xy_noisy = (x_noisy, xy_noisy[1])
        y_noisy = xy_noisy[1]    
        # Extract noisy and clean data
        # x_noisy, y_noisy = xy_noisy
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
        if np.min(x_clean_train) != np.max(x_clean_train) and np.min(y_clean_train) != np.max(y_clean_train):    
            # Calculate the baseline metric and weights
            bl_denominator, bl_weights = calculate_baseline_metric(dataset_generator, metric, x_clean_train, y_clean_train, x_noisy_train, y_noisy_train, input_features, training_type, res_folder)
        else:
            print("Skipping baseline metric calculation due to identical min and max values in input or output.")
            bl_denominator = 1
        print(f"Baseline denominator: {bl_denominator}")

        ############################# calculate the nominator of the metric in case of custom loss
        gxs_dists = []
        for i in range(num_inputs):
            gx = metric.extract_g(x_clean_valid[:, i], x_hat=x_noisy_valid[:, :, i])
            gx, x_clean_valid_i_scaled = metric.rescale_vector(true=x_clean_valid[:, i], noisy=gx)
            gxs_dists.append(L2_distance(gx, x_clean_valid_i_scaled, type="overall"))

        if training_type == "noise-aware":
            gxs_dists = np.append(gxs_dists, np.zeros(len(gxs_dists)))

        # multiply each gx by the corresponding weight
        gxs_dists = gxs_dists * bl_weights
        gxs_dists = np.sum(gxs_dists)
        
        trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
        
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"

        # if path is specified, load the models
        if config["load"] == True:
            models_path = config["model_path"]
            # change epsilon and loss function in model_path according to eps and loss_function values: /home/qamar/workspace/crml/code/results_I_27_6/loss_custom_loss/laplace_dp/epsilon_0.1/linear/clean/models_all
            models = []
            losses = np.loadtxt(f"{models_path}/losses.txt")                
            for i in range(models_num):
                model_path_i = f"{models_path}/model_{i+1}"
                trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
                model = trainer.model
                if config["type"] in ["RF", "LR"]:
                    model = trainer.load_model(model_path_i)
                else:
                    model.compile(optimizer='adam', loss=loss_function)
                    model.load_weights(f"{model_path_i}/model_weights.h5")                      
                models.append(model)
            print("Models loaded successfully")
        
        # if load is False, train the models  
        else:
            models = []
            losses = []
            valid_losses = []
            valid_losses_all_epochs = []
            last_epoch = []
            rm_vals = []
            # set the fit arguments if the loss function is msep
            if loss_function == "msep":
                config["fit_args"]["metric"] = metric
                config["fit_args"]["x_noisy_valid"] = tf.convert_to_tensor(x_noisy, dtype=tf.float64)
                config["fit_args"]["x_noisy_train"] = tf.convert_to_tensor(x_noisy_train, dtype=tf.float64)
                config["fit_args"]["len_input_features"] = input_shape
                config["fit_args"]["bl_ratio"] = tf.convert_to_tensor(bl_denominator, dtype=tf.float64)
                config["fit_args"]["nominator"] = tf.convert_to_tensor(gxs_dists, dtype=tf.float64)
                config["fit_args"]["y_clean_valid"] = tf.convert_to_tensor(y_clean_valid, dtype=tf.float64)
                config["fit_args"]["y_clean_train"] = tf.convert_to_tensor(y_clean_train, dtype=tf.float64)
            
            # train multiple models
            for i in range(models_num):
                model_path_i = f"{models_folder}/model_{i+1}"
                if not os.path.exists(models_folder):
                    os.makedirs(models_folder)
                if os.path.exists(model_path_i):
                    print(f"model_{i+1} already exists, add path in case you want to load, or delete the folder to retrain")
                    continue
                else:
                    if not os.path.exists(model_path_i):
                        os.makedirs(model_path_i)
                    model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])
                models.append(model)
                # save the training learning curves
                if history is not None:
                    if loss_function == "msep":
                        losses.append(history.history['mse'][-1])
                        valid_losses.append(history.history['val_mse'][-1])
                        rm_vals.append(history.history['R'])
                        valid_losses_all_epochs.append(history.history['val_mse'])
                    else:
                        losses.append(history.history['loss'][-1])
                        valid_losses.append(history.history['val_loss'][-1])
                        valid_losses_all_epochs.append(history.history['val_loss'])
                    last_epoch.append(len(history.history['loss']))
                    
                else:
                    losses.append(np.inf)
                    valid_losses.append(np.inf)
                            
                models_all_path = f"{models_folder}/{config['model_path']}"
                if not os.path.exists(models_all_path):
                    os.makedirs(models_all_path)
                    
                for i, model in enumerate(models):
                    if not os.path.exists(f"{models_folder}/model_{i+1}"):
                        os.makedirs(f"{models_folder}/model_{i+1}")
                    if config["type"] in ["RF", "LR"]:
                        trainer.save_model(f"{models_folder}/model_{i+1}")
                    else:
                        model.save_weights(f"{models_folder}/model_{i+1}/model_weights.h5")
                
                # save the losses list in a txt file
                with open(f"{models_folder}/losses.txt", "w") as outfile:
                    outfile.write("\n".join(str(item) for item in losses))
                with open(f"{models_folder}/valid_losses.txt", "w") as outfile:
                    outfile.write("\n".join(str(item) for item in valid_losses))
                with open(f"{models_folder}/last_epoch.txt", "w") as outfile:
                    outfile.write(str(last_epoch))
                with open(f"{models_folder}/rm_vals.txt", "w") as outfile:
                    outfile.write("\n".join(str(item) for item in rm_vals))
                with open(f"{models_folder}/valid_losses_all_epochs.txt", "w") as outfile:
                    outfile.write("\n".join(str(item) for item in valid_losses_all_epochs))
            
            
        # evaluate models robustness
        test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)   

        # Maybe this part can be done inside the loop: or is it better to do it outside the loop?
        models_weights = [[0 for _ in range(num_inputs)] for _ in range(models_num)]
        
        for i in range(models_num):
            model_path_i = f"{models_folder}/model_{i+1}"                               
            if training_type == "noise-aware":
                test_dataset_generator.num_inputs = num_inputs
                if loss_function == "msep":
                    weights = estimate_weights(f"{model_path_i}", input_features, test_dataset_generator, training_type="noise-aware", num_samples=x_len, loss_function=loss_function, metric=metric, 
                                               x_noisy=x_noisy_train, len_input_features=input_shape, bl_ratio=bl_denominator, y_clean=y_clean_train, model_type=config['type'])
                else:
                    weights = estimate_weights(f"{model_path_i}", input_features, test_dataset_generator, training_type="noise-aware", num_samples=x_len, loss_function=loss_function, model_type=config['type'])
            else:
                if loss_function == "msep":
                    weights = estimate_weights(f"{model_path_i}", input_features, test_dataset_generator, num_samples=x_len, loss_function=loss_function, metric=metric, x_noisy=x_noisy, len_input_features=input_shape, bl_ratio=bl_denominator, nominator=gxs_dists, y_clean=y_clean, model_type=config['type'])
                else:
                    weights = estimate_weights(f"{model_path_i}", input_features,test_dataset_generator, num_samples=x_len, model_type=config['type'])
            # if all weights are 0, then we use the correct weights
            if np.all(weights == 0):
                weights = bl_weights
            weights = bl_weights
            models_weights[i] = weights
        
        models_robustness_folder = f"{models_folder}/robustness"
        if not os.path.exists(models_robustness_folder):
            os.makedirs(models_robustness_folder)
        evaluate_robustness(models, test_dataset_generator, config, metric, random_seeds_all, models_robustness_folder, models_weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run robustness testing and training.")
    parser.add_argument('--noise_type', type=str, default="normal", help='Type of noise to apply.')
    parser.add_argument('--loss_function', type=str, default="mse", help='Loss function to use for training.')

    args = parser.parse_args()
    
    eqs_json_files = {
        # "I_6_2a.json",
        # "I_14_3.json",
        "I_6_2b.json",
        # "IV_1.json",
        # "IV_2.json",
        # "IV_6.json",
        # "IV_8.json",
        # "IV_10.json",
        
    }
    
    for json_file in eqs_json_files:
        res_folder = f"./results_{os.path.splitext(json_file)[0]}/loss_{args.loss_function}/{args.noise_type}"
        json_path = f"./configs/equations/{json_file}"
        print(f"Running robustness testing and training for {json_file}.. with noise type {args.noise_type} and loss function {args.loss_function}")
        main(res_folder, json_path, loss_function=args.loss_function, noise_type=args.noise_type)
