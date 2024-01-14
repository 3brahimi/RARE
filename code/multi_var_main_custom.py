import datetime
import os
from matplotlib import pyplot as plt
from DataGens.NoiseGenerator import NoiseGenerator
import numpy as np
from DataGens.DatasetGenerator import DatasetGenerator
from Training.ModelTrainer import ModelTrainer
import json

from Training.Models import LinearModel
from Training.CustomModel import CustomModel
from utils.helpers import evaluate_and_plot, evaluate
from Metric.RobustnessMetric import RobustnessMetric
from Metric.weights_estimation import estimate_weights

def main():

    x_len = 10
    num_noises = 2
    distribution = 'normal'
    percentage = 0.5
    res_folder = f"./results_custom_test"
    
    with open('./configs/multi_var_config_custom.json') as f:
        configs = json.load(f)
    
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    input_shape = num_inputs
    
    for config in configs["models"]:
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        plots_folder = f"{res_folder}/{config['type']}/{config['training_type']}/plots"
 
        training_type = config["training_type"]
        print("************************ Training type is ************************", training_type)
        
        xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(config)
        x_noisy, y_noisy = xy_noisy
        x_clean, y_clean = xy_clean
        gx, gx_y = gx_gy
        indices_train, indices_valid = indices
        
        x_noisy_train = x_noisy[:, indices_train, :]
        x_noisy_valid = x_noisy[:, indices_valid, :]
        
        y_noisy_train = y_noisy[:, indices_train, ]
        y_noisy_valid = y_noisy[:, indices_valid, ]
        
        x_clean_train = x_clean[indices_train, :]
        x_clean_valid = x_clean[indices_valid, :]
        y_clean_train = y_clean[indices_train,]
        y_clean_valid = y_clean[indices_valid,]
        
        if training_type == "noise-aware":
            input_shape = num_inputs * 2

        trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function='mean_squared_error')
        if config["load"] == True:
            model_path = config["model_path"]
            if "models_all" in model_path:
                models = []
                losses = np.loadtxt(f"{model_path}/losses.txt")
                for i in range(2):
                    model_path_i = f"{model_path}/model_{i+1}"
                    model = CustomModel.load_model(f"{model_path_i}/model.pkl", input_shape=input_shape, loss_function='mean_squared_error', output_shape=1)
                    models.append(model)
                best_model = models[np.argmin(losses)]
                print(len(models))
            else:
                
                best_model = trainer.load_model(f"{model_path}/model.pkl")
                models = [best_model]
            
            history = None
            best_history = None
            plot_path = f"{model_path}/plot"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
        else:
            best_model = None
            best_history = None
            models = []
            losses = []
                
            ############## calculate baseline metric for the equation itself for both training and validation sets
            y_noisy_bl_train = np.zeros((y_noisy_train.shape[0], y_noisy_train.shape[1]))
            y_noisy_bl_valid = np.zeros((y_noisy_valid.shape[0], y_noisy_valid.shape[1]))
            for idx_shape, x_noise_vector in enumerate(x_noisy_train):
                y_noise_vector = dataset_generator.apply_equation(x_noise_vector)
                y_noisy_bl_train[idx_shape, :] = y_noise_vector.flatten()
            outer_dists = ["Euclidean"]
            weights = [1/(len(input_features))] * (len(input_features))
            rm_bl_train = metric.calculate_metric(x_clean_train, y_clean_train, 
                                                  x_hat=x_noisy_train, y_hat=y_noisy_bl_train,
                                                  outer_dist=outer_dists, weights=weights, path=f"{res_folder}/baseline/training")["Euclidean"]["Ratio"]
            with open(f"{res_folder}/baseline/training/rm.txt", "w") as outfile:
                outfile.write(str(rm_bl_train))
            for idx_shape, x_noise_vector in enumerate(x_noisy_valid):
                y_noise_vector = dataset_generator.apply_equation(x_noise_vector)
                y_noisy_bl_valid[idx_shape, :] = y_noise_vector.flatten()
            rm_bl_valid = metric.calculate_metric(x_clean_valid, y_clean_valid,
                                                    x_hat=x_noisy_valid, y_hat=y_noisy_bl_valid,
                                                    outer_dist=outer_dists, weights=weights, path=f"{res_folder}/baseline/validation")["Euclidean"]["Ratio"]
            with open(f"{res_folder}/baseline/validation/rm.txt", "w") as outfile:
                outfile.write(str(rm_bl_valid))
            
            ####################################################
            
            for i in range(10):
                trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function='mean_squared_error')
                if not os.path.exists(models_folder):
                    os.makedirs(models_folder)
                model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid,
                                                         bl_training_ratio= rm_bl_train, bl_validation_ratio=rm_bl_valid,
                                                         xy_noisy= xy_noisy, gx_gy=gx_gy, 
                                                         xy_clean=xy_clean, indices=indices,
                                                         fit_args=config["fit_args"])      
                model.save_model(f"{models_folder}/model_{i+1}")
                print("model path is", models_folder)

                if best_model is None or history['loss'][-1] < best_history['loss'][-1]:
                    best_model = model
                    best_history = history
                models.append(model)
                losses.append(history['loss'][-1])
       
            # save the losses list in a txt file
            with open(f"{models_folder}/losses.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in losses))
                    
            # trainer.model = best_model
            # trainer.save_model(f"{model_path}")     
            if not os.path.exists(plots_folder):
                print("creating plot path")
                os.makedirs(plot_path)

        best_model.evaluate_and_plot(xy_test[0], xy_test[1], f"{plots_folder}")
        
############################################## evaluate model robustness
####################################### create new data
        x_len = 10
        num_noises = 2
        # if the random seed is 0, then it will be randomly generated, otherwise it will be as specified
        
        test_noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
        
        test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)
        x_clean, y_clean = test_dataset_generator.generate_dataset()
        # create ten different noisy sets

        outer_dists = ["Euclidean", "L1"]
        random_seeds = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225]
        no_noisy_tests = 2
        for idx, model in enumerate(models):
            
            # if training_type != "clean":
            model_path_i = f"{models_folder}/model_{idx+1}"
            # else:
                # models_folder_i = models_folder
            model_i_res_folder = f"{models_folder}/rm_results/model_{idx}"
            if not os.path.exists(model_i_res_folder):
                os.makedirs(model_i_res_folder)
            if training_type == "noise-aware":
                # weights = estimate_weights(f"{model_path_i}/model.pkl", input_features, training_type="noise-aware", num_samples=x_len)
                weights = estimate_weights(f"{model_path_i}", input_features, training_type="noise-aware", num_samples=x_len, model_type="CustomModel")
                target_feats_ids = [0,1]
                # rm_worst_output = metric.incremental_output_metric(x_clean, y_clean, test_dataset_generator, best_model, outer_dist=outer_dists, 
                #                                                    weights=weights, training_type="noise-aware", path=f"{model_path}/rm_results",
                #                                                    target_feat_ids=target_feats_ids)
            else:
                weights = estimate_weights(f"{model_path_i}", input_features, num_samples=x_len, model_type="CustomModel")
                # rm_worst_output = metric.incremental_output_metric(x_clean, y_clean, test_dataset_generator, best_model, outer_dist=outer_dists, weights=weights, path=f"{model_path}/rm_results")
                target_feats_ids = [el for el in range(len(input_features))]

            rm_worst_output = None
            # for target_feat_idx in target_feats_ids:
            x_noisy_all = {j: None for j in range(no_noisy_tests)}
            y_noisy_all = {k: None for k in range(no_noisy_tests)}
            rms = {rm_i: None for rm_i in range(no_noisy_tests)}
            for i in range(no_noisy_tests):
                x_noisy, y_noisy = test_dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=[0], random_seed=random_seeds[i])
                x_noisy_all[i] = x_noisy
                y_noisy_all[i] = y_noisy

                
            for key_rm, value in x_noisy_all.items():
                x_noisy = value
                y_noisy = y_noisy_all[key_rm]
                ########### estimate the weights of the input features
                y_noisy_new = np.zeros((y_noisy.shape[0], y_noisy.shape[1]))
                if training_type == "noise-aware":
                    x_noisy_new = np.zeros((x_noisy.shape[0], x_noisy.shape[1], x_noisy.shape[2] * 2))
                    # extract gx from each x feature in x_noisy and x 
                    for idx_shape in range(x_noisy.shape[2]):
                        gx = metric.extract_g(x_clean[:, idx_shape], x_hat=x_noisy[:, :, idx_shape])
                        gy = metric.extract_g(y_clean, x_hat=y_noisy)
                        # now append gx as a new feature in x_noisy as a new column
                        x_noisy_new[:, :, idx_shape] = x_noisy[:, :, idx_shape]
                        x_noisy_new[:, :, idx_shape + x_noisy.shape[2]] = gx
                else:
                    x_noisy_new = x_noisy

                for idx_shape, x_noise_vector in enumerate(x_noisy_new):
                    y_noise_vector = model.predict(x_noise_vector)

                    y_noisy_new[idx_shape, :] = y_noise_vector.flatten()
                    
                rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy_new, y_hat=y_noisy_new, outer_dist=outer_dists, weights=weights, 
                                             path=f"{model_i_res_folder}/xbar_{key_rm}")
                # for key_rm in rm.keys():
                #     if rm_worst_output is None:
                #         rm_worst_output = rm
                #     else:
                #        if rm_worst_output[key_rm]['Output distance'] < rm[key_rm]["Output distance"]:
                #            rm_worst_output = rm
                rms[key_rm] = rm
           
            ########### save rm to txt file
            with open(f"{model_i_res_folder}/rm.txt", "w") as outfile:
                json.dump(rms, outfile, indent=4)

if __name__ == '__main__':
    main()
