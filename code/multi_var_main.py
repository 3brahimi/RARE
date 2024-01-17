import os
import json
import numpy as np
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
from Training.ModelTrainer import ModelTrainer
from utils.helpers import evaluate_and_plot, evaluate
from Metric.RobustnessMetric import RobustnessMetric
from Metric.weights_estimation import estimate_weights
import torch
import tensorflow as tf

def main():

    x_len = 1000
    num_noises = 50
    distribution = 'normal'
    percentage = 0.5
    # res_folder = f"results_multi_var_noises_{num_noises}_2_testing"
    res_folder = f"./results_normal_training_sign_distances"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open('./configs/multi_var_config.json') as f:
        configs = json.load(f)
    
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    input_shape = num_inputs
    for config in configs["models"]:
        training_type = config["training_type"]
        print("************************ Training type is ************************", training_type)
        
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        plots_folder = f"{res_folder}/{config['type']}/{config['training_type']}/plots"
        xy_train, xy_valid, xy_test, _ , _, _, _= dataset_generator.split(config, metric_instance=metric)

        # move the data to the device
        
        if training_type == "noise-aware":
            input_shape = num_inputs * 2
        
        trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function='mean_squared_error')
        if config["load"] == True:
            model_path = config["model_path"]
            if "models_all" in model_path:
                models = []
                if config["training_type"] == "clean":
                    models_num = 1
                else:
                    models_num = 10
                losses = np.loadtxt(f"{model_path}/losses.txt")
                for i in range(models_num):
                    model_path_i = f"{model_path}/model_{i+1}"
                    model = trainer.load_model(f"{model_path_i}/model.pkl")
                    # model.to(device)
                    models.append(model)
                best_model = models[np.argmin(losses)]
            else:
                
                best_model = trainer.load_model(f"{model_path}/model.pkl")
                best_model.to(device)
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
            valid_losses = []
            if config["training_type"] == "clean":
                models_num = 1
            else:
                models_num = 11
                
            for i in range(models_num):
                if not os.path.exists(models_folder):
                    os.makedirs(models_folder)
                model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])
                if best_model is None or history.history['loss'][-1] < best_history.history['loss'][-1]:
                    best_model = model
                    best_history = history
                models.append(model)
                losses.append(history.history['loss'][-1])
                valid_losses.append(history.history['val_loss'][-1])
            if models_num > 1:
                # execlude the worst model out of the 11 models
                worst_model = models[np.argmax(losses)]
                models.remove(worst_model)
                losses.remove(np.max(losses))
                valid_losses.remove(np.max(valid_losses))
            # save the other 10 models in a folder called models_all, each with name model_1, model_2, etc.
            models_all_path = f"{models_folder}/{config['model_path']}"
            if not os.path.exists(models_all_path):
                os.makedirs(models_all_path)
            for i, model in enumerate(models):
                trainer.model = model
                trainer.save_model(f"{models_folder}/model_{i+1}")
            
            # save the losses list in a txt file
            with open(f"{models_folder}/losses.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in losses))
            with open(f"{models_folder}/valid_losses.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in valid_losses))
                    
            # trainer.model = best_model
            # trainer.save_model(f"{model_path}")     
            
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        evaluate_and_plot(best_model, best_history, xy_test, f"{plots_folder}")
        
        ####### evaluate model robustness
        ########### create new data
        x_len = 1000
        num_noises = 100
        percentage = 0.4
        # if the random seed is 0, then it will be randomly generated, otherwise it will be as specified
        
        test_noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
        
        test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)
        x_clean, y_clean = test_dataset_generator.generate_dataset()
        # x_clean = torch.from_numpy(x_clean).float() 
        # y_clean = torch.from_numpy(y_clean).float()
        
        # create ten different noisy sets
        outer_dists = ["Euclidean", "L1"]
        random_seeds = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225]
        no_noisy_tests = 2
        for idx, model in enumerate(models):
            if config["load"] == True:
                model_path_i = f"{model_path}/model_{idx+1}"
            else:
                model_path_i = f"{models_folder}/model_{idx+1}"
            if not os.path.exists(model_path_i):
                os.makedirs(model_path_i)    
            
            model_i_res_folder = f"{models_folder}/rm_results/model_{idx}"
            
            if not os.path.exists(model_i_res_folder):
                os.makedirs(model_i_res_folder)
            
            if training_type == "noise-aware":
                weights = estimate_weights(f"{model_path_i}/model.pkl", input_features, test_dataset_generator, training_type="noise-aware", num_samples=x_len)
                # weights = [1/(len(input_features) * 2)] * (len(input_features) * 2)
                target_feats_ids = [0,1]
                # rm_worst_output = metric.incremental_output_metric(x_clean, y_clean, test_dataset_generator, best_model, outer_dist=outer_dists, 
                #                                                    weights=weights, training_type="noise-aware", path=f"{model_path}/rm_results",
                #                                                    target_feat_ids=target_feats_ids)
            else:
                weights = estimate_weights(f"{model_path_i}/model.pkl", input_features,test_dataset_generator, num_samples=x_len)
                # rm_worst_output = metric.incremental_output_metric(x_clean, y_clean, test_dataset_generator, best_model, outer_dist=outer_dists, weights=weights, path=f"{model_path}/rm_results")
                target_feats_ids = [el for el in range(len(input_features))]
            rm_worst_output = None
            # for target_feat_idx in target_feats_ids:
            x_noisy_all = {j: None for j in range(no_noisy_tests)}
            y_noisy_all = {k: None for k in range(no_noisy_tests)}

            rms = {rm_i: None for rm_i in range(no_noisy_tests)}
            for i_noisy in range(no_noisy_tests):
                x_noisy, y_noisy = test_dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=[0,1,2,3], random_seed=random_seeds[i_noisy])
                x_noisy_all[i_noisy] = x_noisy
                y_noisy_all[i_noisy] = y_noisy
                
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
                    # x_noise_vector = torch.from_numpy(x_noise_vector).float()
                    y_noise_vector = model.predict(x_noise_vector)

                    # y_noisy_new[idx_shape, :] = y_noise_vector.flatten().cpu().numpy()
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
            print("results saved in", model_i_res_folder)
            with open(f"{model_i_res_folder}/rm.txt", "w") as outfile:
                json.dump(rms, outfile, indent=4)

if __name__ == '__main__':
    main()
