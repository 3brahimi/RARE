from math import ceil
import os
import json
import numpy as np
import sys
sys.path.append('..')

from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
from Training.ModelTrainer import ModelTrainer
from utils.helpers import evaluate_and_plot
from Metric.RobustnessMetric import RobustnessMetric
from Metric.weights_estimation import estimate_weights
import torch
import tensorflow as tf

# def main(res_folder, loss_function):
def main(res_folder, json_path):
    physical_devices = tf.config.list_physical_devices('GPU')
    
    tf.config.set_visible_devices(physical_devices[1],'GPU')  # for using the first GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # x_len = 7
    num_noises = 10
    # num_noises = 2
    distribution = 'normal'
    percentage = 0.5
    
    res_folder = res_folder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #### reading the config file    
    # with open('./configs/multi_var_config.json') as f:
    with open(json_path) as f:
        configs = json.load(f)

    x_len = configs["num_samples"]

    #### initialize the data generators
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    input_shape = num_inputs
    # loss_function = 'custom_loss'
    loss_function = 'mse'
    
    for config in configs["models"]:
        training_type = config["training_type"]
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        plots_folder = f"{res_folder}/{config['type']}/{config['training_type']}/plots"
        extended_data = config["extended_data"]
    
        xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(config, metric_instance=metric)
        
        x_noisy = xy_noisy[0]
        y_noisy = xy_noisy[1]

        gx, gx_y = gx_gy
                       
        trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
        
        
        ############################# calculate the correct weights of the expression as a function of the input features, using Salib
        correct_weights = estimate_weights(model_path=f"{res_folder}/baseline/", inputs=input_features, dataset_generator=dataset_generator,
                                           num_samples=1000, model_type="expression") 
        
        ####################################################
        if config["training_type"] == "clean":
            models_num = 1
        else:
            models_num = 3
                   
        if config["load"] == True:
            model_path = config["model_path"]
            if "models_all" in model_path:
                models = []
                losses = np.loadtxt(f"{model_path}/losses.txt")
                    
                # for i in range(models_num - 1): # -1 because we removed the worst model
                for i in range(models_num): # -1 because we removed the worst model
                    model_path_i = f"{model_path}/model_{i+1}"
                    trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
                    model = trainer.model
                        
                    model.compile(optimizer='adam', loss=loss_function)
                    model.load_weights(f"{model_path_i}/model_weights.h5")  
                    print("model loaded with mse")
                    models.append(model)
                print("Number of loaded models is: ", len(models))
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
            best_model_path = None
            best_history = None
            models = []
            losses = []
            valid_losses = []
            last_epoch = []
            rm_vals = []
            
            for i in range(models_num):
                if not os.path.exists(models_folder):
                    os.makedirs(models_folder)
                model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])
                # if best_model is None or history.history['val_mse'][-1] < best_history.history['val_mse'][-1]:
                if best_model is None or history.history['val_loss'][-1] < best_history.history['val_loss'][-1]:
                    best_model = model
                    best_history = history
                    best_model_path = f"{models_folder}/model_{i+1}"
                models.append(model)
                if history is not None:
                    if loss_function == "custom_loss":
                        losses.append(history.history['mse'][-1])
                        valid_losses.append(history.history['val_mse'][-1])
                        rm_vals.append(history.history['custom_metric'])
                    else:
                        losses.append(history.history['loss'][-1])
                        valid_losses.append(history.history['val_loss'][-1])
                    last_epoch.append(len(history.history['loss']))
                else:
                    losses.append(np.inf)
                    valid_losses.append(np.inf)                    
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
                if not os.path.exists(f"{models_folder}/model_{i+1}"):
                    os.makedirs(f"{models_folder}/model_{i+1}")
                model.save_weights(f"{models_folder}/model_{i+1}/model_weights.h5")
                # trainer.model = model
                # trainer.save_model(f"{models_folder}/model_{i+1}")           
            # save the losses list in a txt file
            with open(f"{models_folder}/losses.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in losses))
            with open(f"{models_folder}/valid_losses.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in valid_losses))
            with open(f"{models_folder}/last_epoch.txt", "w") as outfile:
                outfile.write(str(last_epoch))
            with open(f"{models_folder}/rm_vals.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in rm_vals))

        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)   
        evaluate_and_plot(best_model, best_history, xy_test, f"{plots_folder}")

    
####### evaluate model robustness
        ########### create new data
        # x_len = 32
        x_len = configs["num_samples"]
        # x_len = 100
        num_noises = 40
        percentage = 0.5
        # if the random seed is 0, then it will be randomly generated, otherwise it will be as specified
        
        test_noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
        x_noisy_new = None
        test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)
        x_clean_orig, y_clean_orig = test_dataset_generator.generate_dataset()

        # apply the meshgrid to x and y clean
        y_clean_orig = y_clean_orig.ravel()
        
        x_clean_orig, y_clean_orig = test_dataset_generator.meshgrid_x_y(x_clean_orig)

        x_clean_orig = x_clean_orig.reshape(x_clean_orig.shape[0], -1).T
        y_clean_orig = y_clean_orig.ravel()

        original_num_samples = test_dataset_generator.num_samples
        test_dataset_generator.num_samples = x_clean_orig.shape[0]
        test_dataset_generator.noise_generator.num_samples = test_dataset_generator.num_samples
        
        
        # create ten different noisy sets
        outer_dists = ["Euclidean", "L1"]
        # random_seeds_all should be of length = no_noisy_tests, where each element is a set of random seeds for each input feature
        no_noisy_tests = 1

        random_seeds_all = [np.linspace(0, 1000, num_inputs, dtype=int) for _ in range(no_noisy_tests)]
        
        for idx, model in enumerate(models):
            

            if config["load"] == True:
                model_path_i = f"{model_path}/model_{idx+1}"
                
            else:
                model_path_i = f"{models_folder}/model_{idx+1}"
            if not os.path.exists(model_path_i):
                os.makedirs(model_path_i)    
            
            model_i_res_folder = f"{models_folder}/rm_results_new3/model_{idx+1}"
            #### new3 is new2 again but for 5 models only
            if not os.path.exists(model_i_res_folder):
                os.makedirs(model_i_res_folder)            
            target_feats_ids = [el for el in range(len(input_features))]
            rm_worst_output = None
            # for target_feat_idx in target_feats_ids:
            x_noisy_all = {j: None for j in range(no_noisy_tests)}
            y_noisy_all = {k: None for k in range(no_noisy_tests)}
            weights_all = {l: None for l in range(no_noisy_tests)}
            rms = {rm_i: None for rm_i in range(no_noisy_tests)}
            
            for i_noisy in range(no_noisy_tests):
                print("i_noisy", i_noisy, "model", idx, "x_clean_orig", x_clean_orig.shape, "y_clean_orig", y_clean_orig.shape)
                x_noisy, y_noisy = test_dataset_generator.modulate_clean(x_clean_orig, y_clean_orig, target_feat_idx=target_feats_ids, random_seeds=random_seeds_all[i_noisy])
                
                x_noisy_all[i_noisy] = x_noisy
                y_noisy_all[i_noisy] = y_noisy
            
            for key_rm, value in x_noisy_all.items():
                x_noisy = value
                y_noisy = y_noisy_all[key_rm]
                

                ########### estimate the weights of the input features
                y_noisy_new = np.zeros((y_noisy.shape[0], y_noisy.shape[1]))
                
                if training_type == "noise-aware":
                    # extract gx from each x feature in x_noisy and x
                    # gx = np.zeros((x_len, test_dataset_generator.num_inputs))
                    gx = np.zeros((test_dataset_generator.num_samples, test_dataset_generator.num_inputs))
                    gx_y = np.zeros((test_dataset_generator.num_samples,))
                    for idx_shape in range(x_noisy.shape[2]):
                        gx_temp = metric.extract_g(x_clean_orig[:, idx_shape], x_hat=x_noisy[:, :, idx_shape])
                        gx[:, idx_shape] = gx_temp
                                                
                gx_y = y_clean_orig
                
                sampling_rate = ceil(x_clean_orig.shape[0]/((original_num_samples*2)*(x_clean_orig.shape[1])))
                print(f"Downsampling the data to {sampling_rate} of the original size")

                x_clean_orig = x_clean_orig[::sampling_rate]
                y_clean_orig = y_clean_orig[::sampling_rate]
                x_noisy = x_noisy[:, ::sampling_rate]
                y_noisy = y_noisy[:, ::sampling_rate]
                gx = gx[::sampling_rate]
                gx_y = gx_y[::sampling_rate]
                
                     
                # for noise-aware training, we need to add the gx to the x_noisy
                if training_type == "noise-aware":
                    x_noisy_new = np.zeros((x_noisy.shape[0], x_noisy.shape[1], x_noisy.shape[2] * 2))
                    for idx_shape in range(x_noisy.shape[2]):
                        
                        x_noisy_new[:, :, idx_shape] = x_noisy[:, :, idx_shape]
                        x_noisy_new[:, :, idx_shape + x_noisy.shape[2]] = gx[:, idx_shape]

                else:
                    x_noisy_new = x_noisy
                    
                y_noisy_new = y_noisy
                
                #### calculate the weights of the input features
                weights = estimate_weights(f"{model_path_i}", input_features,test_dataset_generator, num_samples=x_len)

                # if all weights are 0, then we use the correct weights
                if np.all(weights == 0):
                    weights = correct_weights
                weights = correct_weights
                for idx_shape, x_noise_vector in enumerate(x_noisy_new):
                    y_noise_vector = model.predict(x_noise_vector)
                    y_noisy_new[idx_shape, :] = y_noise_vector.flatten()

                rm = metric.calculate_metric(x_clean_orig, y_clean_orig, x_hat=x_noisy_new, y_hat=y_noisy_new, outer_dist=outer_dists, weights=weights, 
                                             path=f"{model_i_res_folder}/xbar_{key_rm}")
                

                rms[key_rm] = rm
                # TODO: instead of resetting the num_samples, we should create a copy of the dataset generator
                test_dataset_generator.num_samples = x_clean_orig.shape[0]
                test_dataset_generator.noise_generator.num_samples = test_dataset_generator.num_samples
           
            ########### save rm to txt filethe thre
            print("results saved in", model_i_res_folder)
            with open(f"{model_i_res_folder}/rm_new.txt", "w") as outfile:
                json.dump(rms, outfile, indent=4)

if __name__ == '__main__':
    json_files = [f for f in os.listdir('../configs/equations/') if f.endswith('.json')]
    json_files = [
        "I_6_2.json",
    ]
    # Iterate over the JSON files
    for json_file in json_files:
        # Set the res_folder variable based on the name of the JSON file
        res_folder = f"../results_all_noisy_{os.path.splitext(json_file)[0]}"
        json_path = f'../configs/equations/{json_file}'
        main(res_folder, json_path)
        