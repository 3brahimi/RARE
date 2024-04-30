from math import ceil
import os
import json
import numpy as np
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
from Training.ModelTrainer import ModelTrainer
from utils.training_utils import CustomLoss
from utils.helpers import evaluate_and_plot
from Metric.RobustnessMetric import RobustnessMetric
from Metric.weights_estimation import estimate_weights
import torch
import tensorflow as tf
from utils.dists import L2_distance
import matplotlib.pyplot as plt
def plot_feature(x,y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"./{x_label}_{y_label}.png")


def main(res_folder, json_path, loss_fuction, noise_type, epsilon=0.5):
    physical_devices = tf.config.list_physical_devices('GPU')
    
    tf.config.set_visible_devices(physical_devices[1],'GPU')  # for using the first GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    num_noises = 20

    distribution = noise_type
    percentage = 0.3
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #### reading the config file    
    with open(json_path) as f:
        configs = json.load(f)

    x_len = configs["num_samples"]

    #### initialize the data generators
    noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=distribution, percentage=percentage, epsilon=epsilon)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    input_shape = num_inputs
    # loss_function = 'custom_loss'
    loss_function = loss_fuction
    config = configs["models"][0]   

    training_type = config["training_type"]
    plots_folder = f"{res_folder}/{config['type']}/{training_type}/plots"

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    
    x_clean, y_clean = dataset_generator.generate_dataset()
    print("****** For equation ", equation_str, ", Number of input features:", num_inputs , "******")
          
    print("Shape before meshgrid", x_clean.shape, y_clean.shape)
    # plot each input feature
    for i in range(x_clean.shape[1]):
        plt.clf()
        plt.plot(np.linspace(0, x_clean.shape[0], x_clean.shape[0]), x_clean[:, i], label="x_clean")
        plt.legend()
        plt.savefig(f"{plots_folder}/x_clean_{i}.png") 
    
    x_clean, y_clean = dataset_generator.meshgrid_x_y(x_clean)
        
    print("Shape after meshgrid", x_clean.shape, y_clean.shape)
    
    if num_inputs == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_clean[0,:,:], x_clean[1,:,:], y_clean)
        # set the labels
        ax.set_xlabel('$q$')
        ax.set_ylabel('$C$')
        ax.set_zlabel('$V$')
        ax.set_title(r'$V = q / C$')
        plt.savefig(f"{plots_folder}/clean_data_surface.png")     
    
    if num_inputs > 1:
        x_clean = x_clean.reshape(x_clean.shape[0], -1).T
        
    y_clean = y_clean.ravel()    
    
    print("Number of samples:", x_clean.shape[0], "Number of features:", x_clean.shape[1])
    # plot each input feature
    for i in range(x_clean.shape[1]):
        plt.clf()
        plt.plot(np.linspace(0, x_clean.shape[0], x_clean.shape[0]), x_clean[:, i], label="x_clean")
        plt.legend()
        plt.savefig(f"{plots_folder}/x_clean_meshgrid_{i}.png")
    
    # Modulate the clean dataset
    target_feats_ids = [el for el in range(len(input_features))]
    random_seeds = np.linspace(0, 1000, num_inputs, dtype=int)
    
    dataset_generator.num_samples = x_clean.shape[0]
    dataset_generator.noise_generator.num_samples = dataset_generator.num_samples
    
    x_noisy, y_noisy = dataset_generator.modulate_clean(x_clean, y_clean, target_feats_ids, random_seeds)
    print("Shape of noisy data: $X`$", x_noisy.shape, "$Y`$", y_noisy.shape)

    metric = RobustnessMetric()

    # Extract Gx
    # Extract the noisy signal with the maximum distance from the clean signal --> Gx
    if dataset_generator.num_inputs == 1:
        gx = np.zeros((dataset_generator.num_samples, dataset_generator.num_inputs))
        gx = dataset_generator.extract_g(x_noisy, x_clean)
        gx = metric.extract_g(x_hat=x_noisy[:,:,0], x=x_clean[:,0]) if metric else gx
        gx = gx.reshape(-1, 1)
    else:
        gx = np.zeros((dataset_generator.num_samples, dataset_generator.num_inputs))
        gx_y = np.zeros((dataset_generator.num_samples,))
        for i in range(dataset_generator.num_inputs):
            gx_temp = dataset_generator.extract_g(x_noisy[:, :, i], x_clean[:, i])
            gx_temp = metric.extract_g(x_hat=x_noisy[:, :, i], x = x_clean[:, i]) if metric else gx_temp
            gx[:, i] = gx_temp

    # plot Gx for each input feature
    for i in range(gx.shape[1]):
        plt.clf()
        plt.plot(np.linspace(0, gx.shape[0], gx.shape[0]), gx[:, i], label="$G(x_i)$")
        plt.plot(np.linspace(0, gx.shape[0], gx.shape[0]), x_clean[:, i], label="x_i")
        plt.legend()
        plt.savefig(f"{plots_folder}/gx_{i}.png")
                
if __name__ == '__main__':
    json_files = [
        "I_12_4.json",   
        "I_12_2.json",   
        "I_25_13.json",
        "I_6_2.json",
        "I_6_2b.json",
        "I_9_18.json", # not enough space for the meshgrid
        ]
    # Iterate over the JSON files
    for json_file in json_files:
        noise_type = "normal"
        loss_fuction = "mse"
        res_folder = f"./results_{os.path.splitext(json_file)[0]}/"
        json_path = f'./configs/equations/{json_file}'

        main(res_folder, json_path, loss_fuction="mse", noise_type=noise_type)