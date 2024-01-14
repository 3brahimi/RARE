import datetime
import os
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from DataGens import NoiseGenerator, DatasetGenerator
import numpy as np
from Training import ModelTrainer, Models, CustomModel
import tensorflow as tf
import json

from utils.helpers import evaluate_and_plot, evaluate
from Metric.RobustnessMetric import RobustnessMetric

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
    # res_folder = f"results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    res_folder = f"results_multi_outputs"
    models_folder = f"{res_folder}/models"
    plots_folder = f"{res_folder}/plots"
    
    with open('./configs/multi_outputs_config.json') as f:
        configs = json.load(f)
    
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    
    equations_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)

    dataset_generator = DatasetGenerator(equations_str[0], noise_model, input_features, num_samples=1000)
    metric = RobustnessMetric()
    
    for config in configs["models"]:
        xy_train, xy_valid, xy_test= dataset_generator.split_multi_outputs(config, equations_str)
        x_train = xy_train[0].reshape(-1, 1)
        y_train = xy_train[1]
        x_test = xy_test[0].reshape(-1, 1)
        y_test = xy_test[1]
        # mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
        # mlp.fit(x_train, y_train)

        # # Use the trained MLPRegressor to make predictions
        # y_pred = mlp.predict(x_test)
        # y1_pred = y_pred[:,0]
        # y2_pred = y_pred[:,1]

        trainer = ModelTrainer().get_model(config["type"], shape_input=1, loss_function='mean_squared_error', output_shape=2)
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
            
        # evaluate_and_plot(model, history, xy_test, plot_path)
        
        ####### evaluate model robustness
        ########### create new data 
        testing_no_samples = 1000
        testing_dataset_generator = DatasetGenerator(equations_str[0], noise_model, input_features, num_samples=testing_no_samples)
        y_clean = np.zeros((testing_no_samples, len(equations_str)))
        for i, equation in enumerate(equations_str):
            testing_dataset_generator.set_equation(equation)
            x_clean, y_temp = testing_dataset_generator.generate_dataset()
            y_temp = y_temp.ravel()
            y_clean[:, i] = y_temp
                    
        x_noisy, y_noisy = testing_dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=[0])
 
        ########### create y_noisy_new by predicting x_noisy, with the same shape as y_noisy
        y_noisy_new = np.zeros((y_noisy.shape[0], y_noisy.shape[1], y_noisy.shape[2]))
        for x_noise_vector in x_noisy:
            y_noise_vector = model.predict(x_noise_vector)
            np.append(y_noisy_new, y_noise_vector)
        
        outer_dist=["Euclidean", "L1"]
        # for each output, calculate robustness metric and then calculate the average of all outputs
        for dist in outer_dist:
            rm_vals = []
            for i in range(len(equations_str)):
                rm = metric.calculate_metric(x_clean, y_clean[:, i], x_hat=x_noisy, y_hat=y_noisy_new[:, i], outer_dist=outer_dist)
                rm_vals.append(rm[dist]["Ratio"])
            # calculate the 1-norm of the rm_vals    
            print("avg is", np.linalg.norm(rm_vals, ord=1), "for", dist)

        # rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy, y_hat=y_noisy_new, outer_dist=["Euclidean", "L1"])

        ########### save rm to txt file
        # if not os.path.exists(f"{model_path}/rm_results"):
        #     os.makedirs(f"{model_path}/rm_results")
        # with open(f"{model_path}/rm_results/rm.txt", "w") as outfile:
        #     json.dump(rm, outfile, indent=4)

if __name__ == '__main__':
    main()
