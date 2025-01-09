import os
from DataGens import NoiseGenerator, MultivarDatasetGenerator
import numpy as np
import sys
sys.path.append('..')
from Training import ModelTrainer
import tensorflow as tf
import json
from Metric.RobustnessMetric import RobustnessMetric
from Metric.weights_estimation import estimate_weights


def main():

    x_len = 1000
    num_noises = 100
    distribution = 'normal'
    percentage = 0.1
    # res_folder = f"results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    res_folder = f"results_mimo"
    models_folder = f"{res_folder}/models"
    plots_folder = f"{res_folder}/plots"
    
    with open('./configs/mimo_config.json') as f:
        configs = json.load(f)
    
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    
    equations_str = configs["equation"]
    print("equations_str is", equations_str)
    input_features = configs["features"]
    num_inputs = len(input_features)

    dataset_generator = MultivarDatasetGenerator(equations_str, noise_model, input_features, num_samples=1000)
    metric = RobustnessMetric()
    
    for config in configs["models"]:
        xy_train, xy_valid, xy_test= dataset_generator.split(config)
        x_train = xy_train[0]
        y_train = xy_train[1]
        x_test = xy_test[0]
        y_test = xy_test[1]
          # mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
        # mlp.fit(x_train, y_train)

        # # Use the trained MLPRegressor to make predictions
        # y_pred = mlp.predict(x_test)
        # y1_pred = y_pred[:,0]
        # y2_pred = y_pred[:,1]

        trainer = ModelTrainer().get_model(config["type"], shape_input=len(input_features), loss_function='mean_squared_error', output_shape=len(equations_str))
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
        testing_no_samples = 10
        x_len = 50
        num_noises = 10
        test_noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
        testing_dataset_generator = MultivarDatasetGenerator(equations_str, test_noise_model, input_features, num_samples=x_len)
        
        x_clean, y_clean = testing_dataset_generator.generate_dataset()
        print("x_clean is", x_clean.shape)
        print("y_clean is", y_clean.shape)            
        x_noisy, y_noisy = testing_dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=[1])
        print("x_noisy is", x_noisy)
        print("y_noisy is", y_noisy.shape)
        # plt.plot(y_noisy[0, :, 0], label="y1_noisy")
        # plt.plot(y_clean[:, 0], label="y1_clean")
        # plt.legend()
        # plt.show()        
        ########### create y_noisy_new by predicting x_noisy, with the same shape as y_noisy
        y_noisy_new = np.zeros((y_noisy.shape[0], y_noisy.shape[1], y_noisy.shape[2]))
        for idx, x_noise_vector in enumerate(x_noisy):
            y_noise_vector = model.predict(x_noise_vector)
            y_noisy_new[idx, :, :] = y_noise_vector
        #plot the first noisy vector in y_noisy_new
        # plt.plot(x_clean[:, 0], y_noisy[0, :, 0], label="y1_noisy")
        # plt.plot(x_clean[:, 0], y_noisy_new[0, :, 0], label="y1_noisy_new")
        # plt.legend()
        # plt.show()
        weights = estimate_weights(f"{model_path}/model.pkl", input_features)
        print("weights are", weights)
        outer_dist=["Euclidean", "L1"]
        # for each output, calculate robustness metric and then calculate the average of all outputs
        for dist in outer_dist:
            rm_vals = []
            for i in range(len(equations_str)):
                
                rm = metric.calculate_metric(x_clean, y_clean[:, i], x_hat=x_noisy, y_hat=y_noisy_new[:, :, i], outer_dist=outer_dist, weights=weights)
                print("rm is", rm)
                rm_vals.append(rm[dist]["Ratio"])
            # calculate the 1-norm of the rm_vals    
            print("avg is", np.linalg.norm(rm_vals, ord=1), "for", dist)

        # rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy, y_hat=y_noisy_new, outer_dist=["Euclidean", "L1"])

        ########## save rm to txt file
        if not os.path.exists(f"{model_path}/rm_results"):
            os.makedirs(f"{model_path}/rm_results")
        with open(f"{model_path}/rm_results/rm.txt", "w") as outfile:
            json.dump(rm, outfile, indent=4)

if __name__ == '__main__':
    main()
