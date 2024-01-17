import datetime
import os
from matplotlib import pyplot as plt
import tensorflow as tf
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
from utils.dists import L2_distance

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, model, metric, x_noisy, len_input_features, bl_ratio, gx_dist, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.metric = metric
        self.x_noisy = x_noisy
        self.input_features = len_input_features
        self.bl_ratio = bl_ratio
        self.nominator = gx_dist
    
    
    def extract_g(self, y_noisy_tf, x_clean):
        y_noisy_min = tf.reduce_min(y_noisy_tf, axis=0)
        y_noisy_max = tf.reduce_max(y_noisy_tf, axis=0)
        y_dist_min = tf.abs(y_noisy_min - y_noisy_tf)
        y_dist_max = tf.abs(y_noisy_max - y_noisy_tf)
        gy = tf.maximum(y_dist_min, y_dist_max)
        
        mask_min = tf.equal(y_dist_min, gy)
        mask_max = tf.equal(y_dist_max, gy)
        # x_noisy_min_exp = tf.expand_dims(y_noisy_min, axis=0)
        # x_noisy_max_exp = tf.expand_dims(y_noisy_max, axis=0)
        g_points_min = tf.where(mask_min, y_noisy_min, tf.float32.min)
        g_points_max = tf.where(mask_max, y_noisy_max, tf.float32.min)
        g_points = tf.maximum(g_points_min, g_points_max)
        
        return g_points
    def call(self, y_true, y_pred):
        # Calculate the base loss (e.g., MSE loss)
        base_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        y_noisy = self.model(self.x_noisy)
        
        gy = self.extract_g(y_noisy, y_true)
        gy_y_dist = tf.sqrt(tf.square(gy - y_true))
        
        ratio =  (gy_y_dist / self.nominator) + 1
        
        # diff_ratio = tf.math.subtract(ratio, self.bl_ratio)
        # penalty = tf.maximum(tf.constant(0.), diff_ratio)

        # Return the total loss
        return base_loss * ratio
import tensorflow as tf

def extract_g(y_noisy_tf, y_clean):
        y_noisy_min = tf.reduce_min(y_noisy_tf, axis=0)
        y_noisy_max = tf.reduce_max(y_noisy_tf, axis=0)

        dist_min = tf.sqrt(tf.square(y_noisy_min - y_clean))
        dist_max = tf.sqrt(tf.square(y_noisy_max - y_clean))

        agg_dists = tf.maximum(dist_min, dist_max)
        
        mask_min = tf.equal(dist_min, agg_dists)
        
        mask_max = tf.equal(dist_max, agg_dists)
        # x_noisy_min_exp = tf.expand_dims(y_noisy_min, axis=0)
        # x_noisy_max_exp = tf.expand_dims(y_noisy_max, axis=0)
        g_points_min = tf.where(mask_min, y_noisy_min, 1.0)
        g_points_max = tf.where(mask_max, y_noisy_max, 1.0)

        g_points = g_points_min * g_points_max
        
        return g_points
        
class MyModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def main():

    x_len = 7
    num_noises = 2
    distribution = 'normal'
    percentage = 0.5
    # res_folder = f"results_multi_var_noises_{num_noises}_2_testing"
    res_folder = f"./results_custom_test"
    
    with open('./configs/multi_var_config_custom.json') as f:
        configs = json.load(f)

    # models_folder = f"{res_folder}/models_all"
    # plots_folder = f"{res_folder}/plots"
    
    noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    input_shape = num_inputs
    
    ####################################################
    for config in configs["models"]:
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        plots_folder = f"{res_folder}/{config['type']}/{config['training_type']}/plots"
 
        training_type = config["training_type"]
        print("************************ Training type is ************************", training_type)
        
        # xy_train, xy_valid, xy_test, xy_noisy_train, xy_noisy_valid, xy_noisy_test = dataset_generator.split(config)
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
        
        ######################################################################
        correct_weights = estimate_weights(model_path=f"{res_folder}/baseline/", inputs=input_features, dataset_generator=dataset_generator,
                                   num_samples=100, model_type="expression")
         
        y_noisy_bl_train = np.zeros((y_noisy_train.shape[0], y_noisy_train.shape[1]))
        y_noisy_bl_valid = np.zeros((y_noisy_valid.shape[0], y_noisy_valid.shape[1]))
        for idx_shape, x_noise_vector in enumerate(x_noisy_train):
            y_noise_vector = dataset_generator.apply_equation(x_noise_vector)
            y_noisy_bl_train[idx_shape, :] = y_noise_vector.flatten()
        outer_dists = ["Euclidean"]
        if training_type == "noise-aware":
            correct_weights = np.append(correct_weights, np.zeros(len(correct_weights)))
        rm_bl_train = metric.calculate_metric(x_clean_train, y_clean_train, 
                                              x_hat=x_noisy_train, y_hat=y_noisy_bl_train,
                                              outer_dist=outer_dists, weights=correct_weights, 
                                              vis=False, save=False,
                                              path=f"{res_folder}/baseline/training")["Euclidean"]["Output distance"]
        with open(f"{res_folder}/baseline/training/rm.txt", "w") as outfile:
            outfile.write(str(rm_bl_train))
        for idx_shape, x_noise_vector in enumerate(x_noisy_valid):
            y_noise_vector = dataset_generator.apply_equation(x_noise_vector)
            y_noisy_bl_valid[idx_shape, :] = y_noise_vector.flatten()
        rm_bl_valid = metric.calculate_metric(x_clean_valid, y_clean_valid,
                                                x_hat=x_noisy_valid, y_hat=y_noisy_bl_valid,
                                                outer_dist=outer_dists, weights=correct_weights, 
                                                vis=False, save=False, path=f"{res_folder}/baseline/validation")["Euclidean"]["Output distance"]
        with open(f"{res_folder}/baseline/validation/rm.txt", "w") as outfile:
            outfile.write(str(rm_bl_valid))
        print("rm_bl_train is", rm_bl_train, "rm_bl_valid is", rm_bl_valid, "correct weights are", correct_weights)
        ######################################################################
        
        if training_type == "noise-aware":
            input_shape = num_inputs * 2
        print("input shape is", input_shape)
        trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function='mean_squared_error')
        model = MyModel(input_shape=input_shape)
        gxs_dists = []
        
        # # for i in range(num_inputs):
        # gx = metric.extract_g(x_clean_train[:, 0], x_hat=x_noisy_train[:, :, 0])
        # print("gx is", gx)
        # print("l2 distance between gx and x_clean_train[:, 0] is", L2_distance(gx, x_clean_train[:, 0], type="overall"))
        # # gxs_dists.append(L2_distance(gx, x_clean_train[:, i], type="overall"))
        
        # # apply the tensorflow extract_g function instead of the numpy one
        # x_noisy_train_tf = tf.convert_to_tensor(x_noisy_train[:, :, 0])
        # x_clean_train_tf = tf.convert_to_tensor(x_clean_train[:, 0])
        # gx_new = extract_g(x_noisy_train_tf, x_clean_train_tf)
        # print("gx_new is", gx_new)
        # print("tf distance is:", tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(gx_new, x_clean_train_tf)))))
        # gx_new = gx_new.numpy()

        for i in range(num_inputs):
            gx = metric.extract_g(x_clean_train[:, i], x_hat=x_noisy_train[:, :, i])
            # apply the tensorflow extract_g function instead of the numpy one
            # x_noisy_train_tf = tf.convert_to_tensor(x_noisy_train[:, :, i])
            # x_clean_train_tf = tf.convert_to_tensor(x_clean_train[:, i])
            # gx = extract_g(x_noisy_train_tf, x_clean_train_tf)
            gxs_dists.append(L2_distance(gx, x_clean_train[:, i], type="overall"))
            
        if training_type == "noise-aware":
            gxs_dists = np.append(gxs_dists, np.zeros(len(gxs_dists)))
        # multiply each gx by the corresponding weight
        gxs_dists = gxs_dists * correct_weights
        gxs_dists = np.sum(gxs_dists)
        model.compile(optimizer='adam', loss=CustomLoss(model=model, metric=metric, x_noisy=x_noisy_train, gx_dist=gxs_dists, len_input_features=input_shape, bl_ratio=rm_bl_train))
        model.fit(x_clean_train, y_clean_train, epochs=300, verbose=1)
        # evaulate the model
        x_test = xy_test[0]
        y_test = xy_test[1]
        y_pred = model.predict(x_clean).flatten()

        plt.scatter(y_clean[:,], y_pred[:,])
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.show()
        
if __name__ == '__main__':
    main()
