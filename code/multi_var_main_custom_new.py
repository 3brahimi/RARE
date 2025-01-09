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

        # Calculate the penalty
        outer_dists = ["Euclidean"]

        y_noisy = self.model(self.x_noisy)
        
        gy = self.extract_g(y_noisy, y_true)
        gy_y_dist = tf.sqrt(tf.square(gy - y_true))
        
        # second norm distance between y_noisy and y_true
        # diff = gy - tf.expand_dims(y_true, axis=0)
        # square_diff = tf.square(diff)
        # square_diff_sum = tf.reduce_sum(square_diff, axis=-1)
        # gy_dist = tf.sqrt(square_diff_sum)
        # mean_gy_dist = tf.reduce_mean(gy_dist)

        # # ratio = self.metric.calculate_metric(x=y_true, y=y_pred,
        # #                                      x_hat=x_noisy_tf, y_hat=y_noisy,
        # #                                      outer_dist=outer_dists, weights=weights, save=False, vis=False)
        # ratio = tf.constant(ratio[outer_dists[0]]["Ratio"])
        # ratio = tf.cast(ratio, tf.float32)
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
        # print("g_points_min is", g_points_min.numpy(), "g_points_max is", g_points_max.numpy())
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
        
        # x_noisy_valid, y_noisy_valid = xy_noisy_valid
        # x_noisy_test, y_noisy_test = xy_noisy_test
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
        print("x_clean_train shape is", x_clean_train.shape, "x_noisy_train shape is", x_noisy_train.shape, "num_inputs is", num_inputs)
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
        print("weights are", correct_weights, "gxs_dists is", gxs_dists)
        gxs_dists = gxs_dists * correct_weights
        gxs_dists = np.sum(gxs_dists)
        # model.compile(optimizer='adam', loss=CustomLoss(model=model, metric=metric, x_noisy=x_noisy_train, gx_dist=gxs_dists, len_input_features=input_shape, bl_ratio=rm_bl_train))
        model.compile(optimizer='adam', loss=CustomLoss(model=model, metric=metric, x_noisy=x_noisy_train, gx_dist=gxs_dists, len_input_features=input_shape, bl_ratio=rm_bl_train))
        # model.compile(optimizer='adam', loss="mse")
        model.fit(x_clean_train, y_clean_train, epochs=300, verbose=1)
        # evaulate the model
        x_test = xy_test[0]
        y_test = xy_test[1]
        y_pred = model.predict(x_clean).flatten()

        plt.scatter(y_clean[:,], y_pred[:,])
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.show()
        exit()
        
        
        if config["load"] == True:
            model_path = config["model_path"]
            if "models_all" in model_path:
                models = []
                losses = np.loadtxt(f"{model_path}/losses.txt")
                for i in range(10):
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
                
            ##############3 calculate baseline metric for the equation itself for both training and validation sets
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
            print("rm_bl_train is", rm_bl_train)
            for idx_shape, x_noise_vector in enumerate(x_noisy_valid):
                y_noise_vector = dataset_generator.apply_equation(x_noise_vector)
                y_noisy_bl_valid[idx_shape, :] = y_noise_vector.flatten()
            rm_bl_valid = metric.calculate_metric(x_clean_valid, y_clean_valid,
                                                    x_hat=x_noisy_valid, y_hat=y_noisy_bl_valid,
                                                    outer_dist=outer_dists, weights=weights, path=f"{res_folder}/baseline/validation")["Euclidean"]["Ratio"]
            with open(f"{res_folder}/baseline/validation/rm.txt", "w") as outfile:
                outfile.write(str(rm_bl_valid))
            
            ####################################################
            
            # sys.setrecursionlimit()
            for i in range(10):
                trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function='mean_squared_error')
                if not os.path.exists(models_folder):
                    os.makedirs(models_folder)
                model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid,
                                                         bl_training_ratio= rm_bl_train, bl_validation_ratio=rm_bl_valid,
                                                         xy_noisy= xy_noisy, gx_gy=gx_gy, 
                                                         xy_clean=xy_clean, indices=indices,
                                                         fit_args=config["fit_args"], weights=correct_weights)    
                model.save_model(f"{models_folder}/model_{i+1}/model.pkl")
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
            #     os.makedirs(plot_path)
            # exit()
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
        no_noisy_tests = 5
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
                weights = estimate_weights(f"{model_path_i}/model.pkl", input_features, training_type="noise-aware", num_samples=x_len, model_type="CustomModel")
                # weights = [1/(len(input_features) * 2)] * (len(input_features) * 2)
                target_feats_ids = [0,1]
                # rm_worst_output = metric.incremental_output_metric(x_clean, y_clean, test_dataset_generator, best_model, outer_dist=outer_dists, 
                #                                                    weights=weights, training_type="noise-aware", path=f"{model_path}/rm_results",
                #                                                    target_feat_ids=target_feats_ids)
            else:
                weights = estimate_weights(f"{model_path_i}/model.pkl", input_features, num_samples=x_len, model_type="CustomModel")
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
            # print("x_noisy_all is", x_noisy_all)
            # print("y_noisy_all is", y_noisy_all)
                
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
