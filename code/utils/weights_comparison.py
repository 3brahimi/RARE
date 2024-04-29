import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp
from Training.ModelTrainer import ModelTrainer
from Metric.RobustnessMetric import RobustnessMetric
from utils.training_utils import CustomLoss
import numpy as np
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
import json

physical_devices = tf.config.list_physical_devices('GPU')

tf.config.set_visible_devices(physical_devices[0],'GPU')  # for using the first GPU
tf.keras.mixed_precision.set_global_policy('mixed_float16')

num_noises = 20
# num_noises = 2
distribution = 'normal'
percentage = 0.5
eq_num = "I_12_4"
with open( f'./configs/equations/{eq_num}.json') as f:
    configs = json.load(f)

x_len = configs["num_samples"]

#### initialize the data generators
noise_model = NoiseGenerator(x_len, num_noises, distribution, percentage)
equation_str = configs["equation"]

input_features = configs['features']
num_inputs = len(input_features)
input_shape = num_inputs
# loss_function = 'custom_loss'
loss_function = "mse"

dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
metric = RobustnessMetric()


xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(configs['models'][0], metric_instance=metric)
print(xy_train[0].shape, xy_train[1].shape)

# m0 is adam and mse with no noise model
mse_trainer_non_dp = ModelTrainer().get_model("linear", shape_input=input_shape, loss_function="mse")
m0 = mse_trainer_non_dp.model

m0.compile(optimizer="adam", loss='mse')
# m0.load_weights(f"/home/qamar/workspace/crml/code/results_{eq_num}/loss_mse/normal/non_dp/linear/clean/models_all/model_1/model_weights.h5")
m0.load_weights(f"/home/qamar/workspace/crml/code/results/results_mse_I_12_4/linear/clean/models_all/model_1/model_weights.h5")

# # predict on the trainin data 
# y_pred = m0.predict(xy_test[0])
# print("MSE non_dp mse", np.mean((y_pred - xy_valid[1])**2))


# m1 is adam and custom loss with noise model gaussian
custom_trainer_non_dp = ModelTrainer().get_model("linear", shape_input=input_shape, loss_function="customloss")
m1 = custom_trainer_non_dp.model
customloss_non_dp = CustomLoss(model=m1, metric=metric, y_clean=xy_train[1], x_noisy=xy_noisy[0], len_input_features=input_shape, bl_ratio=3)
m1.compile(optimizer="adam", loss=customloss_non_dp)
# m1.load_weights(f"/home/qamar/workspace/crml/code/results_{eq_num}/loss_custom_loss/normal/non_dp/linear/clean/models_all/model_1/model_weights.h5")
m1.load_weights(f"/home/qamar/workspace/crml/code/results/results_custom_I_12_4/linear/clean/models_all/model_1/model_weights.h5")
# y_pred = m1.predict(xy_test[0])
# print("MSE non_dp custom", np.mean((y_pred - xy_valid[1])**2))


# m2 is dp adam and mse with no noise model
mse_trainer_dp = ModelTrainer().get_model("linear", shape_input=input_shape, loss_function="mse")
m2 = mse_trainer_dp.model

import tensorflow_privacy
optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
    l2_norm_clip=0.7,
    noise_multiplier=2.1,
    num_microbatches=1,
    learning_rate=0.001,
)
optimizer = "adam"
m2.compile(optimizer=optimizer, loss='mse')
m2.load_weights(f"/home/qamar/workspace/crml/code/results_{eq_num}/loss_mse/normal/dp/linear/clean/models_all/model_1/model_weights.h5")


# m3 is dp adam and custom loss with noise model gaussian
custom_trainer_dp = ModelTrainer().get_model("linear", shape_input=input_shape, loss_function="customloss")
m3 = custom_trainer_dp.model
customloss_dp = CustomLoss(model=m3, metric=metric, y_clean=xy_train[1], x_noisy=xy_noisy[0], len_input_features=input_shape, bl_ratio=3)

# import tensorflow_privacy
# optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
#     l2_norm_clip=0.7,
#     noise_multiplier=2.1,
#     num_microbatches=1,
#     learning_rate=0.001,
# )
# optimizer='adam'
#### dp
m3.compile(optimizer=optimizer, loss=customloss_dp)
m3.load_weights(f"/home/qamar/workspace/crml/code/results_{eq_num}/loss_custom_loss/normal/dp/linear/clean/models_all/model_1/model_weights.h5")

# m4 is dp adam and custom loss with noise model laplace
custom_trainer_dp = ModelTrainer().get_model("linear", shape_input=input_shape, loss_function="customloss")
m4 = custom_trainer_dp.model
customloss_dp = CustomLoss(model=m4, metric=metric, y_clean=xy_train[1], x_noisy=xy_noisy[0], len_input_features=input_shape, bl_ratio=3)
m4.compile(optimizer=optimizer, loss=customloss_dp)
m4.load_weights(f"/home/qamar/workspace/crml/code/results_{eq_num}/loss_custom_loss/laplace_dp/epsilon_1/linear/clean/models_all/model_1/model_weights.h5")
# y_pred = model_dp_mse.predict(xy_test[0])
# print("MSE dp mse", np.mean((y_pred - xy_valid[1])**2))
# y_pred = m3.predict(xy_test[0])
# print("MSE dp custom", np.mean((y_pred - xy_valid[1])**2))

# difference of weights between mse and custom, both non_dp
# Get the weights of the models
m0_weights = m0.get_weights()
m1_weights = m1.get_weights()
m2_weights = m2.get_weights()
m3_weights = m3.get_weights()
m4_weights = m4.get_weights()

m0_weights = [tf.convert_to_tensor(w) for w in m0_weights]
m1_weights = [tf.convert_to_tensor(w) for w in m1_weights]
m2_weights = [tf.convert_to_tensor(w) for w in m2_weights]
m3_weights = [tf.convert_to_tensor(w) for w in m3_weights]
m4_weights = [tf.convert_to_tensor(w) for w in m4_weights]

m0_weights = [tf.nn.softmax(w) for w in m0_weights]
m1_weights = [tf.nn.softmax(w) for w in m1_weights]
m2_weights = [tf.nn.softmax(w) for w in m2_weights]
m3_weights = [tf.nn.softmax(w) for w in m3_weights]
m4_weights = [tf.nn.softmax(w) for w in m4_weights]

# from scipy.special import kl_div

i = 0
print("Difference in weights between M0 and M1")


# for i in range(len(m0_weights)):
m0_m1_diff = np.linalg.norm(m0_weights[i] - m1_weights[i], ord=2)
# kl divergence instead of norm
# kl = tf.keras.losses.KLDivergence()(m1_weights[i], m0_weights[i])
m0_m1_kl = tf.keras.losses.KLDivergence()(m0_weights[i], m1_weights[i]).numpy()
    # kl = kl_div(m0_weights[i], m1_weights[i]).numpy()
# print(f'Difference in weights for layer {i}: {diff} KL: {kl}')

print("\n Difference in weights between M0 and M2")
# For each layer, calculate the difference between the weights
# for i in range(len(m0_weights)):
m0_m2_diff= np.linalg.norm(m0_weights[i] - m2_weights[i], ord=2)
# kl = tf.keras.losses.KLDivergence()(m2_weights[i], m0_weights[i]).numpy()
m0_m2_kl = tf.keras.losses.KLDivergence()(m0_weights[i], m2_weights[i]).numpy()
# kl = kl_div(m0_weights[i], m2_weights[i]).numpy()
# print(f'Difference in weights for layer {i}: {diff} KL: {kl}')

print("\n Difference in weights between M0 and M3")
# For each layer, calculate the difference between the weights
# for i in range(len(m0_weights)):
m0_m3_diff = np.linalg.norm(m0_weights[i] - m3_weights[i], ord=2)
# kl = tf.keras.losses.KLDivergence()(m3_weights[i], m0_weights[i]).numpy()
m0_m3_kl = tf.keras.losses.KLDivergence()(m0_weights[i], m3_weights[i]).numpy()
# kl = kl_div(m0_weights[i], m3_weights[i]).numpy()
# print(f'Difference in weights for layer {i}: {diff} KL: {kl}')

print("\n Difference in weights between M0 and M4")
# For each layer, calculate the difference between the weights
# for i in range(len(m0_weights)):
m0_m4_diff = np.linalg.norm(m0_weights[i] - m4_weights[i], ord=2)
# kl = tf.keras.losses.KLDivergence()(m4_weights[i], m0_weights[i]).numpy()
m0_m4_kl = tf.keras.losses.KLDivergence()(m0_weights[i], m4_weights[i]).numpy()
# kl = kl_div(m0_weights[i], m4_weights[i]).numpy()
# print(f'Difference in weights for layer {i}: {diff} KL: {kl}')
    
print("\n Difference in weights between M2 and M3")
# For each layer, calculate the difference between the weights
# for i in range(len(m0_weights)):
m2_m3_diff =np.linalg.norm(m2_weights[i] - m3_weights[i], ord=2)
m2_m3_kl = tf.keras.losses.KLDivergence()(m2_weights[i], m3_weights[i]).numpy()
    # kl = kl_div(m2_weights[i], m3_weights[i]).numpy()
    # print(f'Difference in weights for layer {i}: {diff} KL: {kl}')

print("\n Difference in weights between M2 and M4")
# For each layer, calculate the difference between the weights
# for i in range(len(m0_weights)):
m2_m4_diff =np.linalg.norm(m2_weights[i] - m4_weights[i], ord=2)
    # kl = tf.keras.losses.KLDivergence()(m4_weights[i], m2_weights[i]).numpy()
m2_m4_kl = tf.keras.losses.KLDivergence()(m2_weights[i], m4_weights[i]).numpy()

print("\n Difference in weights between M2 and M1")
m2_m1_diff = np.linalg.norm(m2_weights[i] - m1_weights[i], ord=2)
m2_m1_kl = tf.keras.losses.KLDivergence()(m2_weights[i], m1_weights[i]).numpy()
    
print("\n Difference in weights between M2 and M0")
m2_m0_diff = np.linalg.norm(m2_weights[i] - m0_weights[i], ord=2)

m2_m0_kl = tf.keras.losses.KLDivergence()(m2_weights[i], m0_weights[i]).numpy()

print("\n Difference in weights between M3 and M2")
m3_m2_diff = np.linalg.norm(m3_weights[i] - m2_weights[i], ord=2)

m3_m2_kl = tf.keras.losses.KLDivergence()(m3_weights[i], m2_weights[i]).numpy()

print("\n Difference in weights between M4 and M2")
m4_m2_diff = np.linalg.norm(m4_weights[i] - m2_weights[i], ord=2)
m4_m2_kl = tf.keras.losses.KLDivergence()(m4_weights[i], m2_weights[i]).numpy()

print("\n Difference in weights between M1 and M2")
m1_m2_diff = np.linalg.norm(m1_weights[i] - m2_weights[i], ord=2)
m1_m2_kl = tf.keras.losses.KLDivergence()(m1_weights[i], m2_weights[i]).numpy()

print("\n Difference in weights between M1 and M3")
m1_m3_diff = np.linalg.norm(m1_weights[i] - m3_weights[i], ord=2)
m1_m3_kl = tf.keras.losses.KLDivergence()(m1_weights[i], m3_weights[i]).numpy()
    
print("\n Difference in weights between M3 and M1")
m3_m1_diff = np.linalg.norm(m3_weights[i] - m1_weights[i], ord=2)
m3_m1_kl = tf.keras.losses.KLDivergence()(m3_weights[i], m1_weights[i]).numpy()

print("\n Difference in weights between M3 and M4")
m3_m4_diff = np.linalg.norm(m3_weights[i] - m4_weights[i], ord=2)
m3_m4_kl = tf.keras.losses.KLDivergence()(m3_weights[i], m4_weights[i]).numpy()
# write these in table of columns models --> name of the variable, kl, norm
import pandas as pd

data = {
    'model': ['m0_m1', 'm0_m2', 'm0_m3', 'm0_m4', 'm2_m3', 'm2_m4', 'm2_m1', 'm2_m0', 'm3_m2', 'm4_m2', 'm1_m2', 'm1_m3', 'm3_m1', 'm3_m4'],
    'kl': [m0_m1_kl, m0_m2_kl, m0_m3_kl, m0_m4_kl, m2_m3_kl, m2_m4_kl, m2_m1_kl, m2_m0_kl, m3_m2_kl, m4_m2_kl, m1_m2_kl, m1_m3_kl, m3_m1_kl, m3_m4_kl],
    'norm': [m0_m1_diff, m0_m2_diff, m0_m3_diff, m0_m4_diff, m2_m3_diff, m2_m4_diff, m2_m1_diff, m2_m0_diff, m3_m2_diff, m4_m2_diff, m1_m2_diff, m1_m3_diff, m3_m1_diff, m3_m4_diff]
}
    
df = pd.DataFrame(data)
# save it to a csv file
df.to_csv(f'./results_{eq_num}/weight_diff_epsilon_1.csv')