
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp

from Training.ModelTrainer import ModelTrainer
from Metric.RobustnessMetric import RobustnessMetric
from utils.training_utils import CustomLoss
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator

# Setup GPU and precision
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load configuration
eq_num = "I_12_4"
with open(f'./configs/equations/{eq_num}.json') as f:
    configs = json.load(f)

x_len = configs["num_samples"]
input_features = configs["features"]
input_shape = len(input_features)
equation_str = configs["equation"]

# Initialize generators
noise_model = NoiseGenerator(x_len, num_noises=20, distribution='normal', percentage=0.5)
dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
metric = RobustnessMetric()

# Generate datasets
xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(configs['models'][0], metric_instance=metric)

# Model loader function
def load_model(name, loss_fn, custom_loss=None, dp=False, weight_path=None):
    trainer = ModelTrainer().get_model(name, shape_input=input_shape, loss_function=loss_fn)
    model = trainer.model
    optimizer = "adam"
    if dp:
        optimizer = tfp.DPKerasAdamOptimizer(
            l2_norm_clip=0.7,
            noise_multiplier=2.1,
            num_microbatches=1,
            learning_rate=0.001
        )
    model.compile(optimizer=optimizer, loss=custom_loss if custom_loss else loss_fn)
    model.load_weights(weight_path)
    return model

# Load models
m0 = load_model("linear", "mse", weight_path=f"/home/qamar/workspace/crml/code/results/results_mse_I_12_4/linear/clean/models_all/model_1/model_weights.h5")
m1 = load_model("linear", "customloss",
                custom_loss=CustomLoss(m1 := ModelTrainer().get_model("linear", shape_input=input_shape, loss_function="customloss").model,
                                       metric, xy_train[1], xy_noisy[0], input_shape, bl_ratio=3),
                weight_path=f"/home/qamar/workspace/crml/code/results/results_custom_I_12_4/linear/clean/models_all/model_1/model_weights.h5")
m2 = load_model("linear", "mse", dp=True, weight_path=f"/home/qamar/workspace/crml/code/results_{eq_num}/loss_mse/normal/dp/linear/clean/models_all/model_1/model_weights.h5")
m3 = load_model("linear", "customloss",
                custom_loss=CustomLoss(m3 := ModelTrainer().get_model("linear", shape_input=input_shape, loss_function="customloss").model,
                                       metric, xy_train[1], xy_noisy[0], input_shape, bl_ratio=3),
                dp=True,
                weight_path=f"/home/qamar/workspace/crml/code/results_{eq_num}/loss_custom_loss/normal/dp/linear/clean/models_all/model_1/model_weights.h5")
m4 = load_model("linear", "customloss",
                custom_loss=CustomLoss(m4 := ModelTrainer().get_model("linear", shape_input=input_shape, loss_function="customloss").model,
                                       metric, xy_train[1], xy_noisy[0], input_shape, bl_ratio=3),
                dp=True,
                weight_path=f"/home/qamar/workspace/crml/code/results_{eq_num}/loss_custom_loss/laplace_dp/epsilon_1/linear/clean/models_all/model_1/model_weights.h5")

# Process model weights
def softmax_weights(model):
    return [tf.nn.softmax(tf.convert_to_tensor(w)) for w in model.get_weights()]

weights = {name: softmax_weights(model) for name, model in zip(['m0', 'm1', 'm2', 'm3', 'm4'], [m0, m1, m2, m3, m4])}

# Weight comparisons
def compare_weights(w1, w2):
    i = 0  # Use first layer for simplicity
    norm_diff = np.linalg.norm(w1[i] - w2[i], ord=2)
    kl_div = tf.keras.losses.KLDivergence()(w1[i], w2[i]).numpy()
    return norm_diff, kl_div

comparisons = [
    ('m0', 'm1'), ('m0', 'm2'), ('m0', 'm3'), ('m0', 'm4'),
    ('m2', 'm3'), ('m2', 'm4'), ('m2', 'm1'), ('m2', 'm0'),
    ('m3', 'm2'), ('m4', 'm2'), ('m1', 'm2'), ('m1', 'm3'),
    ('m3', 'm1'), ('m3', 'm4')
]

data = {
    'model': [],
    'kl': [],
    'norm': []
}

for a, b in comparisons:
    norm, kl = compare_weights(weights[a], weights[b])
    data['model'].append(f"{a}_{b}")
    data['kl'].append(kl)
    data['norm'].append(norm)

df = pd.DataFrame(data)
output_dir = Path(f'./results_{eq_num}')
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_dir / "weight_diff_epsilon_1.csv", index=False)
