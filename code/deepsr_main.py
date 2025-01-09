import pickle
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from utils import functions, pretty_print
from Training.SymbolicNetwork import SymbolicNetL0
from Metric.RobustnessMetric import RobustnessMetric
from utils.dists import L2_distance
import argparse
import time
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator

class CustomLossPT(nn.Module):
    def __init__(self, model, metric, x_noisy, y_clean, x_clean, bl_dist=1.0, device=None):
        super().__init__()
        self.model = model
        self.metric = metric
        self.x_noisy = x_noisy  
        self.y_clean = y_clean
        self.x_clean = x_clean
        self.bl_dist = bl_dist
        self.mse = nn.MSELoss()
        self.device = device
    
    def forward(self, y_true, y_pred):
        # Base MSE loss
        base_loss = self.mse(y_true, y_pred)
        # Forward pass for x_noisyy_pred.shape)
       
        y_noisy_pred = torch.stack([self.model(x.to(self.device)) for x in self.x_noisy])
        
        # y_noisy_reshaped = y_noisy_pred.view(self.x_noisy.shape[0], self.x_noisy.shape[1], 1)
        
        gy = self.extract_g(y_noisy_pred, self.y_clean)
       
        y_clean_scaled, gy_scaled = self.rescale_vectors(self.y_clean, gy)
        gy_y_dist = torch.sqrt(torch.sum((gy_scaled - y_clean_scaled) ** 2))
        ratio = gy_y_dist / self.bl_dist
        penalty = base_loss * ratio
        
        return base_loss + penalty
    
    
    def extract_g(self, y_noisy, y_clean):
        y_noisy_min, y_noisy_max = torch.min(y_noisy, dim=0).values, torch.max(y_noisy, dim=0).values
        
        y_noisy_min = y_noisy_min.to(torch.float64)
        y_noisy_max = y_noisy_max.to(torch.float64)
        
        y_noisy_min = y_noisy_min.to(self.device)
        y_noisy_max = y_noisy_max.to(self.device)
        y_clean = y_clean.to(self.device)
  
        y_dist_min = torch.sqrt(torch.square(y_noisy_min - y_clean.reshape(y_clean.shape[0], y_noisy_min.shape[1])))
        y_dist_max = torch.sqrt(torch.square(y_noisy_max - y_clean.reshape(y_clean.shape[0], y_noisy_max.shape[1])))
  
        gy = torch.maximum(y_dist_min, y_dist_max)

        mask_min = torch.eq(y_dist_min, gy)
        mask_max = torch.eq(y_dist_max, gy)
        
        fallback_value = torch.tensor(torch.finfo(torch.float32).min, device=y_noisy.device)
        g_points_min = torch.where(mask_min, y_noisy_min, fallback_value)
        g_points_max = torch.where(mask_max, y_noisy_max, fallback_value)
        g_points = torch.maximum(g_points_min, g_points_max)

        return g_points
  
    def rescale_vectors(self, x, y):
        x = x.to(x.device)
        y = y.to(x.device)
        min_val = torch.min(x)
        min_val = min_val.to(x.device)
        max_val = torch.max(x)
        max_val = max_val.to(x.device)
        x = (x - min_val) / (max_val - min_val)
        y = (y - min_val) / (max_val - min_val)
        return x, y
    
##############################################
# Data generation functions & training pipeline
# combine multi_var_deepsr and deep_sr_benchmark
##############################################

def custom_train_function(configs, res_folder, loss_function="mse", bl_dist=1, models_no=3):
    """
    Example combined function that generates data like multi_var_deepsr,
    then trains using a symbolic approach as in deep_sr_benchmark.
    """
    x_len = configs["num_samples"]
    num_noises = 10
    distribution = "normal"
    variance = 0.2
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    print("Equation:", equation_str, "Features:", input_features, "Loss function:", loss_function)
    # 2) Create noise model & dataset generator
    noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=distribution,
                                 variance=variance, epsilon=0.5)
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    configs["models"][0]['type'] = "symbolic"
    config = configs["models"][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # x_noisy, y_noisy = dataset_generator.generate_noisy_data()
    # x_clean, y_clean = dataset_generator.generate_clean_data()
    xy_train_i, xy_valid_i, xy_test_i, xy_noisy_i, xy_clean_i, gx_gy_i, indices_i = dataset_generator.split(config, metric_instance=metric)
    x_train = xy_train_i[0]
    y_train = xy_train_i[1]
    x_valid = xy_valid_i[0]
    y_valid = xy_valid_i[1]
    x_test = xy_test_i[0]
    y_test = xy_test_i[1]
    x_noisy = xy_noisy_i[0]
    y_noisy = xy_noisy_i[1]
    x_clean = xy_clean_i[0]
    y_clean = xy_clean_i[1]

    x_noisy_torch = torch.tensor(x_noisy, dtype=torch.float).to(device)

    x_clean_torch = torch.tensor(x_clean, dtype=torch.float).to(device)
    y_clean_torch = torch.tensor(y_clean, dtype=torch.float).to(device)
    x_train_torch = torch.tensor(x_train, dtype=torch.float).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float).to(device)
    x_valid_torch = torch.tensor(x_valid, dtype=torch.float).to(device)
    y_valid_torch = torch.tensor(y_valid, dtype=torch.float).to(device)
    
    # 3) Create model instance (SymbolicNetL0, for example)
    activation_funcs = [
        *[functions.Constant()] * 2,
        *[functions.Identity()] * 4,
        *[functions.Square()] * 4,
        *[functions.Sin()] * 2,
        *[functions.Exp()] * 2,
        *[functions.Sigmoid()] * 2,
        *[functions.Product(1.0)] * 2
    ]

    # generate seeds for the models initialization
    random.seed(42)  # Set a fixed seed for reproducibility
    seeds = [random.randint(0, 10000) for _ in range(models_no)]
    
    for i in range(models_no):    
        net = SymbolicNetL0(2, in_dim=num_inputs, funcs=activation_funcs, seed=seeds[i]).to(device)
        # 4) Train the model (similar to 'train' method in deep_sr_benchmark)
        # Define loss function, optimizer, and scheduler
        
        if loss_function == "mse":
            #### mse criterion
            criterion = nn.MSELoss()
        elif loss_function == "custom_loss" or loss_function == "custom" or loss_function == "pl":
            # TODO: use only validation data for the custom loss
            criterion = CustomLossPT(
                model=net,
                metric=metric,
                x_noisy=torch.tensor(x_noisy, dtype=torch.float).to(device),
                y_clean=torch.tensor(y_clean, dtype=torch.float).to(device),
                x_clean=torch.tensor(x_clean, dtype=torch.float).to(device),
                bl_dist=bl_dist,
                device=device,
            )
        else:
            raise ValueError(f"Loss type {loss_function} not recognized")
        
        optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-2, alpha=0.9, eps=1e-10, momentum=0.0, centered=False)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.1)
        
        # Arrays to keep track of various quantities as a function of epoch
        loss_list = []          # Total loss (MSE + regularization)
        mse_list = []         # Train MSE
        reg_list = []           # Regularization
        valid_test_list = []    # Test mse
        best_test_loss = float('inf')
        
        early_stop_counter = 0
        early_stop_delta = 1e-4  # Define early_stop_delta with a small value
        patience = 50  # Define patience with a suitable value
        
        for epoch in range(1001):
            optimizer.zero_grad()
            # outputs = net(x_clean_torch)
            outputs = net(x_train_torch)
            outputs = outputs.to(device).squeeze()
            # y_noisy_torch = y_noisy_torch.to(outputs.device)  # Ensure y_noisy_torch is on the same device as outputs
            # mse_loss = criterion(outputs, y_noisy_torch)
        
            epoch_loss = criterion(y_train_torch, outputs)
            # epoch_loss = criterion(y_clean_torch, outputs)
            
            if loss_function == "mse":
                mse_list.append(epoch_loss.item())
            else:
                # mse_loss = F.mse_loss(outputs, y_clean_torch)
                mse_loss = F.mse_loss(outputs, y_train_torch)
                mse_list.append(mse_loss.item())
            
            reg_loss = net.get_loss()
            loss = epoch_loss + 5e-3 * reg_loss
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            
            if epoch == 2000 or epoch == 10001:
                scheduler.step()
            
            if epoch % 1000 == 0:
                with torch.no_grad():
                    # for noisy_sample in x_noisy_torch:
                    # noisy_sample = noisy_sample.to(device)
                    # test_outputs = net(noisy_sample)
                    test_outputs = net(x_valid_torch)
                    test_outputs = test_outputs.to(device).squeeze()
                    # test_loss = F.mse_loss(test_outputs, y_clean_torch)
                    test_loss = F.mse_loss(test_outputs, y_valid_torch)
                    if loss_function == "mse":
                        print(f"Epoch: {epoch}, Total Loss: {loss.item()}, Test Loss: {test_loss.item()}")
                    else:
                        print(f"Epoch: {epoch}, Total Loss: {loss.item()}, Test Loss: {test_loss.item()}, MSE Loss: {mse_loss.item()}")
            
                # Early stopping check
                if test_loss.item() < best_test_loss - early_stop_delta:
                    best_test_loss = test_loss.item()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} with validation loss {best_test_loss}")
                    break
            
        # 5) Evaluate or save results
        plot_folder = os.path.join(res_folder, "plots")
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
            
        with torch.no_grad():
            weights = net.get_weights()
            expr = pretty_print.network(weights, activation_funcs, input_features)
            print("Final expression:", expr)
            # evaluate_and_plot(net, None, (x_noisy_torch, y_noisy_torch), plot_folder)
            
        # Save the trained model
        # Save results
        model_save_path = os.path.join(res_folder, "SymbolicNetL0", "clean", "models_all", f"model{i}")
        print("Model save path:", model_save_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        y_pred = net(x_clean_torch).to(device).squeeze()
        validation_mse = F.mse_loss(y_pred, y_clean_torch)
        # save validation mse to txt file; append to existing file, if it exists, otherwise create a new file
        with open(f"{model_save_path}/validation_mse.txt", "a") as f:
            f.write(f"Validation MSE: {validation_mse.item()}\n")
        
        model_file = os.path.join(model_save_path, f"model{i}.pickle")
        results = {
            "weights": net.get_weights(),
            "loss_list": loss_list,
            "mse_list": mse_list,
            "expr": expr,
            "runtime": time.time()  # Assuming you have a runtime variable
        }
        with open(model_file, "wb+") as f:
            pickle.dump(results, f)
        
        print(f"Model saved to {model_file}")

def load_and_test(models_folder, models_no, equation_str, noise_model, input_features, x_len):
    """Load a saved model from a pickle file and test it on new data."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use cuda:", use_cuda, "Device:", device)

    for no in models_no:
        # Load the saved model
        with open(f"{models_folder}/model{no}/model{no}.pickle", "rb") as f:
            results = pickle.load(f)
        weights = results["weights"]
        expr = results["expr"]
        print("Loaded model expression:", expr)
        
        test_dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
        x_clean_orig, y_clean_orig = test_dataset_generator.generate_dataset()
        print("1. Data shapes:", x_clean_orig.shape, y_clean_orig.shape)
        # apply the meshgrid to x and y clean
        y_clean_orig = y_clean_orig.ravel()
        x_clean_orig, y_clean_orig = test_dataset_generator.meshgrid_x_y(x_clean_orig)

        num_input_features = x_clean_orig.shape[1]
        
        if num_input_features > 1:
            x_clean_orig = x_clean_orig.reshape(x_clean_orig.shape[0], -1).T
        y_clean_orig = y_clean_orig.ravel()
        
        original_num_samples = test_dataset_generator.num_samples
        test_dataset_generator.num_samples = x_clean_orig.shape[0]
        test_dataset_generator.noise_generator.num_samples = test_dataset_generator.num_samples
        target_feats_ids = [el for el in range(len(input_features))]
        
        random_seeds = [random.randint(0, 1000) for _ in range(len(target_feats_ids))]
        initial_weights = [torch.tensor(w).to(device) for w in weights]
        
        net = SymbolicNetL0(2, in_dim=2, funcs=functions.default_func, initial_weights=initial_weights).to(device)

        # Ensure the input dimension of x matches the expected input dimension
        if x_clean_orig.shape[1] != net.hidden_layers[0].in_dim:
            raise ValueError(f"Expected input dimension {net.hidden_layers[0].in_dim}, but got {x_clean_orig.shape[1]}")

        # Generate noisy data using modulate_clean
        x_noisy, y_noisy = test_dataset_generator.modulate_clean(x_clean_orig, y_clean_orig, target_feat_idx=target_feats_ids, random_seeds=random_seeds)
        
        # Test the network
        with torch.no_grad():
            test_outputs = net(torch.tensor(x_clean_orig, dtype=torch.float).to(device))
            test_loss = F.mse_loss(test_outputs, torch.tensor(y_clean_orig, dtype=torch.float).to(device))
            error_test_val = test_loss.item()

            # save test error to txt file
            with open(f"{models_folder}/model{no}/test_error.txt", "w") as f:
                f.write(f"Test error: {error_test_val}")
        # Robustness testing
        noisy_samples = 20
        y_hat = torch.zeros((noisy_samples, y_clean_orig.size))
        with torch.no_grad():
            for noisy_i in range(noisy_samples):
                temp = net(torch.tensor(x_noisy[noisy_i], dtype=torch.float).to(device))
                y_hat[noisy_i] = temp.squeeze()
        
        metric = RobustnessMetric()
        outer_dists = ["L2"]
        f_weights = [1,1]
        y_hat_np = y_hat.cpu().numpy()

        rm = metric.calculate_metric(x_clean_orig, y_clean_orig, x_hat=x_noisy, y_hat=y_hat_np, outer_dist=outer_dists, weights=f_weights, path=f"{models_folder}/robustness")        
        print("Robustness metric:", rm)
        # save rm to txt file
        rm_folder = f"{models_folder}/model{no}/robustness"
        if not os.path.exists(rm_folder):
            os.makedirs(rm_folder)
        with open(f"{rm_folder}/rm.txt", "w") as f:
            f.write(f"Robustness metric: {rm}")
        

def main(res_folder, json_path, loss_function, noise_type, epsilon=0.5):
    """
    Example main combining multi_var_deepsr approach with symbolic training.
    """
    configs = json_path
    # 1) Load config, set noise, read equation
    with open(configs) as f:
        configs = json.load(f)
    eq_res_folder = f'{res_folder}'

    # read the bl_distance to be used in the custom loss function
    if loss_function == "custom" or loss_function == "custom_loss" or loss_function == "pl":
        # read txt file from bl_dist_path to get baseline distance
        bl_dist_path = f"{res_folder}baseline/training/rm.txt"
        with open(bl_dist_path, 'r') as f:
            bl_dist = eval(f.read())["Output distance"]
    else:
        bl_dist = 1.0
        
    custom_train_function(configs, eq_res_folder, loss_function, bl_dist)
    exit()
    training_type = "clean"
    model_type = "SymbolicNetL0"
    models_folder = f"{eq_res_folder}/{model_type}/{training_type}/models_all"
    x_len = configs["num_samples"]
    num_noises = 20
    distribution = "normal"
    variance = 0.5
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    print("Equation:", equation_str, "Features:", input_features)
    # 2) Create noise model & dataset generator
    noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=distribution,
                                 variance=variance, epsilon=0.5)
    # dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    
    load_and_test(models_folder, [0,1,2], equation_str, noise_model, input_features, x_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test_custom')
    # parser.add_argument("--json-path", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--loss-function", type=str, default="mse", help="Loss function to use")
    parser.add_argument("--noise-type", type=str, default="normal", help="Type of noise to add")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Epsilon value for noise")
    
    eqs ={
        # "I_25_13",
        "I_6_2",
    }
    args = parser.parse_args()
    noise_type = "normal"
    for eq in eqs:
        json_path = f"./configs/equations/{eq}.json"
        results_dir = f"./results_{eq}_recent/loss_{args.loss_function}/{noise_type}/non-dp/"
        main(results_dir, json_path, args.loss_function, noise_type, args.epsilon)
    