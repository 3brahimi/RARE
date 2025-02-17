import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
from Training.ModelTrainer import ModelTrainer
from utils.compare_results import compare_models
from Metric.RobustnessMetric import RobustnessMetric
from Metric.weights_estimation import estimate_weights


### ---------- MODEL LOADING FUNCTIONS ---------- ###
def load_models(models_folder):
    """Load multiple models from a folder."""
    models = []
    models_files = [f for f in os.listdir(models_folder)]
    
    for model_name in models_files:
        model_path = f"{models_folder}/{model_name}"
        model = load_one_model(model_path)
        models.append(model)
    
    return models


def load_one_model(model_folder):
    """Load a single model based on the folder name."""
    model_name = model_folder.split("/")[-1]
    model_type = model_name.split("_")[0]
    loss_function = model_name.split("_")[-1]
    input_shape = 1
    
    trainer = ModelTrainer().get_model(model_type, shape_input=input_shape, loss_function=loss_function)
    model = trainer.model
    
    if model_type in ["RF", "LR"]:
        model = trainer.load_model(model_folder)
    else:
        model.compile(optimizer='adam', loss=loss_function)
        model.load_weights(f"{model_folder}/model_weights.h5")
           
    return model


### ---------- ROBUSTNESS EVALUATION ---------- ###
def evaluate_robustness_model(model, dataset_generator, config, metric, random_seeds_all, robustness_res_path, weights):
    """Evaluate model robustness and save results."""
    print("Evaluating robustness of models..")
    x_clean, y_clean = dataset_generator.generate_dataset()
    target_features = config["noisy_input_feats"]
    
    x_noisy, y_noisy = dataset_generator.modulate_clean(x_clean, y_clean, target_feat_idx=target_features, random_seeds=random_seeds_all)

    y_noisy_pred = y_noisy.copy()
    for idx_shape, x_noise_vector in enumerate(x_noisy):
        y_noise_vector = model.predict(x_noise_vector)             
        y_noisy_pred[idx_shape, :] = y_noise_vector.flatten()     

    if np.min(x_clean) != np.max(x_clean) and np.min(y_clean) != np.max(y_clean):                              
        rm = metric.calculate_metric(x_clean, y_clean, x_hat=x_noisy, y_hat=y_noisy_pred, outer_dist=["Euclidean", "L1"], weights=[1], path=f"{robustness_res_path}", vis=False, save=False)
    else:
        print("Skipping metric calculation due to identical min and max values in input or output.")
        rm = {"Output distance": 1}
        
    try:
        with open(f"{robustness_res_path}/robustness_values.txt", "w") as f:
            json.dump(rm, f, indent=4)
            print("Results saved in", f"{robustness_res_path}/robustness_values.txt")
    except Exception as e:
        print(f"Error writing to file {robustness_res_path}/robustness_values.txt: {e}")


### ---------- WEIGHT ESTIMATION ---------- ###
def get_weights(model_path, inputs, dataset_generator, num_samples, loss_function="mse", metric=None, x_noisy=None, y_clean=None, model_type="expression"):
    """Estimate model weights based on loss function."""
    if loss_function == "msep":
        weights = estimate_weights(model_path, inputs, dataset_generator, num_samples=num_samples, loss_function=loss_function, 
                                   metric=metric, x_noisy=x_noisy, bl_ratio=1, y_clean=y_clean, model_type=model_type)
    else:
        weights = estimate_weights(model_path, inputs, dataset_generator, num_samples=num_samples, model_type=model_type)
    return weights


### ---------- TESTING MODELS ---------- ###
def test_models(models_folder, configs):
    """Test multiple models with different noise configurations."""
    with open(configs) as f:
        configs = json.load(f)
    
    input_features = configs["features"]
    x_len = configs["num_samples"]
    equation_str = configs["equation"]
    models_files = [f for f in os.listdir(models_folder)]
    
    nmodels_path = "./configs/testing_noise_models.json"
    with open(nmodels_path) as f:
        nmodels = json.load(f)
    
    config = configs["models"][0]
    metric = RobustnessMetric()
    
    for model_name in models_files:
        model_path = f"{models_folder}/{model_name}"
        model = load_one_model(model_path)

        for nmodel in nmodels:
            num_noises = nmodel["num_noises"]
            variance = nmodel["variance"]
            distribution = nmodel["noise_type"]
            nmodel_name = nmodel["name"]
            percentage = nmodel["percentage"]

            test_noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=distribution, variance=variance, percentage=percentage)
            test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)

            model_robustness_results = f"{models_folder}_robustness_testing/{model_name}/nmodel_{nmodel_name}"
            os.makedirs(model_robustness_results, exist_ok=True)

            evaluate_robustness_model(model, test_dataset_generator, config, metric, [22], model_robustness_results, [1])
            
import os
import pandas as pd
import matplotlib.pyplot as plt

### ---------- PLOTTING FUNCTION ---------- ###
def plot_results(variance_files, percentage_files, x_labels, title_labels):
    """Plot robustness metrics with 2-row layout (Variances on top, Percentages below) and save the figure."""
    
    fig, axes = plt.subplots(2, 3, figsize=(7, 6), sharey=True)
    
    datasets = [variance_files, percentage_files]
    y_limits = [6, 10]  
    
    legend_labels = {}
    x_ticks_variances = [0.2, 0.4, 0.6, 0.8, 1]
    x_ticks_percentages = [20, 40, 60, 80, 100]
    for row, dataset_group in enumerate(datasets):  
        for col, (dataset, xlabel, title) in enumerate(zip(dataset_group, x_labels[row], title_labels[row])):
            
            ax = axes[row, col]

            if os.path.exists(dataset):
                data = pd.read_csv(dataset)                
                # Plot each model's robustness
                line_rf, = ax.plot(data["variance" if row == 0 else "percentage"], data["rm-RF-mse"], color="orange", linewidth=2.5)
                line_mlp, = ax.plot(data["variance" if row == 0 else "percentage"], data["rm-linear-mse"], color="blue", linewidth=2.5)
                line_mlp_p, = ax.plot(data["variance" if row == 0 else "percentage"], data["rm-linear-msep"], linestyle="dashed", color="blue", linewidth=2.5)
                line_cnn, = ax.plot(data["variance" if row == 0 else "percentage"], data["rm-cnn-mse"], color="purple", linewidth=1.5)
                line_cnn_p, = ax.plot(data["variance" if row == 0 else "percentage"], data["rm-cnn-msep"], linestyle="dashed", color="purple", linewidth=1.5)
                line_lr, = ax.plot(data["variance" if row == 0 else "percentage"], data["rm-LR-mse"], linestyle="dotted", color="green", linewidth=2.5)

                ax.set_title(title, fontsize=14, pad=10)
                ax.set_xlabel(xlabel, fontsize=12)
                
                if row == 0 and col == 0:
                    legend_labels = {
                        "RF": line_rf,
                        r"$\mathrm{MLP}(\mathrm{MSE})$": line_mlp,
                        r"$\mathrm{MLP}(\mathrm{MSE}_\mathcal{P})$": line_mlp_p,
                        r"$\mathrm{CNN}(\mathrm{MSE})$": line_cnn,
                        r"$\mathrm{CNN}(\mathrm{MSE}_\mathcal{P})$": line_cnn_p,
                        "LR": line_lr
                    }
            ax.set_ylim(0, y_limits[row])
            ax.set_autoscale_on(False)  # Disable auto rescaling

            # ✅ Adjust x-axis ticks to match original TikZ figure
            if row == 0:  # Variance row
                ax.set_xticks(x_ticks_variances)
            else:  # Percentage row
                ax.set_xticks(x_ticks_percentages)
                ax.set_xticklabels([str(tick) for tick in x_ticks_percentages])  # Ensure proper labels


    plt.draw()
    fig.text(0.02, 0.49, r"$\mathcal{R}$", va='center', rotation='vertical', fontsize=10)    
    fig.legend(legend_labels.values(), legend_labels.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=10, frameon=False)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("./robustness_evaluation_figure2.png", dpi=300,bbox_inches="tight")
    plt.show()

def generate_plots():
    plot_results(
    variance_files= ["./checkpoints_robustness_testing/rm-vs-noise-variance-N0.txt",
     "./checkpoints_robustness_testing/rm-vs-noise-variance-N1.txt",
     "./checkpoints_robustness_testing/rm-vs-noise-variance-N2.txt"],
    percentage_files= ["./checkpoints_robustness_testing/rm-vs-noise-percentage-N3.txt",
        "./checkpoints_robustness_testing/rm-vs-noise-percentage-N4.txt",
        "./checkpoints_robustness_testing/rm-vs-noise-percentage-N5.txt"],
    x_labels=[
    [r"$\sigma^2$", "b", "b"],  # Top row (variances)
    [r"$\mathcal{P}$", r"$\mathcal{P}$", r"$\mathcal{P}$"]  # Bottom row (percentages)
    ],
    title_labels=[
    [r"$\mathcal{N}_0$", r"$\mathcal{N}_1$", r"$\mathcal{N}_2$"],  # Top row (variances)
    [r"$\mathcal{N}_3$", r"$\mathcal{N}_4$", r"$\mathcal{N}_5$"]  # Bottom row (percentages)
    ]
    )
    
if __name__ == '__main__':
    json_files = [
        "I_6_2a.json",
        ]
    N0_set = ["n0", "n1", "n2", "n3", "n4"] # normal-variances
    N1_set = ["n5", "n6", "n7", "n8", "n9"] # uniform-variances
    N2_set = ["n10", "n11", "n12", "n13", "n14"] # laplace-variances
    N3_set = ["n15", "n16", "n17", "n18", "n19", "n20", "n21", "n22", "n23", "n24"] # normal-percentages 
    N4_set = ["n25", "n26", "n27", "n28", "n29", "n30", "n31", "n32", "n33", "n34"] # uniform-percentages
    N5_set = ["n35", "n36", "n37", "n38", "n39", "n40", "n41", "n42", "n43", "n44"] # laplace-percentages
    
    test_models("./checkpoints", "./configs/equations/I_6_2a.json")
    
    compare_models("./checkpoints_robustness_testing", N0_set, "N0", x_measure="variance")
    compare_models("./checkpoints_robustness_testing", N1_set, "N1", x_measure="variance")
    compare_models("./checkpoints_robustness_testing", N2_set, "N2", x_measure="variance")
    compare_models("./checkpoints_robustness_testing", N3_set, "N3", x_measure="percentage")
    compare_models("./checkpoints_robustness_testing", N4_set, "N4", x_measure="percentage")
    compare_models("./checkpoints_robustness_testing", N5_set, "N5", x_measure="percentage")


    generate_plots()
