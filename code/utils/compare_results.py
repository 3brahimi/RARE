import pandas as pd    

def read_rm(model_results_path, models_set, distance="Euclidean"):
    rms =[]
    
    for nmodel in models_set:
        path = f"{model_results_path}/nmodel_{nmodel}/robustness_values.txt"    
        with open(path, "r") as f:
            rm = eval(f.read())
            rms.append(rm[distance]["Ratio"])

    rms = [sum(rms[:i+1]) for i in range(len(rms))]
    return rms

def compare_models(models_results_path, models_set, models_set_name, x_measure = "variance", distance="Euclidean"):
    rm_linear = read_rm(model_results_path=f"{models_results_path}/linear_mse", models_set=models_set, distance=distance)
    rm_linear_msep = read_rm(model_results_path=f"{models_results_path}/linear_msep", models_set=models_set, distance=distance)
    rm_RF = read_rm(model_results_path=f"{models_results_path}/RF_mse", models_set=models_set, distance=distance)
    rm_LR = read_rm(model_results_path=f"{models_results_path}/LR_mse", models_set=models_set, distance=distance)
    rm_cnn = read_rm(model_results_path=f"{models_results_path}/cnn_mse", models_set=models_set, distance=distance)
    rm_cnn_msep = read_rm(model_results_path=f"{models_results_path}/cnn_msep", models_set=models_set, distance=distance)
    x = []
    if x_measure == "variance":
        x = [0.1, 0.4, 0.6, 0.8, 1]
    elif x_measure == "percentage":
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    df = pd.DataFrame()
    df[x_measure] = x
    df["rm-linear-mse"] = rm_linear
    df["rm-linear-msep"] = rm_linear_msep
    df["rm-RF-mse"] = rm_RF
    df["rm-LR-mse"] = rm_LR
    df["rm-cnn-mse"] = rm_cnn
    df["rm-cnn-msep"] = rm_cnn_msep
    df.to_csv(f"./{models_results_path}/rm-vs-noise-{x_measure}-{models_set_name}.txt", header=True, index=False)