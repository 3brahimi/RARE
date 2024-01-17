from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import keras


def estimate_weights(model_path, inputs, dataset_generator, training_type="clean", num_samples=100, model_type="Linear"):
    """
    Estimate the weights of the input features.

    Parameters:
    model (object): path to the trained model.
    inputs (obj): The input data, which is a dictionary of the form {feature_name: {range: [min, max], type: data_type}}.
    Returns:
    ndarray: The estimated weights of the input features.
    """
    
    if training_type == "noise-aware":
        # for noise-aware training, we need to add the noise-aware noise as a feature for each input in inputs with value 0
        new_inputs = inputs.copy()
        for key in inputs.keys():
            new_key = "g" + key
            new_inputs[new_key] = inputs[key]
            
    else:
            new_inputs = inputs.copy()
        
    input_feats = list(new_inputs.items())
     
    problem = {
        'num_vars': len(new_inputs),
        'names': [f"x{i}" for i in range(len(new_inputs))],
        'bounds': [[val["range"][0], val["range"][1]] for key, val in input_feats]
    }
    param_values = saltelli.sample(problem, num_samples, calc_second_order=False)
    

    if training_type == "noise-aware":
        param_values[:, -len(inputs):] = 0

    if model_type == "CustomModel":
        from Training.CustomModel import CustomModel
        model = CustomModel(input_shape=len(inputs), loss_function="mean_squared_error", output_shape=1)
        # model = CustomModel.load_model(model_path, input_shape=(len(new_inputs), 2), loss_function="mean_squared_error", output_shape=1)
        model = keras.models.load_model(model_path)
    elif model_type == "expression":
        model = dataset_generator
    else:
        model = keras.models.load_model(model_path)

    param_values = np.reshape(param_values, (param_values.shape[0], len(new_inputs)))
    
    if model_type == "expression":
        Y = model.apply_equation(param_values).flatten()
    else:
        Y = model.predict(param_values).flatten()
    # # Run model
    Si = sobol.analyze(problem, Y,calc_second_order=False)
    weights = Si["ST"]
    
    return weights












