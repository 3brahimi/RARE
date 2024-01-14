from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import keras


def estimate_weights(model_path, inputs, training_type="clean", num_samples=100, model_type="Linear"):
    """
    Estimate the weights of the input features.

    Parameters:
    model (object): path to the trained model.
    inputs (obj): The input data, which is a dictionary of the form {feature_name: {range: [min, max], type: data_type}}.
    Returns:
    ndarray: The estimated weights of the input features.
    """
    
    if training_type == "noise-aware":
        # for adversarial training, we need to add the adversarial noise as a feature for each input in inputs
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
    
    if model_type == "CustomModel":
        from Training.CustomModel import CustomModel
        model = CustomModel(input_shape=2, loss_function="mean_squared_error", output_shape=1)
        # model = CustomModel.load_model(model_path, input_shape=(len(new_inputs), 2), loss_function="mean_squared_error", output_shape=1)
        model = keras.models.load_model(model_path)
    else:
        model = keras.models.load_model(model_path)

    param_values = np.reshape(param_values, (param_values.shape[0], len(new_inputs)))
    
    
    Y = model.predict(param_values).flatten()
    # # Run model
    Si = sobol.analyze(problem, Y,calc_second_order=False)
    return Si["S1"]












