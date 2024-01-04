from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import keras

def estimate_weights(model, inputs):
    """
    Estimate the weights of the input features.

    Parameters:
    model (object): The trained model.
    inputs (obj): The input data, which is a dictionary of the form {feature_name: {range: [min, max], type: data_type}}.
    Returns:
    ndarray: The estimated weights of the input features.
    """
    print("inputs", inputs)
    input_feats = list(inputs.items())
    problem = {
        'num_vars': len(inputs),
        'names': [f"x{i}" for i in range(len(inputs))],
        'bounds': [[val["range"][0], val["range"][1]] for key, val in input_feats]
    }
    
    model = keras.models.load_model(model)

    param_values = saltelli.sample(problem, 10000, calc_second_order=False)
    param_values = np.reshape(param_values, (param_values.shape[0], len(inputs), 1))
    Y = model.predict(param_values).flatten()
    # # Run model
    Si = sobol.analyze(problem, Y,calc_second_order=False)
    return Si["S1"]












