from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from Training.ModelTrainer import ModelTrainer
import keras
from utils.training_utils import CustomLoss

def get_custom_loss(model, metric, y_clean, x_noisy, len_input_features, bl_ratio, nominator):
    def custom_loss(y_true, y_pred):
        return CustomLoss(model, metric, y_clean, x_noisy, len_input_features, bl_ratio, nominator)(y_true, y_pred)
    return custom_loss

def estimate_weights(model_path, inputs, dataset_generator, training_type="clean", num_samples=100, model_type="Linear", loss_function="mean_squared_error", **kwargs):
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
        if loss_function == "custom_loss":
            # read the custom loss parameters from kwargs
            metric = kwargs["metric"]
            x_noisy = kwargs["x_noisy"]
            len_input_features = kwargs["len_input_features"]
            bl_ratio = kwargs["bl_ratio"]
            nominator = kwargs["nominator"]
            y_clean = kwargs["y_clean"]
            trainer = ModelTrainer().get_model(model_type, shape_input=len(input_feats)
                                               , loss_function=loss_function)
            model = trainer.model
            customloss = CustomLoss(model=model, metric=metric, y_clean=y_clean, x_noisy=x_noisy, len_input_features=len(input_feats), bl_ratio=bl_ratio, nominator=nominator)
            model.compile(optimizer='adam', loss=customloss)
            model.load_weights(f"{model_path}/model_weights.h5")
            # model = keras.models.load_model(model_path, compile=False)
            
            # customloss = CustomLoss(model=model, metric=metric, y_clean=y_clean, x_noisy=x_noisy, len_input_features=len_input_features, bl_ratio=bl_ratio, nominator=nominator)
            # model.compile(optimizer='adam', loss=customloss)
        else:
            
            trainer = ModelTrainer().get_model(model_type, shape_input=len(input_feats)
                                               , loss_function=loss_function)
            model = trainer.model
            model.compile(optimizer='adam', loss=loss_function)
            model.load_weights(f"{model_path}/model_weights.h5")
            # model = keras.models.load_model(model_path)

    param_values = np.reshape(param_values, (param_values.shape[0], len(new_inputs)))
    
    if model_type == "expression":
        Y = model.apply_equation(param_values).flatten()
    else:
        Y = model.predict(param_values).flatten()
    # # Run model
    Si = sobol.analyze(problem, Y,calc_second_order=False)
    weights = Si["ST"]
    
    return weights












