from Training.Models import CNNModel, LinearModel, LinearRegressionModel, RandomForestModel
# from Training.CustomModel import CustomModel

class ModelTrainer:
    def __init__(self):
        self.model_architectures = {
            "linear": LinearModel,
            "Linear": LinearModel,
            "cnn": CNNModel,
            "LR": LinearRegressionModel,
            "RF": RandomForestModel,
            # "CustomModel": CustomModel,
            # Add more architectures here...
        }

    def get_model(self, model_name, shape_input, loss_function, **kwargs):
        if model_name in self.model_architectures:            
            self.model = self.model_architectures[model_name](shape_input=shape_input, loss_function=loss_function, **kwargs)
            return self.model
        else:
            raise ValueError(f"Model architecture '{model_name}' not recognized.")
        
    def save_model(self, model, path):
        """
        call save_model function of the model
        """
        self.model.save_model(path)
