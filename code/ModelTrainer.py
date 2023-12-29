from Models import CNNModel, LinearModel, LinearRegressionModel, RandomForestModel

class ModelTrainer:
    def __init__(self):
        self.model_architectures = {
            "linear": LinearModel,
            "cnn": CNNModel,
            "LR": LinearRegressionModel,
            "RF": RandomForestModel,
            # Add more architectures here...
        }

    def get_model(self, model_name, loss_function):
        if model_name in self.model_architectures:
            return self.model_architectures[model_name](loss_function)
        else:
            raise ValueError(f"Model architecture '{model_name}' not recognized.")
