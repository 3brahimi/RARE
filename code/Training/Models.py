import pickle
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils.training_utils import CustomLoss, CustomMetric, CustomCallback, LossCallback, CustomLossDP
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

def load_my_model():
    model = keras.models.load_model('model_path', compile=False)
    return model

class BaseModel(tf.keras.Model):
    """
    Base class for models.

    Args:
        loss_function (str): The loss function to be used for model training.

    Attributes:
        loss_function (str): The loss function used for model training.
        model (tf.keras.Model): The compiled model.

    Methods:
        model_architecture(): Abstract method to be overridden by each subclass.
        compile_and_fit(xy_train, xy_valid, fit_args): Compiles and fits the model.
        save_model(path): Saves the model to a file.
        load_model(filepath, loss_function): Loads a saved model.

    """
    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def model_architecture(self):
        """
        Abstract method to be overridden by each subclass.
        This method should define the architecture of the model.

        Returns:
            tf.keras.Model: The model architecture.

        """
        pass
    def compile_and_fit(self, xy_train, xy_valid, fit_args):
        """
        Compiles and fits the model using the provided training and validation data.
        Overrides the method from BaseModel.

        Args:
            xy_train (tuple): A tuple containing the training data (x_train, y_train).
            xy_valid (tuple): A tuple containing the validation data (x_valid, y_valid).
            fit_args (dict): Additional arguments for model fitting.

        Returns:
            tuple: A tuple containing the trained model and the training history.

        """
        self.model = self.model_architecture()
        x_train = xy_train[0]
        y_train = xy_train[1]
        x_valid = xy_valid[0]
        y_valid = xy_valid[1]
        fit_args_copy = fit_args.copy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # use differential privace optimizer
        # import tensorflow_privacy
        # optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
        #     l2_norm_clip=0.7,
        #     noise_multiplier=2.1,
        #     num_microbatches=1,
        #     learning_rate=0.001,
        # )
        call_back = None
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=fit_args_copy["early_stopping"])
        if self.loss_function == "custom_loss":

            metric = fit_args_copy["metric"]
            x_noisy_valid = fit_args_copy["x_noisy_valid"]
            x_noisy_train = fit_args_copy["x_noisy_train"]
            len_input_features = fit_args_copy["len_input_features"]
            bl_ratio = fit_args_copy["bl_ratio"]
            gx_dist = fit_args_copy["nominator"]
            y_clean_valid = fit_args_copy["y_clean_valid"]
            y_clean_train = fit_args_copy["y_clean_train"]
            
            del fit_args_copy["metric"]
            del fit_args_copy["x_noisy_train"]
            del fit_args_copy["x_noisy_valid"]
            del fit_args_copy["len_input_features"]
            del fit_args_copy["bl_ratio"]
            del fit_args_copy["nominator"]
            del fit_args_copy["y_clean_valid"]
            del fit_args_copy["y_clean_train"]
            loss_inst = CustomLoss(model=self.model, metric=metric, 
                                               y_clean=y_clean_train, x_noisy=x_noisy_train,
                                               len_input_features=len_input_features, 
                                               bl_ratio=bl_ratio)
            
            self.model.compile(optimizer=optimizer, 
                               # Metric is calculated on the validation data
                               metrics=["mse", CustomMetric(model=self.model, y_clean=y_clean_valid, x_noisy=x_noisy_valid,
                                                                        len_input_features=len_input_features, 
                                                                         nominator=gx_dist, name="custom_metric")],
                            #    loss is calculated on the training data
                               loss=loss_inst)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=fit_args_copy["early_stopping"])
            # call_back = CustomCallback(loss_inst)
            call_back = LossCallback(x_noisy=x_noisy_train, model=self.model, y_clean=y_clean_train, bl_ratio=bl_ratio)
        else:
        # Compile the model            
            self.model.compile(optimizer=optimizer, loss=self.loss_function)
        # activation_logger = ActivationLogger(self.model, validation_data=(x_valid, y_valid))
        # fit_args_copy["callbacks"] = [early_stopping, activation_logger]
        fit_args_copy["callbacks"] = [early_stopping]
        # # if self.loss_function == "custom_loss":
        # #     fit_args_copy["callbacks"].append(call_back)
        # del fit_args_copy['early_stopping']
        fit_args_new = {}
        for key, value in fit_args_copy.items():
            if key == 'early_stopping':
                continue
            fit_args_new[key] = value
            
        # set the batch size to be the entire dataset
        history = self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=2, **fit_args_new)

        return self.model, history
    
    def save_model(self, path):
        """
        Saves the model to a file.

        Args:
            path (str): The path to save the model.

        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load_model(cls, filepath, loss_function):
        """
        Loads a saved model.

        Args:
            cls (BaseModel): The BaseModel class.
            filepath (str): The path to the saved model.
            loss_function (str): The loss function to be used for model training.

        Returns:
            BaseModel: An instance of the BaseModel class with the loaded model.

        """
        model = tf.keras.models.load_model(filepath)
        instance = cls(loss_function)
        instance.model = model
        return instance

class ActivationLogger(keras.callbacks.Callback):
    """
    
    """
    def __init__(self, model, validation_data):
        self.model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        self.x_val, self.y_val = validation_data
    def on_epoch_end(self, epoch, logs=None):
        if self.x_val is None:
            raise RuntimeError('Requires validation_data.')
        if epoch % 10 == 0:  # Plot every 10 epochs
            for layer_num, layer in enumerate(self.model.layers):
                activation_model = Model(inputs=self.model.input, outputs=layer.output)
                activations = activation_model.predict(self.x_val)
                plt.figure()
                plt.figure(figsize=(10, 6))
                plt.title(f'Layer {layer_num + 1} Activations at Epoch {epoch + 1}')
                plt.hist(activations.flatten(), bins=100)

                if os.path.exists(f"./activations/") == False:
                    os.mkdir(f"./activations/")
                plt.savefig(f"./activations/activation_{layer_num}_epoch_{epoch}.png")

class LinearModel(BaseModel):
    def __init__(self, shape_input, loss_function, shape_output=1):
        """
        Initialize a LinearModel object.

        Args:
            shape_input (int): The input shape of the model.
            loss_function (str): The loss function to be used for training the model.
            
        """
        super().__init__(loss_function)
        self.shape_input = shape_input
        self.shape_output = shape_output
        self.model = self.model_architecture()
        
    def model_architecture(self):
        """
        Creates and returns a Keras Sequential model with a single Dense layer.

        Returns:
            model (tf.keras.models.Sequential): The created model.
        """
        model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=[self.shape_input], activation='tanh', kernel_initializer='he_normal', kernel_regularizer="l2"),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(self.shape_output)
        ])
        return model
    def load_model(self, model_obj, filepath):
        """
        Loads a saved model.

        Args:
            filepath (str): The path to the saved model.
            model_obj (tf.keras.Model): The model object to load the weights into.
        Returns:
            LinearModel: An instance of the LinearModel class with the loaded model.
        """
        model_obj.load_weights(f"{filepath}/model_weights.h5")
        return model_obj
    
    def save_model(self, model, path):
        """
        Saves the model to a file.

        Args:
            path (str): The path to save the model.

        """
        print("saving weights")
        self.model.save_weights(f"{path}/model_weights.h5")
    
    def compile_and_fit(self, xy_train, xy_valid, fit_args):
            """
            Compiles and fits the model using the provided training and validation data.
    
            Args:
                xy_train (tuple): A tuple containing the training data (x_train, y_train).
                xy_valid (tuple): A tuple containing the validation data (x_valid, y_valid).
                fit_args (dict): Additional arguments for model fitting.
    
            Returns:
                tuple: A tuple containing the trained model and the training history.
    
            """
            self.model = self.model_architecture()
            x_train, y_train = xy_train
            x_valid, y_valid = xy_valid
            fit_args_copy = fit_args.copy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=fit_args_copy["early_stopping"])
    
            if self.loss_function == "msep":
                self._compile_custom_loss(fit_args_copy, optimizer)
            else:
                self.model.compile(optimizer=optimizer, loss=self.loss_function)
    
            fit_args_copy["callbacks"] = [early_stopping]
            fit_args_new = {key: value for key, value in fit_args_copy.items() if key != 'early_stopping'}
    
            history = self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=2, **fit_args_new)
    
            return self.model, history
    
    def _compile_custom_loss(self, fit_args_copy, optimizer):
            metric = fit_args_copy["metric"]
            x_noisy_valid = fit_args_copy["x_noisy_valid"]
            x_noisy_train = fit_args_copy["x_noisy_train"]
            len_input_features = fit_args_copy["len_input_features"]
            bl_ratio = fit_args_copy["bl_ratio"]
            gx_dist = fit_args_copy["nominator"]
            y_clean_valid = fit_args_copy["y_clean_valid"]
            y_clean_train = fit_args_copy["y_clean_train"]
    
            del fit_args_copy["metric"]
            del fit_args_copy["x_noisy_train"]
            del fit_args_copy["x_noisy_valid"]
            del fit_args_copy["len_input_features"]
            del fit_args_copy["bl_ratio"]
            del fit_args_copy["nominator"]
            del fit_args_copy["y_clean_valid"]
            del fit_args_copy["y_clean_train"]
    
            loss_inst = CustomLoss(model=self.model, metric=metric, 
                                   y_clean=y_clean_train, x_noisy=x_noisy_train,
                                   len_input_features=len_input_features, 
                                   bl_ratio=bl_ratio)
    
            self.model.compile(optimizer=optimizer, 
                               metrics=["mse", CustomMetric(model=self.model, y_clean=y_clean_valid, x_noisy=x_noisy_valid,
                                                            len_input_features=len_input_features, 
                                                            nominator=gx_dist, name="R")],
                               loss=loss_inst)
    
class CNNModel(BaseModel):
    def __init__(self, shape_input, loss_function, shape_output=1):
        """
        Initialize a CNNModel object.

        Args:
            shape_input (int): The input shape of the model.
            loss_function (str): The loss function to be used for training the model.
            
        """
        super().__init__(loss_function)
        self.shape_input = shape_input
        self.shape_output = shape_output
        self.model = self.model_architecture()
        
    def model_architecture(self):
        """
        Creates and returns a Keras Sequential model with a single Dense layer.

        Returns:
            model (tf.keras.models.Sequential): The created model.
        """
        model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((self.shape_input, 1), input_shape=[self.shape_input]),

        tf.keras.layers.Conv1D(64, 3, activation='tanh', input_shape=[self.shape_input, 1], kernel_initializer='he_normal', kernel_regularizer="l2", padding='same'),
        # tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='tanh', padding='same'),
        # tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(self.shape_output)
        ])
        return model
    def load_model(self, model_obj, filepath):
        """
        Loads a saved model.

        Args:
            filepath (str): The path to the saved model.
            model_obj (tf.keras.Model): The model object to load the weights into.
        Returns:
            LinearModel: An instance of the LinearModel class with the loaded model.
        """
        # model = tf.keras.models.load_model(filepath)
        model_obj.load_weights(f"{filepath}/model_weights.h5")
        return model_obj
    
    def save_model(self, model, path):
        """
        Saves the model to a file.

        Args:
            path (str): The path to save the model.

        """
        print("saving weights")
        self.model.save_weights(f"{path}/model_weights.h5")
    
    def compile_and_fit(self, xy_train, xy_valid, fit_args):
        """
        Compiles and fits the model using the provided training and validation data.

        Args:
            xy_train (tuple): A tuple containing  the training data (x_train, y_train).
            xy_valid (tuple): A tuple containing the validation data (x_valid, y_valid).
            fit_args (dict): Additional arguments for model fitting.
        """
        self.model = self.model_architecture()
        x_train = xy_train[0]
        y_train = xy_train[1]
        x_valid = xy_valid[0]
        y_valid = xy_valid[1]
        fit_args_copy = fit_args.copy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=fit_args_copy["early_stopping"])
        
        # Compile the model            
        self.model.compile(optimizer=optimizer, loss=self.loss_function)

        fit_args_copy["callbacks"] = [early_stopping]
        fit_args_new = {}
        for key, value in fit_args_copy.items():
            if key == 'early_stopping':
                continue
            fit_args_new[key] = value
            
        # set the batch size to be the entire dataset
        history = self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=2, **fit_args_new)

        return self.model, history


def loss_functions(loss_string):
    if loss_string == "mse":
        return tf.keras.losses.MeanSquaredError()
    elif loss_string == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    else:
        raise ValueError(f"Loss function '{loss_string}' not recognized.")
class History:
    def __init__(self, loss, val_loss):
        self.history = {
            "loss": [loss],
            "val_loss": [val_loss]
        }
class RandomForestModel(BaseModel):
    def __init__(self, loss_function, shape_input=1, output_shape=1):
        super().__init__(loss_function)
        self.model = self.model_architecture()
        self.loss_function = loss_functions(loss_function)

    def model_architecture(self):
        model = RandomForestRegressor()
        return model

    def compile_and_fit(self, xy_train, xy_valid, fit_args):
        x_train = xy_train[0]
        y_train = xy_train[1]
        self.model.fit(x_train, y_train)

        y_pred_valid = self.model.predict(xy_valid[0])
        y_pred_train = self.model.predict(xy_train[0])
        history = History(self.loss_function(y_pred_train, xy_train[1]).numpy(), self.loss_function(y_pred_valid, xy_valid[1]).numpy())

        return self.model, history

    def save_model(self, model, path):
        with open(f"{path}/model.pkl", 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath, model_obj):
        model = pickle.load(open(f'{filepath}/model.pkl', 'rb'))
        return model

    def evaluate(self, x, y):
        return self.loss_function(self.model.predict(x), y).numpy()

    def predict(self, x):
        y_pred= self.model.predict(x)
        # reshape it to be (#num_samples, 1) instead of (#num_samples,)
        y_pred = y_pred.reshape(-1, 1)
        return y_pred    


class LinearRegressionModel(BaseModel):
    def __init__(self, loss_function, shape_input=1, output_shape=1):
        
        
        super().__init__(loss_function)
        self.model = self.model_architecture()
        self.loss_function = loss_functions(loss_function)
    def model_architecture(self):
        model = LinearRegression()
        return model
    
    def compile_and_fit(self, xy_train, xy_valid, fit_args):
        x_train = xy_train[0]
        y_train = xy_train[1]
        self.model.fit(x_train, y_train)
        
        y_pred_valid = self.model.predict(xy_valid[0])
        y_pred_train = self.model.predict(xy_train[0])
        history = History(self.loss_function(y_pred_train, xy_train[1]).numpy(), self.loss_function(y_pred_valid, xy_valid[1]).numpy())
        
        return self.model, history
    
    def save_model(self, model, path):
        with open(f"{path}/model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath, model_obj):
        
        model = pickle.load(open(f'{filepath}/model.pkl', 'rb'))
        
        return model
    def evaluate(self, x, y):
        return self.loss_function(self.model.predict(x), y).numpy()
    
    def predict(self, x):
        y_pred= self.model.predict(x)
        # reshape it to be (#num_samples, 1) instead of (#num_samples,)
        y_pred = y_pred.reshape(-1, 1)
        return y_pred


