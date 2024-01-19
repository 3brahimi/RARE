
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils.training_utils import CustomLoss
import keras

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
        print("loss_function", self.loss_function)
        if self.loss_function == "custom_loss":

            metric = fit_args_copy["metric"]
            x_noisy = fit_args_copy["x_noisy"]
            len_input_features = fit_args_copy["len_input_features"]
            bl_ratio = fit_args_copy["bl_ratio"]
            gx_dist = fit_args_copy["nominator"]
            y_clean = fit_args_copy["y_clean"]
            del fit_args_copy["metric"]
            del fit_args_copy["x_noisy"]
            del fit_args_copy["len_input_features"]
            del fit_args_copy["bl_ratio"]
            del fit_args_copy["nominator"]
            del fit_args_copy["y_clean"]
            self.model.compile(optimizer='adam', loss=CustomLoss(model=self.model, metric=metric, y_clean=y_clean, x_noisy=x_noisy, len_input_features=len_input_features, bl_ratio=bl_ratio, nominator=gx_dist))
        else:
        # Compile the model
            self.model.compile(optimizer='adam', loss=self.loss_function)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=fit_args_copy["early_stopping"])
        fit_args['callbacks'] = [early_stopping]
        del fit_args_copy['early_stopping']
        fit_args_new = {}
        for key, value in fit_args_copy.items():
            if key == 'early_stopping':
                continue
            fit_args_new[key] = value
        history = self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), **fit_args_new)

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
        tf.keras.layers.Dense(32, input_shape=[self.shape_input], activation='relu'),
        # tf.keras.layers.Dense(32, activation='relu', input_shape=(self.shape_input,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(self.shape_output)
        ])
        return model
    def load_model(self, filepath):
        """
        Loads a saved model.

        Args:
            cls (BaseModel): The BaseModel class.
            filepath (str): The path to the saved model.
            loss_function (str): The loss function to be used for model training.

        Returns:
            LinearModel: An instance of the LinearModel class with the loaded model.
        """
        model = tf.keras.models.load_model(filepath)
        return model
    
    def save_model(self, path):
        """
        Saves the model to a file.

        Args:
            path (str): The path to save the model.

        """
        # with open(path, 'wb') as f:
        #     pickle.dump(self.model, f)
        self.model.save(f"{path}/model.pkl")
    
    def fit(self, x_train, y_train):
        """
        Fits the model to the provided training data.

        Args:
            x_train (numpy.ndarray): The training data inputs.
            y_train (numpy.ndarray): The training data outputs.

        """
        self.model.fit(x_train, y_train)

class CNNModel(BaseModel):
    def model_architecture(self):
        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', shape_input=(64, 64, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        return model


class RandomForestModel(BaseModel):
    def __init__(self, loss_function, n_estimators=100, max_depth=None):
        """
        Initialize a RandomForestModel object.

        Args:
            loss_function (str): The loss function to be used for training the model.
            n_estimators (int, optional): The number of trees in the random forest. Defaults to 100.
            max_depth (int, optional): The maximum depth of each tree in the random forest. Defaults to None.
        """
        super().__init__(loss_function)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def model_architecture(self):
        """
        Create the architecture of the random forest model.

        Returns:
            RandomForestRegressor: The random forest model.
        """
        model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)
        return model
    
    def compile_and_fit(self, xy_train, xy_valid, fit_args):
        """
        Compile and fit the random forest model.

        Args:
            xy_train (tuple): A tuple containing the training data (x_train, y_train).
            xy_valid (tuple): A tuple containing the validation data (x_valid, y_valid).
            fit_args (dict): A dictionary containing additional arguments for fitting the model.

        Returns:
            RandomForestRegressor: The fitted random forest model.
        """
        x_train = xy_train[0]
        y_train = xy_train[1]
        self.model.fit(x_train, y_train)
        return self.model
    


class LinearRegressionModel(BaseModel):
    def __init__(self, loss_function):
        super().__init__(loss_function)

    def model_architecture(self):
        model = LinearRegression()
        return model
    
    def compile_and_fit(self, xy_train, xy_valid, fit_args):
        x_train = xy_train[0]
        y_train = xy_train[1]
        self.model.fit(x_train, y_train)
        return self.model