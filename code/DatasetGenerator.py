import random
import numpy as np
from sklearn.model_selection import train_test_split
from helpers import get_data
import sympy as sp
class DatasetGenerator:
    """
    A class for generating datasets with clean and noisy data.

    Attributes:
    - equation: The equation used to generate the clean output data.
    - noise_generator: The noise generator object used to generate noise.
    - num_samples: The number of samples in the dataset.

    Methods:
    - generate_dataset(): Generates the clean input and output data.
    - modulate_clean(x, y): Modulates the clean data with noise.
    - extract_g(x_noisy, x_clean): Extracts the noisy signal with the maximum distance from the clean signal.
    - split(config): Splits the dataset into training, validation, and testing sets.
    """
    def __init__(self, equation_str, noise_generator, input_feats, num_samples=100):
        
        self.num_samples = num_samples
        self.input_feats = input_feats
        self.equation, self.num_inputs = self.make_equation(equation_str)
        self.noise_generator = noise_generator
        self.num_noises = noise_generator.num_noises

    def make_equation(self, equation_str):
                eq_expr = sp.sympify(equation_str)
           
                symbols = sorted(eq_expr.free_symbols, key=lambda x: x.name)
           
                eq = sp.lambdify([symbol.name for symbol in symbols], eq_expr, 'numpy')
                if len(symbols) != len(self.input_feats):
                    raise ValueError(f"Number of input features ({len(self.input_feats)}) does not match the number of symbols in the equation ({len(symbols)})")
                return eq, len(symbols)
    def generate_dataset(self):
        """
        Generates the clean input and output data.

        Returns:
        - x_clean: The clean input data.
        - y_clean: The clean output data.
        """
        x_clean = np.zeros((self.num_inputs, self.num_samples))
        # Generate clean input data
        input_feats = list(self.input_feats.items())
        for i in range(self.num_inputs):
            key, val = input_feats[i]
            step = (val["range"][1] - val["range"][0]) / self.num_samples
            values = np.arange(val["range"][0], val["range"][1], step)
            # x = np.linspace(-3, 3, self.num_samples)
            x = random.sample(list(values), self.num_samples)
            # x = np.random.choice(np.arange(val["range"][0], val["range"][1]), size=self.num_samples, replace=False)
            x = np.sort(x)
            x_clean = x if i == 0 else np.vstack((x_clean, x))

        # Generate clean output data according to the specified equation
        if self.num_inputs == 1:
            x_clean = x_clean.T
            y_clean = self.equation(x_clean)
        else:
            eq_inputs = [x_clean[i] for i in range(self.num_inputs)]
            y_clean = self.equation(*eq_inputs)
            
        if x_clean.shape[0] != y_clean.shape[0]:
            x_clean = x_clean.T
            
        return x_clean, y_clean

    def modulate_clean(self, x, y, target_feat_idx=[0]):
        """
        Modulates the clean data with noise.

        Parameters:
        - x: The clean input data.
        - y: The clean output data.
        - target_feat_idx: The indices of the features to be modulated.
        Returns:
        - x_noisy: The modulated input data with noise.
        - y_noisy: The clean output data repeated num_noises times.
        """
        # initialize x_noisy with the clean input data
        x_noisy = np.zeros((self.num_noises, self.num_samples, self.num_inputs))
        if self.num_inputs == 1:
            noises = self.noise_generator.generate_noise()
            x_noisy = np.array([x + noise for noise in noises])
        else:    
            for i in range(self.num_inputs):
                if i in target_feat_idx:
                    # Generate noise
                    noises = self.noise_generator.generate_noise()
                    # Add noise to the clean input data
                    x_noisy[:, :, i] = np.array([x[:, i] + noise for noise in noises])
                else:
                    x_noisy[:, :, i] = x[:, i]
                    
        # y_noisy is the clean y repeated num_noises times
        y_noisy = np.repeat(y[np.newaxis, :], self.num_noises, axis=0)

        return x_noisy, y_noisy

    def extract_g(self, x_noisy, x_clean):
        """
        Extracts the noisy signal with the maximum distance from the clean signal.

        Parameters:
        - x_noisy: The modulated input data with noise.
        - x_clean: The clean input data.

        Returns:
        - The noisy signal with the maximum distance from the clean signal.
        """
        distances = np.array([np.linalg.norm(x_noisy[i] - x_clean) for i in range(self.num_noises)])

        max_idx = np.argmax(distances)

        # Return the noisy signal with the maximum distance from the clean signal
        return x_noisy[max_idx]

    def extract_bounds(self, x_noisy):
        """
        Extracts the max and min bounds of the noisy set.

        Parameters:
        - x_noisy: The modulated input data with noise.

        Returns:
        - Two signals with the maximum and minimum bounds of the noisy set.
        """
        max_bound = np.max(x_noisy, axis=0)
        min_bound = np.min(x_noisy, axis=0)
        # Return the two bounds
        return max_bound, min_bound
    
    def split(self, config):
        """
        Splits the dataset into training, validation, and testing sets.

        Parameters:
        - config: A dictionary containing the configuration for the data splits.

        Returns:
        - (x_train, y_train): The training data.
        - (x_valid, y_valid): The validation data.
        - (x_test, y_test): The testing data.
        """
        target_feat_idx = config['noisy_input_feats']
        # Generate clean dataset
        x_clean, y_clean = self.generate_dataset()
        # shape of clean data is (num_samples, num_inputs/features)

        y_clean = y_clean.ravel()

        # Modulate the clean dataset
        x_noisy, y_noisy = self.modulate_clean(x_clean, y_clean, target_feat_idx)
        
        # TODO: fix the following for the training part

        # # Extract the noisy signal with the maximum distance from the clean signal --> Gx
        # gx = self.extract_g(x_noisy, x_clean)
        # gx_y = y_clean
        # gx, gx_y = None, None
        # # Split the data into training and temporary sets
        # x_train, x_temp, y_train, y_temp = train_test_split(x_clean, y_clean, test_size=0.2, random_state=42)
        # # Split the temporary set into validation and testing sets
        # x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

        x_train_clean, x_temp_clean, y_train_clean, y_temp_clean = train_test_split(x_clean, y_clean, test_size=0.1, random_state=42)
        # # Split the gx data into training, validation, and testing sets
        # gx_train, gx_temp, gy_train, gy_temp = train_test_split(gx, gx_y, test_size=0.2, random_state=42)
        # gx_valid, gx_test, gy_valid, gy_test = train_test_split(gx_temp, gy_temp, test_size=0.5, random_state=42)

        # x_train, y_train = get_data(config['training'], x_train_clean, y_train_clean, gx_train, gy_train)
        x_train, y_train = get_data(config['training'], x_train_clean, y_train_clean, None, None)

        # x_valid, y_valid = get_data(config['validation'], x_temp_clean, y_temp_clean, gx_valid, gy_valid)
        x_valid, y_valid = get_data(config['validation'], x_temp_clean, y_temp_clean, None, None)

        # x_test, y_test = get_data(config['testing'], x_temp_clean, y_temp_clean, gx_test, gy_test)
        x_test, y_test = get_data(config['testing'], x_temp_clean, y_temp_clean, None, None)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
