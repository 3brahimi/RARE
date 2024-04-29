import random
import numpy as np
from DatasetGenerator import DatasetGenerator
from sklearn.model_selection import train_test_split
from utils.helpers import get_data

# inherit from DatasetGenerator to create a dataset generator for systems with multiple outputs, taking in multiple equations
class MultivarDatasetGenerator(DatasetGenerator):
    def __init__(self, equations_str, noise_generator, input_feats, num_samples=100):
        self.num_samples = num_samples
        self.input_feats = input_feats
        self.noise_generator = noise_generator
        self.num_noises = noise_generator.num_noises
        self.equations, self.num_inputs_list = self.make_equations(equations_str)
        

    def make_equations(self, equations_str):
        eqs, inputs_list = [], []
        for equation_str in equations_str:
            eq, inputs = self.make_equation(equation_str)
            eqs.append(eq)
            inputs_list.append(inputs)
        return eqs, inputs_list        
    
    # for generating the datasets, we need to generate a clean x according to how many input features there are, and then we use each equtaion to generate its own output using the same clean x
    def generate_dataset(self):
        # find the maximum number of inputs from self.num_inputs_list, then create a large x_clean with that number of inputs
        max_num_inputs = max(self.num_inputs_list)
        self.num_inputs = max_num_inputs

       # x_clean = np.zeros((max_num_inputs, self.num_samples))
        input_feats = list(self.input_feats.items())
        for i in range(max_num_inputs):
            key, val = input_feats[i]
            step = (val["range"][1] - val["range"][0]) / self.num_samples
            values = np.arange(val["range"][0], val["range"][1], step)
            
            x = random.sample(list(values), self.num_samples)
            x = np.sort(x)

            x_clean = x if i == 0 else np.vstack((x_clean, x))
        
        # Generate clean output data according to the specified equation
        y_clean = np.zeros((self.num_samples, len(self.equations)))
        for i, eq in enumerate(self.equations):
            eq_inputs = [x_clean[j] for j in range(self.num_inputs_list[i])]
            y_clean[:, i] = eq(*eq_inputs)
        
        if x_clean.shape[0] != y_clean.shape[0]:
            x_clean = x_clean.T
        return x_clean, y_clean

    def split(self, config):
        """
        split function but for multiple outputs
        
        """
        x_clean, y_clean = self.generate_dataset()

        x_train_clean, x_temp_clean, y_train_clean, y_temp_clean = train_test_split(x_clean, y_clean, test_size=0.1, random_state=42)
        x_train, y_train = get_data(config['training'], x_train_clean, y_train_clean, None, None)
        x_valid, y_valid = get_data(config['validation'], x_temp_clean, y_temp_clean, None, None)
        x_test, y_test = get_data(config['testing'], x_temp_clean, y_temp_clean, None, None)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)