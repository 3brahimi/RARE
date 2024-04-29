from math import ceil
import random
from matplotlib import pyplot as plt
from SALib.sample import saltelli
import numpy as np
from sklearn.model_selection import train_test_split
from utils.dists import L2_distance
from utils.helpers import get_data, get_data_from_dict, noise_aware_data, get_adversarial_data, noisy_data
import sympy as sp
import torch
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
        # if len(symbols) != len(self.input_feats):
        #     raise ValueError(f"Number of input features ({len(self.input_feats)}) does not match the number of symbols in the equation ({len(symbols)})")
        return eq, len(symbols)

    def generate_dataset(self, seed=42):
        """
        Generates the clean input and output data.

        Returns:
        - x_clean: The clean input data.
        - y_clean: The clean output data.
        """
        random.seed(seed)
        np.random.seed(seed)
        x_clean = np.zeros((self.num_inputs, self.num_samples))
        # Generate clean input data
        input_feats = list(self.input_feats.items())
        for i in range(self.num_inputs):
            key, val = input_feats[i]
            # step = (val["range"][1] - val["range"][0]) / (self.num_samples * 100)
            step = (val["range"][1] - val["range"][0]) / (self.num_samples)
            
            # values = np.arange(val["range"][0], val["range"][1], step)
            x = np.linspace(val["range"][0], val["range"][1], self.num_samples)
            # x = random.sample(list(values), self.num_samples)
            # x = np.random.choice(np.arange(val["range"][0], val["range"][1]), size=self.num_samples, replace=False)
            # x = np.sort(x)
            x_clean = x if i == 0 else np.vstack((x_clean, x))
        
        # uncomment this to use SALib to generate the input data
        # problem = {
        #     'num_vars': len(self.input_feats),
        #     'names': [f"x{i}" for i in range(len(self.input_feats))],
        #     'bounds': [[val["range"][0], val["range"][1]] for key, val in input_feats]
        # }
        
        # x_clean = saltelli.sample(problem, self.num_samples, calc_second_order=False).T
        
        # self.num_samples = x_clean.shape[1]
        # print("x_clean", x_clean.shape, "num_samples", self.num_samples)
        # self.noise_generator.num_samples = self.num_samples
        
        # # uncomment these lines to use the meshgrid to generate the input data
        # x_clean = np.meshgrid(*x_clean)
        # x_clean = np.array(x_clean)
        # x_clean = x_clean.reshape(self.num_inputs, -1)
        # self.num_samples = x_clean.shape[1]
        # self.noise_generator.num_samples = self.num_samples
        # # Generate clean output data according to the specified equation
        if self.num_inputs == 1:
            # x_clean = x_clean.T
            # reshape it to be (num_samples, 1)
            x_clean = x_clean.reshape(-1, 1)
            y_clean = self.equation(x_clean).flatten()
        else:
            eq_inputs = [x_clean[i] for i in range(self.num_inputs)]
            y_clean = self.equation(*eq_inputs)
            
        if x_clean.shape[0] != y_clean.shape[0]:
            x_clean = x_clean.T
            
        return x_clean, y_clean

    def apply_equation(self, x):
        num_inputs = 0
        if self.num_inputs != x.shape[1]:
            num_inputs = x.shape[1]
        else:
            num_inputs = self.num_inputs    
        if num_inputs == 1:
            x = x.T
            y = self.equation(x)
        else:
            eq_inputs = [x[:,i] for i in range(num_inputs)]
            y = self.equation(*eq_inputs)
        return y
    def modulate_clean(self, x, y, target_feat_idx=[0], random_seeds=[0]):
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
            noises = self.noise_generator.generate_noise(random_seed=random_seeds[0])
            x_noisy[:,:,0] = np.array([x[:,0] + noise for noise in noises])
            
        else:    
            for i in range(self.num_inputs):
                if i in target_feat_idx:
                    print("modulating noise for feature", i)
                    # Generate noise
                    noises = self.noise_generator.generate_noise(random_seed=random_seeds[i])
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
    
    def split_old(self, config, metric_instance=None):
        """
        Splits the dataset into training, validation, and testing sets.

        Parameters:
        - config: A dictionary containing the configuration for the data splits.
        - metric_instance: An instance of the RobustnessMetric class.

        Returns:
        - (x_train, y_train): The training data.
        - (x_valid, y_valid): The validation data.
        - (x_test, y_test): The testing data.
        """
        target_feat_idx = config['noisy_input_feats']
        # Generate clean dataset
        x_clean, y_clean = self.generate_dataset()

        # x_clean = np.log(x_clean)
        # y_clean = np.log(y_clean)
        # shape of clean data is (num_samples, num_inputs/features)

        y_clean = y_clean.ravel()
        # generate a set of random seeds for the noise generator for each input feature, make sure that these are the same each time we call the function, so that we can compare the results
        random_seeds = np.linspace(0, 1000, self.num_inputs, dtype=int)

        # Modulate the clean dataset
        x_noisy, y_noisy = self.modulate_clean(x_clean, y_clean, target_feat_idx, random_seeds)
        
        training_type = config["training_type"]

        
        # Extract the noisy signal with the maximum distance from the clean signal --> Gx
        if self.num_inputs == 1:
            gx = self.extract_g(x_noisy, x_clean)
            gx = metric_instance.extract_g(x_noisy, x_clean) if metric_instance else gx
        else:
            gx = np.zeros((self.num_samples, self.num_inputs))
            gx_y = np.zeros((self.num_samples,))
            for i in range(self.num_inputs):
                gx_temp = self.extract_g(x_noisy[:, :, i], x_clean[:, i])
                gx_temp = metric_instance.extract_g(x_hat=x_noisy[:, :, i], x = x_clean[:, i]) if metric_instance else gx_temp
                # get the distance between gx_temp and x_clean[:, i] as a point-wise distance
                gx_temp_distances = np.array([L2_distance(x_clean[:, i], gx_temp, type="pointwise")]).flatten()
                # gx[:, i] = gx_temp - x_clean[:, i]
                gx[:, i] = gx_temp
                # sign = np.sign(gx_temp - x_clean[:, i])
                
                # gx[:, i] = gx_temp_distances * sign
            
            # plot the clean data
            for i in range(x_clean.shape[1]):
                plt.clf()
                plt.plot(np.linspace(0, x_clean.shape[0], x_clean.shape[0]), x_clean[:,i], label=f"Feature {i}")
                plt.plot(np.linspace(0,x_clean.shape[0],x_clean.shape[0]), gx[:,i], label=f"G(Feature {i})")
                plt.legend()
                plt.savefig(f"./clean_data_feature_{i}.png")

            plt.clf()
            plt.plot(np.linspace(0, 1, x_clean.shape[0]), x_clean[:, 0], label="$h$")
            plt.plot(np.linspace(0, 1, x_clean.shape[0]), gx[:,0], label="$G(h)$")
            plt.plot(np.linspace(0, 1, x_clean.shape[0]), x_clean[:, 1], label="$w$")
            
            plt.plot(np.linspace(0, 1, x_clean.shape[0]), gx[:,1], label="$G(w)$")
            plt.legend()
            plt.savefig("./test.png")
            plt.clf()
            plt.plot(np.linspace(0, 1, y_clean.shape[0]), y_clean, label='Clean data')
            plt.savefig(f"./clean_data.png")
            # uncomment these lines to use the meshgrid to generate the input data
            x_clean, y_clean = self.meshgrid_x_y(x_clean)

            def plots():                
                # fig = plt.figure()
                
                # ax = fig.add_subplot(111, projection='3d')
                # # x_clean_0_mesh = x_clean[:,0].reshape(100, 100)
                # # x_clean_1_mesh = x_clean[:,1].reshape(100, 100)
                # # # take only the first sample of y_clean to plot the surface
                # # y_clean_mesh = y_clean.reshape(100, 100)
                # print(x_clean.shape, y_clean.shape, x_clean[0,:,:].shape, x_clean[1,:,:].shape)
                # ax.plot_surface(x_clean[0,:,:], x_clean[1,:,:], y_clean, alpha=0.1, edgecolor='k')
                
                # # the first value of y_clean
                # y_clean_first  = [y_clean[0,0]] * x_clean.shape[1]
                
                # # the last value of x_clean_1
                # x_clean_0_last = [x_clean[0, -1, -1]] * x_clean.shape[1]
                
                # # the last value of x_clean_1
                # x_clean_1_last = [x_clean[1, -1, -1]] * x_clean.shape[1]
                # # ax.plot(gx[:,0], np.linspace(1,5,100), y_clean_first, color='r', label='$G(h)$')
                # # ax.plot(np.linspace(1,5,100), gx[:,0], y_clean_first, color='r', label='$G(h)$')
                # # ax.plot(np.linspace(1,5,100), gx[:,1], y_clean_first, color='b', label='$G(\omega)$', alpha=0.5)
                # x_axis_values = np.linspace(1,5,x_clean.shape[1])

                # ax.plot(gx[:,1]-x_axis_values, x_axis_values, y_clean_first, color='b', label='$G(\omega)$', alpha=0.5)

                # ax.plot(x_axis_values, gx[:,0]-x_axis_values, y_clean_first, color='r', label='$G(h)$')
                # plt.legend()
                
                # ax.set_xlabel('$h$')
                # ax.set_ylabel('$\omega$')
                # ax.set_zlabel('$U$')
                # ax.set_title(r'$U = \frac{h}{2\pi} \cdot \omega$')
                # plt.savefig("./clean_data1.png")
                
                # # plot features of x_clean and y_clean as the output in 3-d plot
                fig = plt.figure()
                
                ax = fig.add_subplot(111, projection='3d')
                # x_clean_0_mesh = x_clean[:,0].reshape(100, 100)
                # x_clean_1_mesh = x_clean[:,1].reshape(100, 100)
                # # take only the first sample of y_clean to plot the surface
                # y_clean_mesh = y_clean.reshape(100, 100)
                # ax.plot_surface(x_clean[0,:,:], x_clean[1,:,:], y_clean, alpha=0.1, edgecolor='k')
                # for 4d, we can use the 3d plot to plot the 4th dimension as the color of the surface
                ax.plot_surface(x_clean[0,:,:], x_clean[1,:,:], y_clean, alpha=0.1, edgecolor='k')
                
                # # the first value of y_clean
                # y_clean_first  = [y_clean[0,0]] * x_clean.shape[1]
                
                # # the last value of x_clean_1
                # x_clean_0_last = [x_clean[0, -1, -1]] * x_clean.shape[1]
                
                # # the last value of x_clean_1
                # x_clean_1_last = [x_clean[1, -1, -1]] * x_clean.shape[1]
                # # ax.plot(gx[:,0], np.linspace(1,5,100), y_clean_first, color='r', label='$G(h)$')
                # ax.plot(x_axis_values, gx[:,0]-x_axis_values, y_clean_first, color='r', label='$G(h)$')
                
                plt.legend()
                
                ax.set_xlabel('$h$')
                ax.set_ylabel('$\omega$')
                ax.set_zlabel('$U$')
                ax.set_title(r'$U = \frac{h}{2\pi} \cdot \omega$')
                plt.savefig("./clean_data4d.png")
                # the clean data after the meshgrid
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(x_clean[0,:,:], x_clean[1,:,:], y_clean)
                # set the labels
                ax.set_xlabel('$q$')
                ax.set_ylabel('$C$')
                ax.set_zlabel('$V$')
                ax.set_title(r'$V = q / C$')
                plt.savefig("./clean_data_after_Downsampling.png")       
        gx_temp = gx
        # gy is the clean y repeated num_noises times
        gx, gx_y = self.meshgrid_gx_gy(gx, y_clean)
        
        # flatten the clean and noisy data
        original_x_clean_shape = x_clean.shape
        x_clean = x_clean.reshape(x_clean.shape[0], -1).T
        y_clean = y_clean.ravel()
        original_num_samples = self.num_samples

        self.num_samples = x_clean.shape[0]
        self.noise_generator.num_samples = self.num_samples
                
        x_noisy, y_noisy = self.meshgrid_noisy(x_noisy, y_clean)
        
        # downsample the data according to the sampling rate
        # max_sampling_rate = x_clean.shape[0]/(original_num_samples*2)
        sampling_rate = ceil(x_clean.shape[0]/((original_num_samples*2)*x_clean.shape[1]))

        print(f"Downsampling the data to {sampling_rate} of the original size")

        x_clean = x_clean[::sampling_rate]
        y_clean = y_clean[::sampling_rate]
        x_noisy = x_noisy[:, ::sampling_rate]
        y_noisy = y_noisy[:, ::sampling_rate]
        gx = gx[::sampling_rate]
        gx_y = gx_y[::sampling_rate]
        # let's plot after the downsampling to see if the data is still the same
        # but first, let's reshape the data to the meshgrid format according to the sampling rate
        # meshgrid_size = int(np.sqrt(x_clean.shape[0]))
        # meshgrid_size = 32
        # print(f"Reshaping the data to {meshgrid_size}x{meshgrid_size} meshgrid")
        # # x_clean = x_clean.reshape(self.num_inputs, meshgrid_size, meshgrid_size)
        # change the original shape to align with the downsampling
 
        # x_clean = x_clean.T
        # x_clean = x_clean.reshape(self.num_inputs, meshgrid_size, meshgrid_size)
        # y_clean = y_clean.reshape(meshgrid_size, meshgrid_size)
        # plots()
        # # x_noisy = x_noisy.reshape(self.num_noises, meshgrid_size, meshgrid_size, self.num_inputs)
        # y_noisy = y_noisy.reshape(self.num_noises, meshgrid_size, meshgrid_size)
        # gx = gx_temp
        # # gx_y = gx_y.reshape(meshgrid_size, meshgrid_size)

        # now shuffle the data
        self.num_samples = x_clean.shape[0]
        self.noise_generator.num_samples = self.num_samples
        
        # x_clean = self.add_multiplications(x_clean)
        # x_noisy_new = np.zeros((self.num_noises, self.num_samples, x_clean.shape[1]))
        # for i in range(x_noisy.shape[0]):
        #     x_noisy_new[i] = self.add_multiplications(x_noisy[i])
        # x_noisy = x_noisy_new
        # self.num_inputs = x_clean.shape[1]
        
        x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid = None, None, None, None, None, None, None, None
        if "clean" in training_type:
            
            indices = np.arange(self.num_samples)
            # split the clean data into training, validation, and testing sets
            indices_train, indices_temp,  x_train, x_temp, y_train, y_temp = train_test_split(indices, x_clean, y_clean, test_size=0.2, random_state=42, shuffle=True)
   
            indices_valid, indices_test, x_valid, x_test, y_valid, y_test = train_test_split(indices_temp, x_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
            
        elif training_type == "noise-aware":
            x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid = noise_aware_data(config, x_clean, y_clean, gx, gx_y)
        
        elif training_type == "adversarial":
            x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid = get_adversarial_data(config, x_clean, y_clean, gx, gx_y)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (x_noisy, y_noisy), (x_clean, y_clean), (gx, gx_y), (indices_train, indices_valid)

    def split(self, config, metric_instance=None):
        """
        Splits the dataset into training, validation, and testing sets.
        Adding noise on the meshgrid data, not the original

        Parameters:
        - config: A dictionary containing the configuration for the data splits.
        - metric_instance: An instance of the RobustnessMetric class.

        Returns:
        - (x_train, y_train): The training data.
        - (x_valid, y_valid): The validation data.
        - (x_test, y_test): The testing data.
        """
        target_feat_idx = config['noisy_input_feats']
        # Generate clean dataset
        x_clean, y_clean = self.generate_dataset()
        x_clean, y_clean = self.meshgrid_x_y(x_clean)
        
        if self.num_inputs > 1:
            x_clean = x_clean.reshape(x_clean.shape[0], -1).T
        
        y_clean = y_clean.ravel()
        
        original_num_samples = self.num_samples
        self.num_samples = x_clean.shape[0]
        self.noise_generator.num_samples = self.num_samples
        
        # generate a set of random seeds for the noise generator for each input feature, make sure that these are the same each time we call the function, so that we can compare the results
        random_seeds = np.linspace(0, 1000, self.num_inputs, dtype=int)

        # Modulate the clean dataset
        x_noisy, y_noisy = self.modulate_clean(x_clean, y_clean, target_feat_idx, random_seeds)

        training_type = config["training_type"]
        
        # Extract the noisy signal with the maximum distance from the clean signal --> Gx
        if self.num_inputs == 1:
            gx = np.zeros((self.num_samples, self.num_inputs))
            gx = self.extract_g(x_noisy, x_clean)
            gx = metric_instance.extract_g(x_hat=x_noisy[:,:,0], x=x_clean[:,0]) if metric_instance else gx
            gx = gx.reshape(-1, 1)
        else:
            gx = np.zeros((self.num_samples, self.num_inputs))
            gx_y = np.zeros((self.num_samples,))
            for i in range(self.num_inputs):
                gx_temp = self.extract_g(x_noisy[:, :, i], x_clean[:, i])
                gx_temp = metric_instance.extract_g(x_hat=x_noisy[:, :, i], x = x_clean[:, i]) if metric_instance else gx_temp
                # get the distance between gx_temp and x_clean[:, i] as a point-wise distance
                gx_temp_distances = np.array([L2_distance(x_clean[:, i], gx_temp, type="pointwise")]).flatten()
                # gx[:, i] = gx_temp - x_clean[:, i]
                gx[:, i] = gx_temp

            
        gx_temp = gx
        gx_y = y_clean
        
        if self.num_inputs == 1:
            sampling_rate = 1
        else:
            sampling_rate = ceil(x_clean.shape[0]/((original_num_samples*2)*x_clean.shape[1]))
        print(f"Downsampling the data to {sampling_rate} of the original size")

        x_clean = x_clean[::sampling_rate]
        y_clean = y_clean[::sampling_rate]
        x_noisy = x_noisy[:, ::sampling_rate]
        y_noisy = y_noisy[:, ::sampling_rate]
        gx = gx[::sampling_rate]
        gx_y = gx_y[::sampling_rate]

        self.num_samples = x_clean.shape[0]
        self.noise_generator.num_samples = self.num_samples
        
        x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid = None, None, None, None, None, None, None, None
        if "clean" in training_type:
            
            indices = np.arange(self.num_samples)
            # split the clean data into training, validation, and testing sets
            indices_train, indices_temp,  x_train, x_temp, y_train, y_temp = train_test_split(indices, x_clean, y_clean, test_size=0.2, random_state=42, shuffle=True)
   
            indices_valid, indices_test, x_valid, x_test, y_valid, y_test = train_test_split(indices_temp, x_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
            
        elif training_type == "noise-aware":
            x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid = noise_aware_data(config, x_clean, y_clean, gx, gx_y)
        
        elif training_type == "adversarial":
            x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid = get_adversarial_data(config, x_clean, y_clean, gx, gx_y)
        
        elif training_type == "noisy_all":
            # append all x_noisy to x_clean as new rows
            x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid = noisy_data(x_clean, y_clean, x_noisy, y_noisy)
        elif training_type == "noisy_g":
            # append gx to x_clean as new rows
            x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid = noisy_data(x_clean, y_clean, gx, gx_y)
                
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (x_noisy, y_noisy), (x_clean, y_clean), (gx, gx_y), (indices_train, indices_valid)

    def add_multiplications(self, x_clean):
        new_features = []
        # print(x_clean.shape, y_clean.shape)
        # for the features of the clean input, add new features as the multiplication of the original features together: if inputs are x1,x2,x3, then add x1*x2, x1*x3, x2*x3
        for i in range(self.num_inputs):
            for j in range(i+1, self.num_inputs):
                new_feature = x_clean[:, i] * x_clean[:, j]
                new_features.append(new_feature)
        new_features = np.column_stack(new_features)
        x_clean = np.hstack((x_clean, new_features))
        return x_clean
    
    # meshgrid function for the gx and gy and return the reshaped/flatten gx and gy
    def meshgrid_gx_gy(self, gx, gx_y):
        gx = np.meshgrid(*gx.T)
        gx = np.array(gx)
        gx = gx.reshape(gx.shape[0], -1).T
        gx_y = gx_y.ravel()
        return gx,gx_y
    
    
    def meshgrid_x_y(self, x_clean):
        """
        meshgrid function for the x_clean and y_clean
        """
        if self.num_inputs > 1:
            x_clean = np.meshgrid(*x_clean.T)
            x_clean = np.array(x_clean)
            eq_inputs = [x_clean[i] for i in range(self.num_inputs)]

        else:
            eq_inputs = [x_clean]
            # x_clean = x_clean.reshape(x_clean.shape[0], -1).T
            
        y_clean = self.equation(*eq_inputs)
        
        return x_clean,y_clean

    def meshgrid_noisy(self, x_noisy, y_clean):
        # recreate the y_noisy with the same shape as y_clean
        y_noisy = np.repeat(y_clean[np.newaxis, :], self.num_noises, axis=0)
        y_noisy = y_noisy.reshape(y_noisy.shape[0], -1)

        x_noisy_new = np.zeros((self.num_noises, self.num_samples, self.num_inputs))

        # meshgrid of the noisy data
        for idx in range(x_noisy.shape[0]):
            x_noisy_sample = x_noisy[idx, :, :]
            x_noisy_sample = np.meshgrid(*x_noisy_sample.T)
            x_noisy_sample = np.array(x_noisy_sample)
            x_noisy_sample = x_noisy_sample.reshape(x_noisy_sample.shape[0], -1)
            x_noisy_new[idx, :, :] = x_noisy_sample.T
        return x_noisy_new, y_noisy
    
    def split_multi_outputs(self, config, equations):
        """
        split function but for multiple outputs
        
        """
        target_feat_idx = config['noisy_input_feats']
        # Generate clean dataset
        y_clean = np.zeros((self.num_samples, len(equations)))
        for i, equation_str in enumerate(equations):
            x_clean, y_temp = self.generate_dataset(equation_str)
            # shape of clean data is (num_samples, num_inputs/features)

            y_temp = y_temp.ravel()
            y_clean[:, i] = y_temp
        
        x_train_clean, x_temp_clean, y_train_clean, y_temp_clean = train_test_split(x_clean, y_clean, test_size=0.1, random_state=42)
        x_train, y_train = get_data(config['training'], x_train_clean, y_train_clean, None, None)
        x_valid, y_valid = get_data(config['validation'], x_temp_clean, y_temp_clean, None, None)
        x_test, y_test = get_data(config['testing'], x_temp_clean, y_temp_clean, None, None)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
            
    def set_equation(self, equation_str):
        self.equation, self.num_inputs = self.make_equation(equation_str)
        return self.equation, self.num_inputs
    