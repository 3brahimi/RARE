import numpy as np

class NoiseGenerator:
    def __init__(self, num_samples, num_noises, noise_type, variance, percentage=1, **kwargs):
        self.num_samples = num_samples
        self.num_noises = num_noises
        self.noise_type = noise_type
        self.variance = variance
        self.percentage = percentage
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        else:
            self.epsilon = 1

    def generate_noise(self, random_seed=0):
        noises = []
        # random_seed = np.random.randint(0, 1000) if random_seed == 0 else random_seed
        np.random.seed(random_seed)
        for _ in range(self.num_noises):
            if self.noise_type == 'normal':
                noise = np.random.normal(0, self.variance, self.num_samples) * self.percentage
                # noise = np.random.normal(0, 1, self.num_samples) * self.variance
            elif self.noise_type == 'uniform':
                noise = np.random.uniform(-self.variance, self.variance, self.num_samples) * self.percentage
            elif self.noise_type == 'laplace':
                noise = np.random.laplace(0, self.variance, self.num_samples) * self.percentage
                
                noise = noise / self.epsilon
            elif self.noise_type == 'laplace_dp':
                noise = np.random.laplace(0, self.variance, self.num_samples) * self.percentage
                noise = noise / self.epsilon
            elif self.noise_type == 'pareto':
                noise = np.random.pareto(self.variance, self.num_samples) * self.percentage
            else:
                raise ValueError('Invalid noise type. Choose "normal" or "uniform" or "laplace" or "laplace_dp"')
            noises.append(noise)
        return noises
