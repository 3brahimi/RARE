import numpy as np

class NoiseGenerator:
    def __init__(self, num_samples, num_noises, noise_type, percentage, **kwargs):
        self.num_samples = num_samples
        self.num_noises = num_noises
        self.noise_type = noise_type
        self.percentage = percentage
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        else:
            self.epsilon = 0.5

    def generate_noise(self, random_seed=0):
        noises = []
        # random_seed = np.random.randint(0, 1000) if random_seed == 0 else random_seed
        np.random.seed(random_seed)
        for _ in range(self.num_noises):
            if self.noise_type == 'normal':
                noise = np.random.normal(0, self.percentage, self.num_samples)
                # noise = np.random.normal(0, 1, self.num_samples) * self.percentage
            elif self.noise_type == 'uniform':
                noise = np.random.uniform(-self.percentage, self.percentage, self.num_samples)
            elif self.noise_type == 'laplace':
                noise = np.random.laplace(0, self.percentage, self.num_samples)
                noise = noise / self.epsilon
            elif self.noise_type == 'laplace_dp':
                noise = np.random.laplace(0, self.percentage, self.num_samples)
                noise = noise / self.epsilon
            else:
                raise ValueError('Invalid noise type. Choose "normal" or "uniform" or "laplace" or "laplace_dp"')
            noises.append(noise)
        return noises
