import numpy as np

class RandomDataLoader:
    def __init__(self, dataset, num_envs, seed=None):
        """
        Custom infinite data loader that returns random samples from the dataset.
        """
        self.dataset = dataset
        self.num_envs = num_envs
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def get_data(self):
        """
        Get a random data sample from the dataset.
        """
        random_index = self.rng.integers(0, len(self.dataset))
        return self.dataset[random_index]
