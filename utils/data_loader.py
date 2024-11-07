
import numpy as np
from torch.utils.data import Subset
class InfiniteDataLoader:
    def __init__(self, dataset, num_envs):
        """
        Custom infinite data loader that distributes dataset across environments.
        """
        self.dataset = dataset
        self.num_envs = num_envs
        
        # Randomly split dataset into mutually exclusive subsets
        self.env_datasets = self._split_dataset()
        self.current_indices = [0] * self.num_envs
    
    def _split_dataset(self):
        """
        Split dataset into mutually exclusive subsets for each environment.
        """
        # Shuffle the entire dataset
        indices = np.random.permutation(len(self.dataset))
        
        # Divide indices into roughly equal chunks
        split_indices = np.array_split(indices, self.num_envs)
        
        # Create subset for each environment
        return [Subset(self.dataset, subset) for subset in split_indices]
    
    def get_data(self):
        """
        Get a data sample for a specific environment.
        """
        env_dataset = self.env_datasets[0]
        current_idx = self.current_indices[0]
        
        # Wrap around if we've reached the end of the subset
        if current_idx >= len(env_dataset):
            current_idx = 0
        
        data = env_dataset[current_idx]
        self.current_indices[0] = current_idx + 1
        
        return data
