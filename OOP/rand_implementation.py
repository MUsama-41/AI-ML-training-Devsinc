import random
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy import stats
# Parent Class
class DataCleaner(ABC):
    def __init__(self, file_path=None):
        self.df = pd.DataFrame()
        if file_path:
            self.load_data(file_path)
    
    @abstractmethod
    def load_data(self, file_path):
        """Load data from a file."""
        pass
    
    @abstractmethod
    def handle_missing_values(self):
        """Handle missing values in the data."""
        pass
    
    @abstractmethod
    def remove_duplicates(self):
        """Remove duplicate rows from the data."""
        pass
    
    @abstractmethod
    def handle_outliers(self, columns):
        """Handle outliers in specified columns."""
        pass

# Child Classes

# Read CSV File
class CSVDataLoader(DataCleaner):
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)

# Handle Missing Values
class MissingValueHandler(DataCleaner):
    def handle_missing_values(self):
        self.df = self.df.dropna()  # Drop rows with any missing values
        self.df = self.df.dropna(axis=1)  # Drop columns with any missing values

# Remove Duplicates
class DuplicateRemover(DataCleaner):
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()

# Handle Outliers
class OutlierHandler(DataCleaner):
    def handle_outliers(self, columns):
        for column in columns:
            # Apply z-score method
            z_scores = stats.zscore(self.df[column])
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3)
            self.df = self.df[filtered_entries]




class RandomGenerator:
    def __init__(self, seed=None):
        # Initialize the RandomGenerator with an optional seed.
        self.seed = seed
        random.seed(seed)
    
    def random_float(self):
        # Return a random float in the range [0.0, 1.0).
        return random.random()

    def uniform(self, a, b):
        # Return a random float in the range [a, b].
        return random.uniform(a, b)

    def randint(self, a, b):
        # Return a random integer in the range [a, b] (inclusive).
        return random.randint(a, b)

    def choice(self, seq):
        # Return a random element from the non-empty sequence `seq`.
        if not seq:
            raise ValueError("Cannot choose from an empty sequence")
        return random.choice(seq)

    def sample(self, population, k):
        # Return a list of `k` unique elements from the `population`.
        if k > len(population):
            raise ValueError("Sample larger than population")
        return random.sample(population, k)

    def choices(self, population, k, weights=None):
        # Return a list of `k` elements from the `population` with optional weights.
        return random.choices(population, weights=weights, k=k)

    def shuffle(self, x):
        # Shuffle the sequence `x` in place.
        random.shuffle(x)
    
    def set_seed(self, seed):
        # Set the seed for reproducibility.
        self.seed = seed
        random.seed(seed)

# Example usage:
if __name__ == "__main__":
    rng = RandomGenerator(seed=42)

    print("Random float between 0 and 1:", rng.random_float())
    print("Random float between 1 and 10:", rng.uniform(1, 10))
    print("Random integer between 1 and 10:", rng.randint(1, 10))
    
    items = ['apple', 'banana', 'cherry']
    print("Random choice from list:", rng.choice(items))
    
    print("Random sample of 2 elements:", rng.sample(items, 2))
    
    print("Random choices with replacement:", rng.choices(items, k=2))
    
    numbers = [1, 2, 3, 4, 5]
    rng.shuffle(numbers)
    print("Shuffled list:", numbers)

    # Load data
    loader = CSVDataLoader(file_path='iris.csv')
    loader.load_data('data.csv')
    
    # Handle missing values
    missing_handler = MissingValueHandler()
    missing_handler.df = loader.df
    missing_handler.handle_missing_values()
    
    # Remove duplicates
    duplicate_remover = DuplicateRemover()
    duplicate_remover.df = missing_handler.df
    duplicate_remover.remove_duplicates()
    
    # Handle outliers in specific columns
    outlier_handler = OutlierHandler()
    outlier_handler.df = duplicate_remover.df
    outlier_handler.handle_outliers(columns=['column1', 'column2'])
    
    # Normalize specific columns
    normalizer = DataNormalizer()
    normalizer.df = outlier_handler.df
    normalizer.normalize_data(columns=['column1', 'column2'])
    
    # Access cleaned DataFrame
    cleaned_df = normalizer.df
    print(cleaned_df.head())
