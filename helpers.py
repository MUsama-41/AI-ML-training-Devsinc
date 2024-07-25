import pandas as pd

def download_and_save_iris_dataset(file_path='iris.csv'):
    """
    Downloads the Iris dataset from UCI Machine Learning Repository and saves it as a CSV file.

    Args:
        file_path (str): The path where the CSV file will be saved.
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    
    # Download and load the dataset
    data = pd.read_csv(url, header=None, names=columns)
    
    # Save to CSV
    data.to_csv(file_path, index=False)
    print(f"Dataset saved to {file_path}")



# Example usage:
if __name__ == "__main__":
    download_and_save_iris_dataset()
