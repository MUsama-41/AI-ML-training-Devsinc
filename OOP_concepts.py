from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
#from helpers.py import download_and_save_iris_dataset
# BaseModel Class
class BaseModel(ABC):
    def __init__(self):
        self.model = None
    
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        pass

# LinearRegressionModel Class
class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        return mse

# SVMModel Class
class SVMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SVC()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy

# RandomForestModel Class
class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy

# DecisionTreeModel Class
class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy

# ModelManager Class
class ModelManager:
    def __init__(self):
        self.models = []
    
    def add_model(self, model):
        self.models.append(model)
    
    def train_all(self, X, y):
        for model in self.models:
            model.train(X, y)
    
    def evaluate_all(self, X, y):
        results = {}
        for model in self.models:
            model_name = type(model).__name__
            results[model_name] = model.evaluate(X, y)
        return results

class DataHandler:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    @staticmethod
    def display_columns(self):
        print("columns are : ", self.data.columns)


    def preprocess(self):
        print("Original data:")
        print(self.data.head())
        
        DataHandler.display_columns(self)

        # Example preprocessing
        self.data.dropna(inplace=True)
        
        # Convert categorical target to numeric values
        if self.data['target'].dtype == 'object':
            self.data['target'] = self.data['target'].astype('category').cat.codes
        
        print("Processed data:")
        print(self.data.head())
    
    def get_train_test_split(self, test_size=0.2):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        return train_test_split(X, y, test_size=test_size, random_state=42)
    

class decorator(object):
    #staticmethod
    def __init__(self,func):
        self.function=func
    def __call__(self,*args, **kwargs):
        return self.function(*args, **kwargs)
    

# Define a dictionary to simulate switch-case behavior
def get_model(model_name):
    models = {
        '1': LinearRegressionModel,
        '2': SVMModel,
        '3': RandomForestModel,
        '4': DecisionTreeModel
    }
    return models.get(model_name, None)()

@decorator
def best_performance(resutls):
    vall = max(resutls.values())
    name = max(resutls, key=resutls.get)
    #print("temp : ",temp)
    return vall,name

def main():
    data_handler = DataHandler('iris.csv')
    data_handler.preprocess()  # this is static method which is directly called by class name    
    X_train, X_test, y_train, y_test = data_handler.get_train_test_split()

    # Instantiate the model manager
    model_manager = ModelManager()

    # Prompt the user for model selection
    print("Select models to add:")
    print("1: Linear Regression")
    print("2: SVM")
    print("3: Random Forest")
    print("4: Decision Tree")
    print("Enter the numbers of the models you want to add, separated by commas:")

    user_input = input().strip().split(',')
    print(user_input)
    for model_choice in user_input:
        model_class = get_model(model_choice.strip())
        #print("model class : ",model_class)
        if model_class:
            model_manager.add_model(model_class)
        else:
            print(f"Invalid choice: {model_choice.strip()}")

    # Train all models
    model_manager.train_all(X_train, y_train)

    # Evaluate all models
    results = model_manager.evaluate_all(X_test, y_test)
    #print("results : ",results)

    val,name = best_performance(results)
    print("best performed model is : ",name,"with accuracy : ", val)

    #for model_name, performance in results.items():
        #print(f"{model_name} performance: {performance}")


if __name__ == '__main__':
    #download_and_save_iris_dataset(file_path='iris.csv')
    main()