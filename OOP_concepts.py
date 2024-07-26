from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import csv
#from parent_class.py import matrix_multiplication
#from helpers.py import download_and_save_iris_dataset
# BaseModel Class

from abc import ABC, abstractmethod
import numpy as np
from time import time

class matrix_multiplication:

  row_col_time = 0  #class attributes
  vector_time = 0

  def __init__(self, row, col):
    self.row= row
    self.col= col
    self.A = np.random.rand(self.row,self.col)
    self.B = np.random.rand(self.row,self.col)
    self.C = np.zeros((self.row,self.col))


    

  @abstractmethod
  def vector_product():
    pass

  @abstractmethod
  def row_col_multiplication():
    pass

  @abstractmethod
  def timefor_RC_method():
    pass

  @abstractmethod
  def timeforvector():
    pass

class row_col(matrix_multiplication):

  def __init(self,row,col):
    super().__init__(row,col)


  def row_col_multiplication(self):
    s_time = time()
    for i in range(self.row):
      row_a = self.A[i]
      for j in range(self.row):
        col_b = self.B[:,j]

        entry = 0
        for k in range(self.row):
          entry += row_a[k]*col_b[k]
          self.C[i,j] = entry
    e_time = time()
    self.row_col_time=e_time-s_time


  def timefor_RC_method(self):
    return self.row_col_time




class vector_multiplication(matrix_multiplication):
  def __init__(self,row,col):
    super().__init__(row,col)
  def vector_product(self):
    s_time = time()
    C2 = np.dot(self.A,self.B)
    e_time = time()
    self.vector_time = e_time - s_time

  def time_forvector(self):
    return self.vector_time
  





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
    

class textfilehandling:
    def __init__(self,file_path):
        self.file_path = file_path
    
    def create_file(self):
        try:
            with open(self.file_path, 'w') as f:
                f.write('Hello, world!\n')
            print("File " + self.file_path + " created successfully.")
        except IOError:
            print("Error: could not create file " + self.file_path)

    def read_file(self):
        try:
            with open (self.file_path,'r') as f:
                contents = f.read()

            print("file ",self.file_path,"has been read successfuly ! ")
        except IOError:
            print("Error: could not read file !! ")

    def append_file(self,text):
        try:
            with open(self.file_path,'a') as f:
                f.write(text)
            print("Text appended to file " + self.file_path + " successfully.")
        except IOError:
            print("Error! could not load file : ")
    
    def rename_file(self,new_name):
        try:
            os.rename(self.file_path, new_name)
            print("File " + self.file_path + " renamed to " + new_name + " successfully.")

        except IOError:
            print("Error: could not rename file " + self.file_path)

    def delete_file(self):
        try:
            os.remove(self.file_path)
            print("File " + self.file_path + " deleted successfully.")

        except IOError:
            print("Error: could not delete file " + self.file_path)

    def display(self):
        try:
            file = open(self.file_path, "r") 
            print (file.read())

            print("file read !! now displaying data!",self.file_path)
            #print(data)
        except IOError:
            print("error occured while opening file !! ")

class Iterator:

    def __init__(self, start, end):
        self.a =start
        self.start= start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.a<=self.end:
            x = self.a
            self.a +=1
            return x
        else:
            raise StopIteration


class DataHandler:
    def __init__(self, file_path):
        self.data = pd.DataFrame(self._read_csv(file_path))
        # Assuming first row is header
        self.data.columns = self.data.iloc[0]
        self.data = self.data[1:].reset_index(drop=True)

    def _read_csv(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                yield row

    @staticmethod
    def display_columns(data):
        print("columns are : ", data.columns)

    def preprocess(self):
        print("Original data:")
        print(self.data.head())
        
        DataHandler.display_columns(self.data)

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
        #print("in best performance decorator init : ")
        self.function=func
    def __call__(self,*args, **kwargs):
        #print("in best performance decorator __Call__ : ")
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
    print("in best performance function : ")
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
    #text_obj=textfilehandling("practice.txt")
    #text_obj.display()

    iterator = Iterator(1, 5)
    for num in iterator:
        print(num)

    obj1 = row_col(100,100)
    obj1.row_col_multiplication()
    print("time for Row column multiplication method : ",obj1.timefor_RC_method())



    obj = vector_multiplication(100,100)
    obj.vector_product()
    print("time for vector multiplication : ",obj.time_forvector())

if __name__ == '__main__':
    #download_and_save_iris_dataset(file_path='iris.csv')
    main()