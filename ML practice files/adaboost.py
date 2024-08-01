import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

 
class load_data():
   
    def __init__(self):
        print("in load data constructor : ")
        self.path = 'iris.csv'
        self.df = pd.read_csv(self.path)

    def split_data(self):
        self.x = self.df.drop('target', axis =1)
        self.y = self.df['target']

    def encoding(self):
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)

    def split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y_encoded, test_size = 0.2, random_state = 42)


class AdaBoost(load_data):
    def __AdaBoost__(self):
        super().__init__()

    def creat_classifier(self):
        # Create the AdaBoost classifier with a decision tree as the base estimator
        self.base_estimator = DecisionTreeClassifier(max_depth=1)  # Weak learner
        self.ada_classifier = AdaBoostClassifier(base_estimator=self.base_estimator, n_estimators=50, random_state=42)

    def fit_ada(self):
        self.ada_classifier.fit(self.X_train, self.y_train)

    def prediction(self):
        self.y_pred = self.ada_classifier.predict(self.X_test)


    def accuracy(self):
       self.accuracy = accuracy_score(self.y_test, self.y_pred)
       print(f"AdaBoost Accuracy: {self.accuracy:.2f}")
    
    def call_at_once(obj):
        obj.split_data()
        obj.encoding()
        obj.split_train_test()
        obj.creat_classifier()
        obj.fit_ada()
        obj.prediction()
        obj.accuracy()

class KMeans(load_data):
    





if __name__ == "__main__":
    obj = AdaBoost()
    obj.call_at_once()
    
        
        


        
