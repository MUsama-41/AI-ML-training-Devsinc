from flask import Flask, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

CSV_FILE = '/data/iris.csv'  # Path to the CSV file in the Docker container
#MODEL_FILE = '/data/model.pkl'

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        df = pd.read_csv(CSV_FILE)
        
        # Assume 'species' is the column to predict
        X = df.drop('species', axis=1)
        y = df['species']

        # Train Decision Tree model
        model = DecisionTreeClassifier()
        model.fit(X, y)

        # # Save model
        # with open(MODEL_FILE, 'wb') as model_file:
        #     pickle.dump(model, model_file)
        
        return jsonify({'status': 'Model trained and saved'}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
