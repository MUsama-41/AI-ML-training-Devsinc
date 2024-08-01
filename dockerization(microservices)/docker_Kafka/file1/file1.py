from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

CSV_FILE = '/data/iris.csv'  # Path to the CSV file in the Docker container

@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        df = pd.read_csv(CSV_FILE)
        data_json = df.to_dict(orient='records')
        return jsonify({'status': 'data loaded'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
