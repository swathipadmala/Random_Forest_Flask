import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
data = {
    'feature1': [2.5, 3.6, 1.8, 3.1, 3.0, 2.1, 1.5, 3.2, 2.2, 3.8],
    'feature2': [1.5, 2.1, 1.8, 2.2, 2.0, 1.9, 1.4, 2.1, 1.7, 2.5],
    'target': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[['feature1', 'feature2']]
y = df['target']

# Train the model
model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
model.fit(X, y)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        feature1 = float(data['feature1'])
        feature2 = float(data['feature2'])
        
        # Load the trained model
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Make prediction
        prediction = model.predict([[feature1, feature2]])[0]
        
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)