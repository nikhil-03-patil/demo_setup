# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")
# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from model import train_model

app = Flask(__name__)

# Load or train the model
if not os.path.exists('salary_model.pkl') or not os.path.exists('features.pkl'):
    print("Training model...")
    model, feature_names = train_model()
else:
    print("Loading existing model...")
    with open('salary_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        years_experience = float(request.form['years_experience'])
        education = request.form['education']
        role = request.form['role']
        city_tier = int(request.form['city_tier'])
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'years_experience': [years_experience],
            'education': [education],
            'role': [role],
            'city_tier': [city_tier]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template('result.html', 
                              years_experience=years_experience,
                              education=education,
                              role=role,
                              city_tier=city_tier,
                              salary=f"${prediction:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)