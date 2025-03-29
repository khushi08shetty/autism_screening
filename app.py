from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('autism_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [int(request.form[f"A{i}"]) for i in range(1, 11)]
        age = int(request.form['age'])
        
        # Create numpy array
        features = np.array([data + [age]]).reshape(1, -1)

        # Make Prediction
        prediction = model.predict(features)[0]

        # Result Message
        result_text = "You are exhibiting signs of having autism, please seek advice of a professional :)" if prediction == 1 else "You do not exhibit signs of having autism."

        return render_template('index.html', prediction=result_text)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
