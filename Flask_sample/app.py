from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load("linear_regression.pkl")

# Create a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input from the form
            features = [float(x) for x in request.form.values()]
            feature_names = ['area', 'bedroomd', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
            #features = np.array(features).reshape(1, -1)
            features_df = pd.DataFrame([features], columns=feature_names)

            # Make prediction
            prediction = model.predict(features)

            # Return the prediction result to HTML
            return render_template('index.html', prediction=round(prediction[0], 2))
        except Exception as e:
            return render_template('index.html', errors=f"Error: {str(e)}")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
