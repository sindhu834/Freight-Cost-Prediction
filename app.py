from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        distance = float(request.form['distance'])
        weight = float(request.form['weight'])
        fuel = float(request.form['fuel'])

        input_data = pd.DataFrame({
            'Distance': [distance],
            'Weight': [weight],
            'Fuel_Price': [fuel]
        })

        prediction = model.predict(input_data)

        return render_template(
            'index.html',
            prediction_text=f"💰 Predicted Freight Cost: ₹ {round(prediction[0],2)}"
        )

    except Exception as e:
        return render_template('index.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)