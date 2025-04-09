from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))

# Load brand and location encoding maps
with open('brand_encoding.pkl', 'rb') as f:
    brand_encoding = pickle.load(f)

with open('location_encoding.pkl', 'rb') as f:
    location_encoding = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Basic numeric inputs
    km_driven = int(request.form['Kilometers_Driven'])
    transmission = int(request.form['Transmission'])  # 1 for Manual, 0 for Auto
    owner_type = int(request.form['Owner_Type'])
    mileage = float(request.form['Mileage(kmpl)'])
    engine = float(request.form['Engine(cc)'])
    power = float(request.form['Power(bhp)'])
    seats = int(request.form['Seats'])
    age = int(request.form['Age'])

    # Target Encoded features
    brand = request.form['Brand']
    location = request.form['Location']
    brand_encoded = brand_encoding.get(brand, 0)  # Fallback to 0 if brand not found
    location_encoded = location_encoding.get(location, 0)

    # Fuel type (only Petrol column due to drop_first=True)
    fuel = request.form['Fuel_Type']
    fuel_encoded = 1 if fuel == 'Petrol' else 0

    # Final input vector
    input_data = [
        km_driven,
        transmission,
        owner_type,
        mileage,
        engine,
        power,
        seats,
        age,
        brand_encoded,
        location_encoded,
        fuel_encoded
    ]

    final_input = np.array([input_data])

    prediction = model.predict(final_input)[0]
    output = round(float(prediction), 2)

    return render_template('index.html', prediction_text=f"Estimated Car Price: ₹ {output} lakhs")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT if available
    app.run(host="0.0.0.0", port=port)
