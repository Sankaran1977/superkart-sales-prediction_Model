
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model
model = joblib.load("superkart_model.joblib")

app = Flask(__name__)

# Health check endpoint
@app.route("/")
def home():
    return jsonify(
        {
            "message": "SuperKart Sales Prediction API Running"
        }
    )

# Single Prediction Endpoint
@app.route("/v1/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)

    return jsonify(
        {
            "Predicted_Sales": float(prediction[0])
        }
    )

# Batch Prediction Endpoint
@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():

    csv_file = request.files["file"]

    batch_df = pd.read_csv(csv_file)

    predictions = model.predict(batch_df)

    result = {
        str(i): float(value)
        for i, value in enumerate(predictions)
    }

    return jsonify(result)

# Run Flask
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860
    )
