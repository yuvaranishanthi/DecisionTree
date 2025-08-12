from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load("model/decision_tree_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
features = joblib.load("model/features.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            input_data = {}
            for feature in features:
                value = request.form.get(feature)
                if feature in label_encoders:
                    value = label_encoders[feature].transform([value])[0]
                else:
                    value = float(value)
                input_data[feature] = value

            df = pd.DataFrame([input_data])
            pred = model.predict(df)[0]
            prediction = "✅ Will Subscribe" if pred == 1 else "❌ Will Not Subscribe"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", features=features, encoders=label_encoders, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
