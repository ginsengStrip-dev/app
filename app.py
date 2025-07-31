from flask import Flask, render_template, request
import joblib

import pandas as pd

app = Flask(__name__)

# Load model and column names
model = joblib.load("titanic_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Retrieve form values
            Pclass = int(request.form["Pclass"])
            Age = float(request.form["Age"])
            SibSp = int(request.form["SibSp"])
            Parch = int(request.form["Parch"])
            Fare = float(request.form["Fare"])
            Sex = request.form["Sex"]
            Embarked = request.form["Embarked"]

            # One-hot encoding
            input_dict = {
                "Pclass": Pclass,
                "Age": Age,
                "SibSp": SibSp,
                "Parch": Parch,
                "Fare": Fare,
                f"Sex_{Sex}": 1,
                f"Embarked_{Embarked}": 1
            }

            # Build full feature vector with 0s
            input_data = pd.DataFrame([input_dict], columns=model_columns).fillna(0)

            # Prediction
            prediction = model.predict(input_data)[0]
            result = "Survived" if prediction == 1 else "Did not survive"
            return render_template("index.html", result=result)

        except Exception as e:
            return render_template("index.html", result="Error: " + str(e))

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
