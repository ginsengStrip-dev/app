from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

# Load the saved model and column order
model = joblib.load("titanic_model.pkl")
model_columns = joblib.load("model_columns.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Pclass: int = Form(...),
    Age: float = Form(...),
    SibSp: int = Form(...),
    Parch: int = Form(...),
    Fare: float = Form(...),
    Sex: str = Form(...),
    Embarked: str = Form(...)
):
    # Prepare input dictionary
    input_dict = {
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Sex_female': 1 if Sex == 'female' else 0,
        'Sex_male': 1 if Sex == 'male' else 0,
        'Embarked_C': 1 if Embarked == 'C' else 0,
        'Embarked_Q': 1 if Embarked == 'Q' else 0,
        'Embarked_S': 1 if Embarked == 'S' else 0
    }

    input_df = pd.DataFrame([input_dict])
    
    # Ensure same column order
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_df)[0]
    result = "Survived" if prediction == 1 else "Did not survive"
    
    return templates.TemplateResponse("result.html", {"request": request, "result": result})
