# 🚀 Linear Regression API (FastAPI)

A production-ready FastAPI service for training and serving a Linear
Regression model using REST APIs.

------------------------------------------------------------------------

## 📌 Features

-   Train Linear Regression model via API\
-   Save and load model artifacts\
-   Predict outputs using trained model\
-   Input validation with Pydantic\
-   Modular architecture

------------------------------------------------------------------------

## 🧱 Project Structure

ml_api/ ├── main.py\
├── schemas.py\
├── service.py\
├── model.py\
├── models/\
└── .gitignore

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
pip install fastapi uvicorn scikit-learn numpy joblib
```

------------------------------------------------------------------------

## ▶ Run Server

``` bash
uvicorn main:app --reload
```

Open:

http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## 🔹 Train Model

POST /train

``` json
{
  "X": [[1],[2],[3],[4]],
  "y": [2,4,6,8]
}
```

------------------------------------------------------------------------

## 🔹 Predict

POST /predict

``` json
{
  "features": [5]
}
```

Response:

``` json
{
  "prediction": 10.0
}
```

------------------------------------------------------------------------

## 📦 Tech Stack

Python\
FastAPI\
Scikit-learn\
NumPy

------------------------------------------------------------------------

## 👤 Author

Abhishek SR

------------------------------------------------------------------------

## 📄 License

Apache License 2.0
