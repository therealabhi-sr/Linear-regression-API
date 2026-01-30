
# 🚀 Custom Linear Regression API (FastAPI)

A production-ready FastAPI service that trains and serves a **self-implemented Linear Regression model from scratch** using Gradient Descent (no scikit-learn).  
This project demonstrates real Machine Learning engineering fundamentals: data ingestion → training → model persistence → inference API.

---

## 📌 Key Highlights

- Linear Regression implemented from scratch using NumPy
- Batch Gradient Descent optimization
- Feature Scaling for stable convergence
- REST API for training and prediction
- Model saved and loaded from disk
- Clean modular architecture

---

## 🧱 Project Structure

linear-regression-api/
├── main.py                  # FastAPI app
├── schemas.py               # Request / Response schemas
├── service.py               # Business logic
├── model.py                 # Model loader & saver
├── custom_linear_regression.py  # Your ML algorithm
├── models/
│   └── linear_regression.pkl
└── README.md

---

## ⚙️ Installation

Create virtual environment:

python -m venv venv  
venv\Scripts\activate

Install dependencies:

pip install fastapi uvicorn numpy joblib

---

## ▶ Run Server

uvicorn main:app --reload

Swagger UI:

http://127.0.0.1:8000/docs

---

## 🧠 Machine Learning Theory

Model Equation:

y = XW + b

Loss Function:

MSE = (1/n) * Σ( y_pred - y_true )²

Gradient Updates:

W = W - α * dW  
b = b - α * db

---

## 🔹 Train Model

POST /train

Request:

{
  "X": [
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6],
    [4,5,6,7]
  ],
  "y": [40,54,68,82]
}

Response:

{
  "status": "trained"
}

---

## 🔹 Predict

POST /predict

Request:

{
  "features": [2,3,4,5]
}

Response:

{
  "prediction": 54
}

---

## 🧪 Example Rule Used

y = 2*x1 + 3*x2 + 4*x3 + 5*x4

---

## 🏗 System Architecture

Client  
 ↓  
FastAPI  
 ↓  
Service Layer  
 ↓  
Custom Linear Regression  
 ↓  
Saved Model (.pkl)

---

## 🛡 Engineering Principles

- Separation of concerns
- Stateless API
- Deterministic predictions
- Explicit schemas
- Persistent model artifacts

---

## 📦 Tech Stack

- Python
- FastAPI
- NumPy
- Joblib

---

## 🚧 Future Improvements

- Normal Equation Solver
- L2 Regularization
- Mini-batch Gradient Descent
- Logging & Monitoring
- Docker Support
- Model Versioning
- Authentication

---

## 👤 Author

Abhishek  
Computer Science & Engineering Student

---

## 📄 License

MIT License
