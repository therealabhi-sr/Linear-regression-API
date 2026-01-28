from fastapi import  FastAPI,HTTPException
from ml_api.schemas import PredictRequest,TrainRequest,PredictResponse
from ml_api.service import train_model,predict

app=FastAPI()

@app.post("/train")
def train(req:TrainRequest):
    if(len(req.X)!=len(req.y)):
        raise HTTPException("X and y length not matched")
    train_model(req.X,req.y)  
    return {"Status":"trained"}
    
@app.post("/predict",response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    try:
        result= predict(req.features)
        return {"prediction":result}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))