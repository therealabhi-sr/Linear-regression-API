from ml_api.model import create_model,load_model,save_model
import numpy as np 

def train_model(X,y):
    X=np.array(X)
    y=np.array(y)
    model=create_model()
    model.fit(X,y)
    save_model(model)
    return model

def predict(features):
    model=load_model()
    if model is None:
        raise Exception("Model Not Found")
    x=np.array(features).reshape(1,-1)
    return float(model.predict(x)[0])