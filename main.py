import tiktoken
import torch
import time
from model.config import SLMConfig
from model.model import GPT
from model.optimised_model import GPT_KV
from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse, ModelType

config=SLMConfig()
untrained_model=GPT(config)
basic_model=GPT(config)
optimised_model=GPT_KV(config)
best_model_params_path = "model/best_model_params.pt"
device="cuda" if torch.cuda.is_available() else "cpu"
basic_model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device)))
optimised_model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device)))
tokenizer=tiktoken.get_encoding("gpt2")
untrained_model.to(device)
basic_model.to(device)
optimised_model.to(device)

app = FastAPI()

def predict(text,model):
    context=(torch.tensor(tokenizer.encode_ordinary(text)).unsqueeze(dim = 0)).to(device)
    start_time=time.time()
    y = model.generate(context, 100)
    time_taken=time.time()-start_time
    return tokenizer.decode(y.squeeze().tolist()), time_taken
    

@app.get("/")
async def root():
    return {"message": "Welcome to the SLM API!, Currently We are running on {device}".format(device=device)}

@app.post("/predict", response_model=PredictResponse)
async def predict_response(request: PredictRequest):
    if request.model_type == ModelType.BASIC:
        result,time=predict(request.input_text, basic_model)
        return PredictResponse(result=result, time=time)
    elif request.model_type == ModelType.CACHED:
        result,time=predict(request.input_text, optimised_model)
        return PredictResponse(result=result, time=time)
    elif request.model_type == ModelType.UNTRAINED:
        result,time=predict(request.input_text, untrained_model)
        return PredictResponse(result=result, time=time)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
    





