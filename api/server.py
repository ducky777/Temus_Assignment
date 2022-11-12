import yaml
from fastapi import FastAPI
from model.predictor import WindPowerPredictor

app = FastAPI()

with open("configs/cfg.yml", "r") as f:
    configs = yaml.safe_load(f)

predictor = WindPowerPredictor(configs)


@app.post
def predict(data):
    predictions = predictor(data)
    return {"predictions": predictions}
