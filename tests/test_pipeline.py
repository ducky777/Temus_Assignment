import datetime as dt

import numpy as np
import pytest
import yaml
from model.data import WindPowerData
from model.pipeline import WindPowerPipeline
from model.predictor import WindPowerPredictor


@pytest.fixture(scope="session")
def configs():
    with open("configs/cfg.yml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def pipeline(configs):
    stack = [WindPowerData(), WindPowerPredictor(configs)]
    return WindPowerPipeline(stack)


def test_wrong_date_format(pipeline):
    test_data = {
        "date": 91919191,
        "u": "fsdkfjh",
        "v": 1029,
        "wd": -100,
        "ws": 0.00123123,
    }
    with pytest.raises(AssertionError):
        pipeline(test_data)


def test_correct_predictions(pipeline):
    data = {
        "date": dt.datetime.strptime("2009070100", "%Y%m%d%H"),
        "hors": 1,
        "u": 2.34,
        "v": -0.79,
        "ws": 2.47,
        "wd": 108.68,
    }

    prediction = pipeline(data)[0][0]
    prediction = round(prediction, 4)
    assert np.mean(np.square(prediction - 0.0811)) < 0.0001
