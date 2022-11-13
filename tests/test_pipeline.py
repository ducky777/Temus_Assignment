import datetime as dt

import numpy as np
import pytest
import yaml

from model.predictor import WindPowerPredictor


@pytest.fixture(scope="session")
def configs():
    with open("configs/cfg.yml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def pipeline(configs):
    return WindPowerPredictor(configs)


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

def test_wrong_u(pipeline):
    test_data = {
        "date": dt.datetime.strptime("2009070100", "%Y%m%d%H"),
        "u": "should be float",
        "v": 1029,
        "wd": 0.4,
        "ws": 0.00123123,
    }
    with pytest.raises(AssertionError):
        pipeline(test_data)

def test_wrong_v(pipeline):
    test_data = {
        "date": dt.datetime.strptime("2009070100", "%Y%m%d%H"),
        "u": 0.1,
        "v": "should be float",
        "wd": 0.4,
        "ws": 0.0123,
    }
    with pytest.raises(AssertionError):
        pipeline(test_data)

def test_wrong_wd(pipeline):
    test_data = {
        "date": dt.datetime.strptime("2009070100", "%Y%m%d%H"),
        "u": 0.1,
        "v": 0.22,
        "wd": "should be float",
        "ws": 0.0123,
    }
    with pytest.raises(AssertionError):
        pipeline(test_data)

def test_wrong_ws(pipeline):
    test_data = {
        "date": dt.datetime.strptime("2009070100", "%Y%m%d%H"),
        "u": 0.1,
        "v": 0.3,
        "wd": 0.4,
        "ws": "should be float",
    }
    with pytest.raises(AssertionError):
        pipeline(test_data)

def test_correct_predictions_1(pipeline):
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

def test_correct_predictions_2(pipeline):
    data = {
        "date": "2009070800",
        "hors": 1,
        "u": 1.34,
        "v": -0.79,
        "ws": 3.47,
        "wd": 12.68,
    }

    prediction = pipeline(data)[0][0]
    prediction = round(prediction, 4)
    print(prediction)
    assert np.mean(np.square(prediction - 0.1856)) < 0.0001

def test_correct_predictions_3(pipeline):
    data = {
        "date": "2009070400",
        "hors": 1,
        "u": 4.21,
        "v": -1.45,
        "ws": 4.47,
        "wd": 1.68,
    }

    prediction = pipeline(data)[0][0]
    prediction = round(prediction, 4)
    print(prediction)
    assert np.mean(np.square(prediction - 0.3046)) < 0.0001

def test_correct_predictions_4(pipeline):
    data = {
        "date": dt.datetime.strptime("2009070120", "%Y%m%d%H"),
        "hors": 1,
        "u": 2.1,
        "v": 0.6,
        "ws": 2.47,
        "wd": 311.3,
    }

    prediction = pipeline(data)[0][0]
    prediction = round(prediction, 4)
    print(prediction)
    assert np.mean(np.square(prediction - 0.1031)) < 0.0001

def test_correct_predictions_5(pipeline):
    data = {
        "date": dt.datetime.strptime("2009070110", "%Y%m%d%H"),
        "hors": 1,
        "u": 9.34,
        "v": -1.1,
        "ws": 2.17,
        "wd": 9.68,
    }

    prediction = pipeline(data)[0][0]
    prediction = round(prediction, 4)
    print(prediction)
    assert np.mean(np.square(prediction - 0.1934)) < 0.0001
