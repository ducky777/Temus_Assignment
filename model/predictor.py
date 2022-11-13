import datetime as dt
import os
from typing import Union

import gdown
import joblib
import numpy as np
import yaml
from tensorflow import keras
from .data import WindPowerData

class WindPowerFiles:
    def __init__(self, configs: Union[str, dict]):
        best_model_url = configs["best_model_url"]
        best_model_path = configs["model_path"]

        minmax_x_scaler_path = configs["minmax_x_scaler_path"]
        minmax_x_scaler_url = configs["minmax_x_scaler_url"]

        minmax_y_scaler_path = configs["minmax_y_scaler_path"]
        minmax_y_scaler_url = configs["minmax_y_scaler_url"]

        self.download(best_model_url, best_model_path)
        self.download(minmax_x_scaler_url, minmax_x_scaler_path)
        self.download(minmax_y_scaler_url, minmax_y_scaler_path)

    def download(self, url: str, path: str):

        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(path):
            gdown.download(url, path)


class WindPowerPredictor(WindPowerFiles, WindPowerData):
    """Pipeline to store and predict Wind Power"""

    def __init__(self, configs: Union[str, dict]):
        """Loads WindPower pipeline with a config file or dict

        args:
          - configs: Union[str, dict]: can be a path to a yaml file or a loaded dict
        """

        self.configs: dict = self.load_configs(configs)

        super().__init__(self.configs)

        self.model: keras.Model = self.load_model(self.configs["model_path"])
        self.minmax_x_scaler: object = joblib.load(self.configs["minmax_x_scaler_path"])
        self.minmax_y_scaler: object = joblib.load(self.configs["minmax_y_scaler_path"])

    def load_configs(self, configs: Union[dict, str]):
        """Loads configs. Can be either a `dict` or `path_to_config_yml`"""
        if isinstance(configs, str):
            with open(configs, "r") as f:
                return yaml.safe_load(f)
        elif isinstance(configs, dict):
            return configs

        raise ValueError(
            f"Config file needs to be a path "
            f"to string or a dict. You input {type(configs)}."
        )

    def load_model(self, model_path: str):
        """Takes in a path to a keras model to load"""
        return keras.models.load_model(model_path)

    def preprocess(
        self,
        date: Union[dt.datetime, str],
        hors: int,
        u: float,
        v: float,
        ws: float,
        wd: float,
    ) -> np.array:

        """Converts date into cosine hour and cosine day of year
        args:
          date: dt.DateTime or str(YYYYMMDDHH)
          hors: number of forward hours for forecasted values
          u: zonal
          v: meridonal
          ws: wind speed
          wd: wind direction

        return:
          [u, v, ws, hour_cos, day_of_year_cos, wd_cos, hors]
        """

        if isinstance(date, str):
            date = dt.datetime.strptime(str(date), "%Y%m%d%H")

        hour_cos = np.cos(2 * np.pi * (date.hour / 24))
        day_of_year_cos = np.cos(2 * np.pi * (date.timetuple().tm_yday / 365))
        wd_cos = np.cos(2 * np.pi * (wd / 360))

        x = np.array([u, v, ws, hour_cos, day_of_year_cos, wd_cos, hors])

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        x = self.minmax_x_scaler.transform(x)

        return np.array(x)

    def predict(
        self,
        x: np.array,
    ) -> np.array:
        """
        date: dt.DateTime or str(YYYYMMDDHH)
        u: zonal
        v: meridonal
        ws: wind speed
        wd: wind direction
        """

        x = self.validate(x)

        if len(x.shape) > 1:
            x = np.array([self.preprocess(*i)[0] for i in x])
        else:
            x = self.preprocess(*x)

        return self.model.predict(x, verbose=0)

    def __call__(self, x: np.array):
        """Returns self.predict()"""
        return self.predict(x)
