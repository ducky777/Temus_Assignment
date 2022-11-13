import datetime as dt
from typing import Union

import numpy as np
import pandas as pd


class WindPowerData:
    """Class to ensure integrity of data fed
    date: if string, needs to be in `YYYYMMDDHH` format
    """

    schema = {
        "date": (dt.datetime, str),
        "hors": (int, np.integer),
        "u": (float, np.float),
        "v": (float, np.float),
        "ws": (float, np.float),
        "wd": (float, np.float),
    }

    def __val(self, data: Union[pd.DataFrame, pd.Series, dict]):
        """Loops through data to ensure schema is respected"""
        for k, v in self.schema.items():
            assert k in data.keys(), f"{k} not in data!"

            assert isinstance(
                data[k], v
            ), f"Wrong data type for {k}: Found {data[k]} but needs {v}"

            if k == "date":
                data[k] = dt.datetime.strptime("2009070110", "%Y%m%d%H")

        if isinstance(data, pd.DataFrame):
            return np.array(data[["date", "hors", "u", "v", "ws", "wd"]])
        if isinstance(data, pd.Series):
            return np.array(data[["date", "hors", "u", "v", "ws", "wd"]])
        if isinstance(data, dict):
            result = [data[k] for k in self.schema.keys()]
            return np.array(result)

    def validate(self, data: Union[pd.DataFrame, pd.Series, dict]):
        if isinstance(data, (list, np.ndarray)):
            return np.array([self.__val(i) for i in data])
        return self.__val(data)

    def __call__(self, data: Union[pd.DataFrame, pd.Series, dict, list]):
        return self.validate(data)
