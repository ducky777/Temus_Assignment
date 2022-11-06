import datetime as dt
from typing import Union

import numpy as np
import pandas as pd


class WindPowerData:
    """Class to ensure integrity of data fed"""

    schema = {
        "date": dt.datetime,
        "hors": (int, np.integer),
        "u": (float, np.float),
        "v": (float, np.float),
        "ws": (float, np.float),
        "wd": (float, np.float),
    }

    def validate(self, data: Union[pd.DataFrame, pd.Series, dict]):
        for k, v in self.schema.items():
            assert k in data.keys(), f"{k} not in data!"
            assert isinstance(
                data[k], v
            ), f"Wrong data type for {k}: Found {data[k]} but needs {v}"

        if isinstance(data, pd.DataFrame):
            return np.array(data[["date", "hors", "u", "v", "ws", "wd"]])
        if isinstance(data, pd.Series):
            return np.array(data[["date", "hors", "u", "v", "ws", "wd"]])
        if isinstance(data, dict):
            result = [data[k] for k in self.schema.keys()]
            return np.array(result)

    def __call__(self, data: Union[pd.DataFrame, pd.Series, dict]):
        return self.validate(data)
