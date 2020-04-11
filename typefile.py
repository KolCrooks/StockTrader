from typing import TypedDict

class candle_instance(TypedDict):
    close: float
    datetime: int
    high: float
    low: float
    open: float
    volume: int