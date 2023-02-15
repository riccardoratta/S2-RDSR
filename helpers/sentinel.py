from enum import Enum


class Bands(Enum):
    m10 = ["02", "03", "04", "08"]
    m20 = ["05", "06", "07", "8A", "11", "12"]
    m60 = ["01", "09"]


class Resolution(Enum):
    SR20 = ["10", "20"]
    SR60 = ["10", "20", "60"]
