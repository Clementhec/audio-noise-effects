from enum import Enum


class Mode(str, Enum):
    dev = "dev"
    prod = "prod"