import json
import logging
from logging import config, getLogger


def custom_logger(path: str):
   with open(path, "rt") as file:
      config = json.load(file)
   return config
