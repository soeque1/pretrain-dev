import json
import logging
from logging import config, getLogger


with open('./cfgs/logger.json', "rt") as file:
   config = json.load(file)

logging.config.dictConfig(config)
log = getLogger(__name__)
log.info(config)
