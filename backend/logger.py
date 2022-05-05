import logging

FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

logging.basicConfig(level=logging.DEBUG, filename='log.log', filemode='a', format=FORMAT)

logger = logging.getLogger()