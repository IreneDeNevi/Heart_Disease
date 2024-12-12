import logging

def get_logger(name='default_logger'):
    """
    Configure and return a shared logger
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Evita di aggiungere più handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('application.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
    return logger