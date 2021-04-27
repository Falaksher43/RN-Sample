
import logging

def get_logger(name, logpath, streaming=True, to_file=False, debug=False):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if to_file:
        file_handler = logging.FileHandler(logpath, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    if streaming:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    return logger 