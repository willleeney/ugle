import os
import logging
from names_generator import generate_name

real_path = os.path.dirname(os.path.realpath(__file__))
ugle_path, _ = os.path.split(real_path)
    

class CustomFormatter(logging.Formatter):
    COLOR_CODE = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m'  # Purple
    }
    RESET_CODE = '\033[0m'  # Reset to default color

    def format(self, record):
        # Get the original formatted message
        original_message = super().format(record)

        # Shorten the file path
        record.pathname = os.path.basename(record.pathname)
        original_message = super().format(record)

        # Apply color to the log level
        log_level = record.levelname
        color_code = self.COLOR_CODE.get(log_level, '')
        colored_message = f"{color_code}{original_message}{self.RESET_CODE}"
        
        return colored_message


def create_logger() -> logging.Logger:
    """
    Creates the logging object with logger with custom colour properties

    Returns:
        log (logging.Logger): custom logger
    """
    
    # Create a file handler (logs will be stored in "logs/{exp_name}.log")
    exp_name = generate_name()
    log_path = f'{ugle_path}/logs/'

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    latest_file_index = len(os.listdir(log_path))
    uid = f'{latest_file_index}-{exp_name}'

    # Set up the logger
    logger = logging.getLogger(uid)
    logger.setLevel(logging.INFO)

    # Create a stream handler (console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_path + uid)
    file_handler.setLevel(logging.DEBUG)

    # Set the custom formatters to the handlers
    color_formatter = CustomFormatter(fmt="[%(name)s--%(asctime)s] %(message)s", datefmt="%d/%m--%I:%M:%S")
    stream_handler.setFormatter(color_formatter)
    file_handler.setFormatter(color_formatter)

    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


log = create_logger()
