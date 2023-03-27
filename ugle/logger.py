import numpy as np
import os
from PrettifyLogging.prettify_logging import PrettifyLogging


real_path = os.path.dirname(os.path.realpath(__file__))
ugle_path, _ = os.path.split(real_path)

def create_logger(colour_in: str = 'blue'):
    """
    creates a logger given a colour to display the logs in
    :param colour_in: colour to display logs in
    :return logger: the loging object
    """
    default = '[%(levelname)s:%(name)s:%(asctime)s] %(funcName)s(): %(message)s'

    pokemon_names = np.load(open(ugle_path + '/data/pokemon_names.npy', 'rb'), allow_pickle=True)
    log_path = 'logs/'

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    latest_file_index = len(os.listdir(log_path))
    poke_name = np.random.choice(pokemon_names)
    uid = f'{latest_file_index}-{poke_name}'
    log_path += uid
    log_config = {
        'name': log_path,
        'level': 'debug',
        'set_utc': False,
        'default_format': default,
        'stream_format': default,
        'file_format': default,
        'info_display': (colour_in, 'bold'),
    }
    logger = PrettifyLogging(**log_config)
    logger.configure()
    logger.log.info(f'init {log_path}')

    return logger

logger = create_logger(colour_in='green')
log = logger.log

