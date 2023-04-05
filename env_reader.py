__all__ = ['env']

import configparser
env = configparser.ConfigParser()
env.read('env.ini')