# -*-*- encoding: utf-8 -*-*-
from __future__ import absolute_import, unicode_literals
import sys
from .base import *

DEBUG = True
USE_TZ = False  # stop SQLite from complaining
DATA_DIR = '/Users/adriaant/Documents/calmpy'

LOGGING['root'] = {
    'level': 'DEBUG',
    'handlers': ['log_file', 'console'],
}

LOGGING['handlers']['console'] = {
    'level': 'DEBUG',
    'class': 'logutils.colorize.ColorizingStreamHandler',
    'formatter': 'standard',
    'stream': sys.stdout
}
