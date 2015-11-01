# -*-*- encoding: utf-8 -*-*-
from __future__ import absolute_import, unicode_literals
import os
from .base import *

DEBUG = True
USE_TZ = False  # stop SQLite from complaining
DATA_DIR = os.path.join(BASE_DIR, '../tests/data')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}
