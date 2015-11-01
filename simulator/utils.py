# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1103,E0712
from __future__ import unicode_literals
import logging
import os
import errno
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


def ensure_directory(dir_path):
    """Creates a directory in the local file system under a given root directory"""
    if not dir_path:
        file_dir = settings.DATA_DIR
    else:
        file_dir = os.path.join(settings.DATA_DIR, dir_path)

    try:
        os.makedirs(file_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            logger.critical("Cannot make directory for path {0}!".format(dir_path))
            raise
    return file_dir


def random_val(max, size):
    return np.random.uniform(0.0, max, size=(size, 1))
