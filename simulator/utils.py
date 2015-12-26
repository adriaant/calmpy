# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1103,E0712
from __future__ import unicode_literals
import logging
import os
import errno
import numpy as np
import base64
import json
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


class NumpyEncoder(json.JSONEncoder):
    """From http://stackoverflow.com/a/24375113/244529
       Example usage:
            d = np.arange(100, dtype=np.float)
            dumped = json.dumps(d, cls=NumpyEncoder)
            result = json.loads(dumped, object_hook=json_numpy_obj_hook)
    """

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
           holding dtype, shape and the data, base64 encoded."""
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct
