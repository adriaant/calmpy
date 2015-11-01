# -*-*- encoding: utf-8 -*-*-
# pylint: disable=W0231
from __future__ import unicode_literals
from django.utils.encoding import python_2_unicode_compatible


@python_2_unicode_compatible
class CALMException(Exception):
    """Base class for exceptions in this module."""
    default_detail = "An error occurred."

    def __init__(self, detail=None):
        if detail is not None:
            self.detail = detail
        else:
            self.detail = self.default_detail

    def __str__(self):
        return "{0}: {1}".format(self.__class__.__name__, self.detail)
