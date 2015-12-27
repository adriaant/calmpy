# -*-*- encoding: utf-8 -*-*-
from __future__ import absolute_import, unicode_literals
import numpy as np
from django.db import models
from django.utils.encoding import python_2_unicode_compatible
from enumfields import EnumField
from enum import Enum
from django.db.models.signals import post_save


DEFAULT_PARAMETER_VALUES = [
    0.5, -0.2, -10.0, -1.0, -0.6, 0.4,
    1.0, 0.1, 0.6, 0.1, 0.1, 0.05, 1.0,
    0.0, 1.0, 0.0001, 0.005, 0.5, 0.05,
    1.0, 1.0, 50.0, 0.0001
]


@python_2_unicode_compatible
class NetworkDefinition(models.Model):
    """
    Definition of a network, to be edited with jointJS.
    """

    name = models.CharField(max_length=255, null=False, blank=False)
    definition = models.TextField(blank=True)
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)

    def get_parameters(self):
        parameter_hash = {}
        for par in list(self.parameters.all()):
            parameter_hash[par.ptype.name] = np.float64(par.value)
        return parameter_hash

    def __str__(self):
        return self.name


class ParameterType(Enum):
    UP = 0
    DOWN = 1
    CROSS = 2
    FLAT = 3
    HIGH = 4
    LOW = 5
    AE = 6
    ER = 7
    INITWT = 8
    LOWCRIT = 9    # unused
    HIGHCRIT = 10  # min difference in activation between winner and runner up.
    K_A = 11       # activation decay k
    K_Lmax = 12    # maximum value of weight Kmax
    K_Lmin = 13    # minimum value of weight Kmin
    L_L = 14       # learning competition L
    D_L = 15       # base learning rate d
    WMUE_L = 16    # virtual weight between E-node and d
    G_L = 17       # gaussian learning rate horizontal displacement. 0.5 in equation, but configurable for experimenting
    G_W = 18       # width of gaussian learning rate, g.
    F_Bw = 19
    F_Ba = 20
    P_G = 21
    P_S = 22


@python_2_unicode_compatible
class NetworkParameter(models.Model):
    """
    Definition of a parameter.
    """
    network = models.ForeignKey(NetworkDefinition, related_name='parameters')
    ptype = EnumField(ParameterType)
    value = models.FloatField(null=False, blank=False)

    def __str__(self):
        return "{0}".format(self.ptype.name)


def set_default_parameters(sender, instance, **kwargs):
    """For new definitions we need to set some default parameters."""
    if instance.parameters.count() == 0:
        for idx, val in enumerate(DEFAULT_PARAMETER_VALUES):
            NetworkParameter.objects.create(
                network=instance,
                value=val,
                ptype=ParameterType(idx)
            )

# register the signal
post_save.connect(set_default_parameters, sender=NetworkDefinition, dispatch_uid="default_parameters")
