# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101
from __future__ import absolute_import, unicode_literals
import logging
import json
import types
import numpy as np
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
from django.utils import six
from django.utils.six.moves import xrange
from editor.models import NetworkDefinition
from .nnetwork import CALMNetwork

logger = logging.getLogger(__name__)


def load_network(name):
    """Sets up a CALM network by loading the definition
    using the given name and accompanying input patterns."""

    try:
        definition = NetworkDefinition.objects.get(name=name)
    except ObjectDoesNotExist:
        logger.error("No network with name {0}".format(name))
        return
    except MultipleObjectsReturned:
        logger.error("Multiple networks with name {0}".format(name))
        return

    if not definition.definition:
        logger.warning("No definition found for {0}".format(name))
        return

    definition_data = json.loads(definition.definition)
    network = CALMNetwork(name=name, parameters=definition.get_parameters())
    network.build_from_definition(definition_data)
    network.load_patterns()
    return network


def train_with_random(name):
    """Trains network with random values. Assumes one input module of size 2."""

    def train(self, epochs=100, iterations=50):
        """Trains the network for the given amount of epochs and iterations."""

        input_mdl = six.next(six.itervalues(self.inputs))
        patterns = np.random.uniform(0.0, 1.0, size=(epochs, 2))

        for pat in patterns:
            # reset activations
            for mdl in self.modules:
                mdl.reset()

            # set pattern
            input_mdl.r = pat

            # activation flow and weight update
            for _ in xrange(0, iterations):
                # update activations
                for mdl in self.modules:
                    mdl.activate()

                # update weights
                for mdl in self.modules:
                    mdl.change_weights()

                # swap acts
                for mdl in self.modules:
                    mdl.swap_activations()

    network = load_network(name)
    network.train = types.MethodType(train, network)
    return network
