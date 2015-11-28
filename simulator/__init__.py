# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101
from __future__ import absolute_import, unicode_literals
import logging
import json
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
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
