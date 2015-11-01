# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101
from __future__ import unicode_literals
import pytest
import numpy as np
from simulator.tasks import load_network
from .factories import get_fake_random

pytestmark = pytest.mark.django_db


def test_training(network):
    """Trains a simple network with one input module of size 2 connected
       to a CALM module of size 2. See fixture for details."""

    network = load_network('simple')
    network.original_calm = True
    for module in network.modules:
        module.random_func = get_fake_random
    network.train(1, 10)
    weights = network.modules[0].connections[0].weights
    # we use pre-determined values instead of random numbers,
    # so if the outcome is different, then something in the
    # algorithm has changed.
    w1 = str(np.round(weights[0, 0], 8))
    assert w1 == '0.60004046'
    w2 = str(np.round(weights[0, 1], 8))
    assert w2 == '0.60003977'
    w3 = str(np.round(weights[1, 0], 8))
    assert w3 == '0.60004092'
    w4 = str(np.round(weights[1, 1], 8))
    assert w4 == '0.60003948'
